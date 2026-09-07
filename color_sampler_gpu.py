"""
GPU-resident re-implementation of the per-batch color-sampler hot path.

Motivation: ColorBatchSampler (color_sampler.py) makes its per-batch decisions
(sample_batch / update_node_indptr_direct) via ColorSamplerCore.ColoringSampler, a
CPU/OpenMP C++ extension using up to num_workers * num_threads_per_worker (64 by
default) threads every single batch. GPUColorBatchSampler replaces just that hot
path with vectorized PyTorch tensor ops on GPU-resident CSR tensors, eliminating
that OpenMP thread pool from the per-batch loop entirely.

Scope (see plan): only the freeze_any=True / node_stable_mode=True path is ported.
Every config under config/adapt_exp/ sets freeze_any: true, which is what routes
sample_batch() into the simple branch this class implements: Python pre-filters
root_nodes down to the unstable ones, then for each unstable node the decision is
a single index-offset gather (current_node_color_ptrs[node] + num_colors - 1) into
that node's sorted event list, reduced via a plain min() across nodes -- no
per-event stability scanning, no early-exit walk. The other branch
(node_stable_mode=False, a genuinely sequential per-node scan) is not exercised by
any current config and is intentionally not ported here; calling sample_batch()
without first enabling node_stable_mode raises NotImplementedError rather than
silently doing the wrong thing.

color_graph() itself stays on the CPU/C++ side (ColoringSampler.color_graph) -- it's
cheap and infrequent (once per epoch, or once per chunk with caching in the *_large
mode), so there's no benefit to porting its ragged-set-construction logic to GPU.
This module's job is only to mirror that ragged output into GPU CSR tensors once
per color_graph()/cache-load call, then serve every per-batch sample_batch() /
update_node_indptr_direct() call off those tensors.
"""
import threading

import numpy as np
import torch
import ColorSamplerCore


def _csr_offsets(ragged, device):
    lengths = np.fromiter((len(x) for x in ragged), dtype=np.int64, count=len(ragged))
    offsets = np.zeros(len(ragged) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return torch.from_numpy(offsets).to(device)


def _csr_flat(ragged, device):
    non_empty = [np.asarray(x, dtype=np.int32) for x in ragged if len(x) > 0]
    if not non_empty:
        return torch.empty(0, dtype=torch.int32, device=device)
    flat = np.concatenate(non_empty)
    return torch.from_numpy(flat).to(device)


class GPUColorBatchSampler():
    """
    Drop-in replacement for ColorBatchSampler's public interface (sample_batch,
    update_node_indptr_direct, update_node_stable_flag, reset, color_graph, ...)
    that keeps the per-batch color-sampler state on GPU instead of calling into
    the CPU/OpenMP C++ extension for every batch.
    """

    def __init__(self,
                 indptr,
                 edge_index,
                 indices,
                 eid,
                 num_nodes,
                 num_edges,
                 num_colors=500,
                 num_hops=2,
                 num_recent_edges=4,
                 num_workers=8,
                 num_threads_per_worker=8,
                 num_nodes_per_thread=1,
                 decay_type="disable",
                 decay_step=5,
                 decay_factor=5,
                 minimum_scale_factor=0.5,
                 use_full_edge=False,
                 device="cuda"):
        self.indptr = indptr
        self.edge_index = edge_index
        self.eid = eid
        self.indices = indices

        self.num_nodes = num_nodes
        self.num_edges = num_edges

        self.num_colors = num_colors
        self.num_hops = num_hops
        self.num_recent_edges = num_recent_edges

        self.num_workers = num_workers
        self.num_threads_per_worker = num_threads_per_worker

        self.decay_type = decay_type
        self.decay_step = decay_step
        self.decay_factor = decay_factor

        self.color_bottom = np.floor(num_colors * minimum_scale_factor).astype(np.int32)
        self.related_nodes = None
        self.use_full_edge = use_full_edge
        self.device = torch.device(device)

        # color_graph() itself is kept on the CPU/C++ side -- see module docstring.
        # This C++ object is reused only for color_graph / reset_color_table /
        # get_usage_table / get_update_table / set_usage_table / set_update_table /
        # enable_multi_color / disable_full_edges. Its own current_node_color_ptrs /
        # current_node_self_update_ptrs are never read once GPU tables are built --
        # this class tracks pointer state itself, on GPU.
        self.sampler = ColorSamplerCore.ColoringSampler(indptr,
                                                          edge_index,
                                                          indices,
                                                          eid,
                                                          num_nodes,
                                                          num_hops, num_recent_edges,
                                                          num_threads_per_worker, num_workers, num_nodes_per_thread)
        if not self.use_full_edge:
            self.sampler.disable_full_edges()

        self.prev_node_stable_flag = torch.zeros(num_nodes - 1, dtype=torch.bool, device=self.device)
        self.node_stable_flag = torch.zeros(num_nodes - 1, dtype=torch.bool, device=self.device)
        self.use_memory = False
        self.node_stable_mode = False
        self.batch_index_list = []
        self.batch_index = 0

        # GPU CSR tables -- populated by _build_gpu_tables(), called from
        # color_graph() and from load_multi_color_table() (see
        # GPUMultiColorBatchSampler). None until the first color_graph() call.
        self.usage_offsets = None
        self.usage_flat = None
        self.usage_flat_keys = None
        self.self_update_offsets = None
        self.self_update_flat = None
        self.self_update_flat_keys = None
        self.max_eid = None
        self.large_const = None
        self.current_node_color_ptrs = torch.zeros(num_nodes, dtype=torch.int64, device=self.device)
        self.current_node_self_update_ptrs = torch.zeros(num_nodes, dtype=torch.int64, device=self.device)

        # Same rationale as ColorBatchSampler._state_lock: sample_batch() runs on
        # the prefetch producer thread, update_node_indptr_direct()/
        # update_node_stable_flag() run on the consumer thread. This lock makes
        # each individual call atomic.
        self._state_lock = threading.Lock()

    # ------------------------------------------------------------------
    # GPU table construction
    # ------------------------------------------------------------------

    def _build_gpu_tables(self):
        usage_table = self.sampler.get_usage_table()
        update_table = self.sampler.get_update_table()

        self.usage_offsets = _csr_offsets(usage_table, self.device)
        self.usage_flat = _csr_flat(usage_table, self.device)
        self.self_update_offsets = _csr_offsets(update_table, self.device)
        self.self_update_flat = _csr_flat(update_table, self.device)

        max_eid = self.num_edges
        if self.usage_flat.numel() > 0:
            max_eid = max(max_eid, int(self.usage_flat.max().item()))
        if self.self_update_flat.numel() > 0:
            max_eid = max(max_eid, int(self.self_update_flat.max().item()))
        assert max_eid < 2**31 - 1, (
            f"max event id {max_eid} exceeds int32 range; GPUColorBatchSampler's CSR "
            "tables are stored as int32 -- widen the storage dtype before using this "
            "dataset with the GPU color sampler."
        )
        self.max_eid = max_eid
        self.large_const = max_eid + 1

        self.usage_flat_keys = self._flat_keys(self.usage_offsets, self.usage_flat)
        self.self_update_flat_keys = self._flat_keys(self.self_update_offsets, self.self_update_flat)

    def _flat_keys(self, offsets, flat):
        if flat.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        lengths = (offsets[1:] - offsets[:-1]).to(torch.int64)
        node_id_of = torch.repeat_interleave(
            torch.arange(self.num_nodes, device=self.device, dtype=torch.int64), lengths)
        return node_id_of * self.large_const + flat.to(torch.int64)

    # ------------------------------------------------------------------
    # Public interface (mirrors ColorBatchSampler)
    # ------------------------------------------------------------------

    def reset(self):
        self.reset_node_indptr()
        self.record_node_stable_flag()
        self.reset_node_stable_flag()
        if not self.use_memory:
            self.reset_batch_index_list()

    def reset_node_stable_flag(self):
        self.node_stable_flag = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)

    def record_node_stable_flag(self, new_stable_flag=None):
        with torch.no_grad():
            if new_stable_flag is not None:
                self.node_stable_flag = new_stable_flag.to(self.device)
            else:
                self.prev_node_stable_flag = self.node_stable_flag

    def get_node_stable_flag(self):
        return self.node_stable_flag

    def set_node_stable_mode(self, node_stable_mode):
        self.node_stable_mode = node_stable_mode

    def reset_batch_index_list(self):
        self.batch_index_list = []
        self.batch_index = 0

    def update_node_stable_flag(self, node_stable_flag, root_nodes=None):
        with self._state_lock:
            with torch.no_grad():
                if root_nodes is not None:
                    index = root_nodes.long().to(self.device)
                    self.node_stable_flag[index] = node_stable_flag.to(self.device)
                else:
                    self.node_stable_flag = node_stable_flag.to(self.device)

    def set_use_memory(self, use_memory):
        # Matches ColorBatchSampler: this feature is disabled upstream too.
        print("use memory disabled")
        self.use_memory = False
        return False

    def check_if_use_memory(self, edge_index, recent_node_stable_flag):
        print("use memory disabled")
        self.use_memory = False
        return False

    def color_decay(self, batch_index):
        if self.decay_type == "disable":
            color_limit = self.num_colors
        elif self.decay_type == "linear":
            color_limit = max(self.num_colors - (batch_index // self.decay_step) * (self.num_colors // self.decay_factor), self.color_bottom)
        elif self.decay_type == "log":
            color_limit = max(self.num_colors - np.log2(batch_index // self.decay_step + 1).astype(np.int32) * (self.num_colors // self.decay_factor), self.color_bottom)
        else:
            color_limit = self.num_colors
        return color_limit

    def color_graph(self, end_event_id=-1, st_event_id=-1):
        self.sampler.color_graph(st_event_id, end_event_id)
        self._build_gpu_tables()

    def reset_color_graph(self):
        self.sampler.reset_color_table()

    def reset_node_indptr(self):
        self.current_node_color_ptrs.zero_()
        self.current_node_self_update_ptrs.zero_()

    def update_node_indptr(self, recent_event_id, root_nodes):
        with self._state_lock:
            if self.related_nodes is not None:
                root_nodes = self.related_nodes
                self._advance_pointers(recent_event_id, root_nodes)
                self.related_nodes = None
            else:
                self._advance_pointers(recent_event_id, root_nodes)

    def update_node_indptr_direct(self, recent_event_id, related_nodes):
        with self._state_lock:
            self._advance_pointers(recent_event_id, related_nodes)

    def _advance_pointers(self, final_event, related_nodes):
        """
        GPU re-expression of ColoringSampler::update_node_color_ptrs
        (color_sampler_core.cpp): advance current_node_self_update_ptrs[node] to
        the first self-update event >= final_event, then set
        current_node_color_ptrs[node] to the first usage-table event strictly
        greater than that self-update event (clamped to the end of the list).

        torch.searchsorted can't batch a ragged per-node sorted sequence directly
        (its batched form needs equal-length rows), so this uses the flat
        key-offset trick: each node's CSR segment is already sorted and segments
        are concatenated in node order, so combining
        key = node_id * large_const + event_id makes the *entire* flat array
        monotonically increasing (since event_id < large_const always), and a
        single flat searchsorted resolves every node's local position at once
        without leaking across node boundaries.
        """
        if isinstance(related_nodes, np.ndarray):
            nodes = torch.from_numpy(related_nodes.astype(np.int64)).to(self.device)
        else:
            nodes = related_nodes.to(self.device).to(torch.int64)
        if nodes.numel() == 0:
            return

        su_offsets = self.self_update_offsets[nodes]
        su_len = self.self_update_offsets[nodes + 1] - su_offsets

        final_event_keys = nodes * self.large_const + int(final_event)
        flat_pos_su = torch.searchsorted(self.self_update_flat_keys, final_event_keys, side='left')
        local_su_ptr = flat_pos_su - su_offsets
        no_more_update = local_su_ptr >= su_len

        usage_offsets_n = self.usage_offsets[nodes]
        usage_len = self.usage_offsets[nodes + 1] - usage_offsets_n

        if self.self_update_flat.numel() == 0:
            self_update_event = torch.zeros_like(nodes)
        else:
            safe_flat_pos_su = torch.clamp(flat_pos_su, max=self.self_update_flat.numel() - 1)
            self_update_event = self.self_update_flat[safe_flat_pos_su].to(torch.int64)

        event_keys_2 = nodes * self.large_const + self_update_event
        flat_pos_color = torch.searchsorted(self.usage_flat_keys, event_keys_2, side='right')
        local_color_ptr = flat_pos_color - usage_offsets_n
        local_color_ptr = torch.where(no_more_update, usage_len, local_color_ptr)
        local_color_ptr = torch.minimum(local_color_ptr, usage_len)

        local_su_ptr = torch.minimum(local_su_ptr, su_len)

        self.current_node_self_update_ptrs[nodes] = local_su_ptr
        self.current_node_color_ptrs[nodes] = local_color_ptr

    def sample_batch(self,
                      train_df,
                      start_event_id,
                      batch_index,
                      minimal_batch_size=1000,
                      step_size=8000):
        if not self.node_stable_mode:
            raise NotImplementedError(
                "GPUColorBatchSampler only supports the freeze_any/node_stable_mode=True "
                "path (the simple index-offset-gather branch); the node_stable=False scan "
                "branch was not ported since no current config exercises it. Call "
                "set_node_stable_mode(True) (i.e. run with freeze_any: true) before using "
                "the GPU color sampler."
            )

        checked_df = train_df.loc[start_event_id:start_event_id + step_size]
        root_nodes = np.unique(np.concatenate([checked_df['src'].values, checked_df['dst'].values])).astype(np.int32)

        end_edge_id = min(start_event_id + step_size, self.num_edges)
        minimal_batch_end_edge_id = min(start_event_id + minimal_batch_size, self.num_edges)

        with self._state_lock:
            self.related_nodes = root_nodes

            if self.use_memory:
                final_event = self.batch_index_list[batch_index]
                return final_event, root_nodes

            root_nodes_t = torch.from_numpy(root_nodes.astype(np.int64)).to(self.device)
            node_stable = self.node_stable_flag[root_nodes_t]
            unstable_nodes = root_nodes_t[~node_stable]

            num_colors = self.color_decay(batch_index)

            if unstable_nodes.numel() == 0:
                final_event = end_edge_id
            else:
                node_len = self.usage_offsets[unstable_nodes + 1] - self.usage_offsets[unstable_nodes]
                ptr = self.current_node_color_ptrs[unstable_nodes]
                idx_in_node = ptr + int(num_colors) - 1
                valid = (idx_in_node >= 0) & (idx_in_node < node_len)
                # torch.clamp doesn't accept a scalar min together with a tensor max in
                # one call -- split into two single-bound clamps instead.
                clamped_idx = torch.clamp(idx_in_node, min=0)
                clamped_idx = torch.clamp(clamped_idx, max=torch.clamp(node_len - 1, min=0))
                flat_idx = self.usage_offsets[unstable_nodes] + clamped_idx
                candidates = self.usage_flat[flat_idx].to(torch.int64)
                candidates = torch.where(valid, candidates, torch.full_like(candidates, end_edge_id))
                final_event = end_edge_id if candidates.numel() == 0 else min(end_edge_id, int(candidates.min().item()))

            final_event = max(final_event, minimal_batch_end_edge_id)
            self.batch_index_list.append(final_event)

        return final_event, root_nodes


class GPUMultiColorBatchSampler(GPUColorBatchSampler):
    """
    Mirrors MultiColorBatchSampler: adds the chunk-level cache/pickle layer used
    by batch_stable_freezing_large. Caching still round-trips through the CPU
    C++ object's ragged tables (get_usage_table/get_update_table/set_usage_table/
    set_update_table) exactly as ColorBatchSampler does -- only the per-batch hot
    path (sample_batch / update_node_indptr_direct) differs. Restoring a cached
    chunk rebuilds the GPU CSR tensors from the restored ragged tables.
    """

    def __init__(self,
                 indptr,
                 edge_index,
                 indices,
                 eid,
                 num_nodes,
                 num_edges,
                 num_colors=500,
                 num_hops=2,
                 num_recent_edges=4,
                 num_workers=8,
                 num_threads_per_worker=8,
                 num_nodes_per_thread=1,
                 decay_type="disable",
                 decay_step=5,
                 decay_factor=5,
                 minimum_scale_factor=0.5,
                 chunk_num=1,
                 cache_dir="sampler_caches",
                 enable_pickle=False,
                 use_full_edge=False,
                 device="cuda"):
        super(GPUMultiColorBatchSampler, self).__init__(indptr,
                                                          edge_index,
                                                          indices,
                                                          eid,
                                                          num_nodes,
                                                          num_edges,
                                                          num_colors,
                                                          num_hops=num_hops,
                                                          num_recent_edges=num_recent_edges,
                                                          num_workers=num_workers,
                                                          num_threads_per_worker=num_threads_per_worker,
                                                          num_nodes_per_thread=num_nodes_per_thread,
                                                          decay_type=decay_type,
                                                          decay_step=decay_step,
                                                          decay_factor=decay_factor,
                                                          minimum_scale_factor=minimum_scale_factor,
                                                          use_full_edge=use_full_edge,
                                                          device=device)
        self.sampler.enable_multi_color()
        self.chunk_num = chunk_num
        self.multi_update_table = dict()
        self.multi_usage_table = dict()
        self.enable_pickle = enable_pickle
        self.cache_dir = cache_dir

    def disable_pickle(self):
        self.enable_pickle = False

    def cache_multi_color_table(self, chunk_id):
        import pickle
        if self.enable_pickle:
            with open(self.cache_dir + "/multi_color_usage_table_" + str(chunk_id) + ".pkl", "wb") as f:
                pickle.dump(self.sampler.get_usage_table(), f)
            with open(self.cache_dir + "/multi_color_update_table_" + str(chunk_id) + ".pkl", "wb") as f:
                pickle.dump(self.sampler.get_update_table(), f)
        else:
            self.multi_update_table[chunk_id] = self.sampler.get_usage_table()
            self.multi_usage_table[chunk_id] = self.sampler.get_update_table()

    def load_multi_color_table(self, chunk_id):
        import pickle
        import os
        if self.enable_pickle:
            with open(self.cache_dir + "/multi_color_usage_table_" + str(chunk_id) + ".pkl", "rb") as f:
                self.sampler.set_usage_table(pickle.load(f))
            with open(self.cache_dir + "/multi_color_update_table_" + str(chunk_id) + ".pkl", "rb") as f:
                self.sampler.set_update_table(pickle.load(f))
        else:
            self.sampler.set_usage_table(self.multi_update_table[chunk_id])
            self.sampler.set_update_table(self.multi_usage_table[chunk_id])
        self._build_gpu_tables()

    def is_cached(self, chunk_id):
        import os
        if self.enable_pickle:
            return os.path.exists(self.cache_dir + "/multi_color_usage_table_" + str(chunk_id) + ".pkl") and os.path.exists(self.cache_dir + "/multi_color_update_table_" + str(chunk_id) + ".pkl")
        else:
            return chunk_id in self.multi_update_table and chunk_id in self.multi_usage_table
