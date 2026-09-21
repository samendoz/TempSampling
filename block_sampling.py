"""
Block-level superset sampling: sample the union of several consecutive
training iterations' root nodes in one ParallelSampler.sample_union() call,
then slice each iteration's own standalone sub-batch back out of the raw
result -- before any dgl.Block is ever constructed.

Why slice the RAW TemporalGraphBlock arrays (row/col/eid/ts/dts/nodes) instead
of slicing an already-built dgl.Block: dgl.create_block may reorder/canonicalize
edges internally, so there's no guarantee an already-built block's internal
tensors stay in the same order as the COO input that constructed it. Slicing
the raw arrays first and feeding each slice through the existing, unmodified
utils.to_dgl_blocks()/prepare_input() sidesteps that risk entirely -- each
slice goes through the exact same code path a standalone single-iteration call
already goes through today, just fed pre-sliced data instead of a fresh C++
sampler call's data.

Scope: matches sample_union's scope -- num_layers==1, num_history==1,
window_duration==0 (TGN/APAN, the only configs this project trains). A block's
"mfg" is always mfgs == [[block]].
"""
import threading
import queue

import numpy as np
import torch
from time import perf_counter

from utils import to_dgl_blocks, prepare_input


class _SlicedTemporalGraphBlock:
    """
    Exposes the same method-call interface as the pybind11-wrapped
    sampler_core.TemporalGraphBlock (.row()/.col()/.eid()/.ts()/.dts()/
    .nodes()/.dim_in()/.dim_out()), backed by plain numpy arrays, so
    utils.to_dgl_blocks() can consume it completely unmodified.
    """
    __slots__ = ('_row', '_col', '_eid', '_ts', '_dts', '_nodes', '_dim_in', '_dim_out')

    def __init__(self, row, col, eid, ts, dts, nodes, dim_in, dim_out):
        self._row = row
        self._col = col
        self._eid = eid
        self._ts = ts
        self._dts = dts
        self._nodes = nodes
        self._dim_in = dim_in
        self._dim_out = dim_out

    def row(self): return self._row
    def col(self): return self._col
    def eid(self): return self._eid
    def ts(self): return self._ts
    def dts(self): return self._dts
    def nodes(self): return self._nodes
    def dim_in(self): return self._dim_in
    def dim_out(self): return self._dim_out


def slice_union_ret(union_ret0, iteration_root_offsets):
    """
    union_ret0: the single TemporalGraphBlock from sample_union's ret[0]
        (pybind11 object -- .row()/.col()/etc. are methods).
    iteration_root_offsets: [(off_0, n_0), (off_1, n_1), ...] -- root-position
        ranges per iteration within the union call's concatenated root_nodes
        array, in concatenation order.

    Returns: list of K _SlicedTemporalGraphBlock, each equivalent to what a
    standalone sample() call for that iteration's root_nodes/ts alone would
    have produced (same node/edge content; not guaranteed identical internal
    ordering, since union sampling processes occurrences grouped by node, not
    in the standalone per-iteration array order).
    """
    row = np.asarray(union_ret0.row())
    col = np.asarray(union_ret0.col())
    eid = np.asarray(union_ret0.eid())
    ts = np.asarray(union_ret0.ts())
    dts = np.asarray(union_ret0.dts())
    nodes = np.asarray(union_ret0.nodes())
    num_root_total = sum(n for _, n in iteration_root_offsets)

    slices = []
    for off_i, n_i in iteration_root_offsets:
        edge_mask = (row >= off_i) & (row < off_i + n_i)
        sub_row = (row[edge_mask] - off_i).astype(row.dtype)
        sub_eid = eid[edge_mask]
        # col[edge_mask] are GLOBAL indices into the union's nodes/ts/dts
        # arrays (already offset by num_root_total in sample_union's
        # combine_coo_union) -- gather this iteration's own neighbor
        # (node_id, ts, dts) directly via them, one fresh local slot per edge
        # (no dedup -- matches add_neighbor's own one-slot-per-edge behavior,
        # the same as what a standalone call produces before prepare_input's
        # separate, optional combine_first dedup step).
        neighbor_global_idx = col[edge_mask]
        sub_nodes_suffix = nodes[neighbor_global_idx]
        sub_ts_suffix = ts[neighbor_global_idx]
        sub_dts_suffix = dts[neighbor_global_idx]

        sub_nodes_prefix = nodes[off_i:off_i + n_i]
        sub_ts_prefix = ts[off_i:off_i + n_i]

        sub_nodes = np.concatenate([sub_nodes_prefix, sub_nodes_suffix])
        sub_ts = np.concatenate([sub_ts_prefix, sub_ts_suffix])
        sub_dts = np.concatenate([np.zeros(n_i, dtype=dts.dtype), sub_dts_suffix])
        sub_col = (n_i + np.arange(len(sub_row))).astype(col.dtype)

        slices.append(_SlicedTemporalGraphBlock(
            row=sub_row, col=sub_col, eid=sub_eid, ts=sub_ts, dts=sub_dts,
            nodes=sub_nodes, dim_in=len(sub_nodes), dim_out=n_i,
        ))
    return slices


class BlockPrefetchProducer:
    """
    Alternative to PrefetchProducer (prefetch_pipeline.py): instead of one
    sampler.sample() + to_dgl_blocks() + prepare_input() cycle per training
    iteration, groups block_size consecutive iterations, runs ONE
    sampler.sample_union() call for their concatenated root nodes, slices the
    result back into block_size standalone per-iteration sub-batches (via
    slice_union_ret), and queues them exactly as PrefetchProducer would --
    same item shape, same get_stats() keys -- so train_simple_v2.py's consumer
    loop needs zero changes to use this instead.
    """

    def __init__(self, sampler, sample_param, gnn_param, node_feats, edge_feats,
                 combine_first=False, all_gpu=True, queue_size=2, use_nogil=False,
                 block_size=4):
        if sample_param['layer'] != 1 or sample_param['history'] != 1:
            raise NotImplementedError(
                "BlockPrefetchProducer only supports layer==1, history==1 "
                "(matching sample_union's scope -- TGN/APAN configs)."
            )
        self.sampler = sampler
        self.sample_param = sample_param
        self.gnn_param = gnn_param
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.combine_first = combine_first
        self.all_gpu = all_gpu
        self.block_size = block_size
        self._sample_union_fn_name = 'sample_union_nogil' if use_nogil else 'sample_union'

        self.queue = queue.Queue(maxsize=queue_size)
        self.thread = None
        self.stopped = False

        self.stats_lock = threading.Lock()
        self.stats = {
            'sampling_time': 0.0,
            'to_dgl_blocks_time': 0.0,
            'prepare_input_time': 0.0,
            'queue_put_wait_time': 0.0,
            'batches_produced': 0,
            'blocks_produced': 0,
        }

    def get_stats(self):
        with self.stats_lock:
            return dict(self.stats)

    def reset_stats(self):
        with self.stats_lock:
            for k in self.stats:
                self.stats[k] = 0.0 if isinstance(self.stats[k], float) else 0

    def _producer_loop(self, batch_generator):
        gen = iter(batch_generator)
        while True:
            if self.stopped:
                return

            block = []
            for _ in range(self.block_size):
                if self.stopped:
                    return
                try:
                    block.append(next(gen))
                except StopIteration:
                    break
            if not block:
                break

            # block[i] = (rows, root_nodes, ts, ptr_end, unique_pos_root_nodes, related_nodes)
            union_root_nodes = np.concatenate([b[1] for b in block])
            union_ts = np.concatenate([b[2] for b in block])
            offsets = []
            pos = 0
            for b in block:
                n_i = len(b[1])
                offsets.append((pos, n_i))
                pos += n_i

            sample_fn = getattr(self.sampler, self._sample_union_fn_name)
            t0 = perf_counter()
            sample_fn(union_root_nodes, union_ts)
            union_ret0 = self.sampler.get_ret()[0]
            with self.stats_lock:
                self.stats['sampling_time'] += perf_counter() - t0
                self.stats['blocks_produced'] += 1

            sliced = slice_union_ret(union_ret0, offsets)

            for (rows, root_nodes, ts, ptr_end, unique_pos_root_nodes, related_nodes), sub_ret0 in zip(block, sliced):
                if self.stopped:
                    return
                t0 = perf_counter()
                mfgs = to_dgl_blocks([sub_ret0], self.sample_param['history'], cuda=self.all_gpu)
                with self.stats_lock:
                    self.stats['to_dgl_blocks_time'] += perf_counter() - t0

                t0 = perf_counter()
                mfgs = prepare_input(mfgs, self.node_feats, self.edge_feats, combine_first=self.combine_first)
                with self.stats_lock:
                    self.stats['prepare_input_time'] += perf_counter() - t0
                    self.stats['batches_produced'] += 1

                item = {
                    'rows': rows,
                    'root_nodes': root_nodes,
                    'ts': ts,
                    'ret': [sub_ret0],
                    'mfgs': mfgs,
                    'ptr_end': ptr_end,
                    'unique_pos_root_nodes': unique_pos_root_nodes,
                    'related_nodes': related_nodes,
                }
                t0 = perf_counter()
                self.queue.put(item)
                with self.stats_lock:
                    self.stats['queue_put_wait_time'] += perf_counter() - t0

        self.queue.put(None)

    def start(self, batch_generator):
        self.stopped = False
        self.thread = threading.Thread(target=self._producer_loop, args=(batch_generator,), daemon=True)
        self.thread.start()

    def get_next(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
