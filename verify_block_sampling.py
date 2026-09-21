"""
Correctness test for block_sampling.py: confirms the union-then-slice path
(sample_union + slice_union_ret + to_dgl_blocks + prepare_input, run once per
block) reproduces exactly the same content as today's path (one sampler.sample()
+ to_dgl_blocks + prepare_input call per iteration, run sequentially) for a
block of K=4 iterations.

Not literal tensor equality -- union sampling processes root-node occurrences
grouped by node id, not in the standalone per-iteration array order, so the
neighbor SUFFIX of a block can come out in a different order (though a given
iteration's own root/dst-node prefix is always emitted in its original,
untouched order by both paths). Compared as sets/sorted sequences of
(node_id, ts) instead.

Requires the compiled sampler_core extension and torch. Run on the target
machine:

    python verify_block_sampling.py
"""
import numpy as np
import torch

from sampler_core import ParallelSampler
from utils import to_dgl_blocks, prepare_input
from block_sampling import slice_union_ret
from verify_sample_union import build_synthetic_graph, make_sampler, NUM_NEIGHBORS

BLOCK_SIZE = 4
FEAT_DIM = 8


def make_iterations(num_nodes, num_events, block_size, seed):
    """
    Simulate block_size consecutive training iterations' (root_nodes, ts)
    pairs: increasing timestamps across iterations (matching the real
    pipeline's sequential batch-boundary advancement), with deliberate node
    overlap both within and across iterations to stress the union path.
    """
    rng = np.random.RandomState(seed)
    iterations = []
    t = 0.0
    for _ in range(block_size):
        n = rng.randint(6, 12)
        root_nodes = rng.randint(0, num_nodes, size=n).astype(np.int32)
        ts = np.sort(rng.uniform(t, t + 20, size=n)).astype(np.float32)
        t = float(ts[-1]) + 1.0
        iterations.append((root_nodes, ts))
    return iterations


def mfg_content(mfg):
    """(root_ids_in_order, root_ts_in_order, sorted[(neighbor_id, neighbor_ts)], sorted[(eid, dt)])"""
    b = mfg[0][0]
    num_dst = b.num_dst_nodes()
    ids = b.srcdata['ID'].cpu().numpy()
    ts = b.srcdata['ts'].cpu().numpy()
    root_ids = ids[:num_dst].tolist()
    root_ts = ts[:num_dst].tolist()
    neighbor_pairs = sorted(zip(ids[num_dst:].tolist(), ts[num_dst:].tolist()))
    eids = b.edata['ID'].cpu().numpy() if 'ID' in b.edata else np.array([])
    dts = b.edata['dt'].cpu().numpy() if 'dt' in b.edata else np.array([])
    edge_pairs = sorted(zip(eids.tolist(), np.round(dts, 4).tolist()))
    return root_ids, root_ts, neighbor_pairs, edge_pairs


def main():
    indptr, indices, eidarr, ets, num_events = build_synthetic_graph(num_nodes=40, num_events=800, seed=2)
    iterations = make_iterations(num_nodes=40, num_events=num_events, block_size=BLOCK_SIZE, seed=3)

    node_feats = torch.randn(40, FEAT_DIM)
    edge_feats = torch.randn(num_events, FEAT_DIM)

    sample_param = {'history': 1}

    # --- Reference: one sampler, BLOCK_SIZE sequential sample() calls ---
    ref_sampler = make_sampler(indptr, indices, eidarr, ets)
    ref_mfgs = []
    for root_nodes, ts in iterations:
        ref_sampler.sample(root_nodes, ts)
        ret = ref_sampler.get_ret()
        mfg = to_dgl_blocks(ret, sample_param['history'], cuda=False)
        mfg = prepare_input(mfg, node_feats, edge_feats, combine_first=False)
        ref_mfgs.append(mfg)

    # --- New: fresh sampler, ONE sample_union() call for the whole block ---
    union_sampler = make_sampler(indptr, indices, eidarr, ets)
    union_root_nodes = np.concatenate([r for r, _ in iterations])
    union_ts = np.concatenate([t for _, t in iterations])
    union_sampler.sample_union(union_root_nodes, union_ts)
    union_ret0 = union_sampler.get_ret()[0]

    offsets = []
    pos = 0
    for root_nodes, _ in iterations:
        offsets.append((pos, len(root_nodes)))
        pos += len(root_nodes)

    sliced = slice_union_ret(union_ret0, offsets)
    new_mfgs = []
    for sub_ret0 in sliced:
        mfg = to_dgl_blocks([sub_ret0], sample_param['history'], cuda=False)
        mfg = prepare_input(mfg, node_feats, edge_feats, combine_first=False)
        new_mfgs.append(mfg)

    # --- Compare ---
    for i in range(BLOCK_SIZE):
        ref_root_ids, ref_root_ts, ref_nbrs, ref_edges = mfg_content(ref_mfgs[i])
        new_root_ids, new_root_ts, new_nbrs, new_edges = mfg_content(new_mfgs[i])

        assert ref_root_ids == new_root_ids, (
            f"[iter {i}] root/dst node IDs differ (order matters here):\n"
            f"  ref={ref_root_ids}\n  new={new_root_ids}")
        assert ref_root_ts == new_root_ts, f"[iter {i}] root/dst timestamps differ"
        assert ref_nbrs == new_nbrs, (
            f"[iter {i}] neighbor (id, ts) content differs:\n"
            f"  ref={ref_nbrs}\n  new={new_nbrs}")
        assert ref_edges == new_edges, f"[iter {i}] edge (eid, dt) content differs"

        # Since node-ID sets match, and prepare_input's feature gather is a
        # pure deterministic function of the IDs, confirm the actual gathered
        # feature tensors agree too (order-independent: compare sorted by ID).
        ref_b, new_b = ref_mfgs[i][0][0], new_mfgs[i][0][0]
        ref_order = torch.argsort(ref_b.srcdata['ID'])
        new_order = torch.argsort(new_b.srcdata['ID'])
        assert torch.allclose(ref_b.srcdata['h'][ref_order], new_b.srcdata['h'][new_order]), (
            f"[iter {i}] gathered node features differ after sorting by ID")
        if 'f' in ref_b.edata:
            ref_eorder = torch.argsort(ref_b.edata['ID'])
            new_eorder = torch.argsort(new_b.edata['ID'])
            assert torch.allclose(ref_b.edata['f'][ref_eorder], new_b.edata['f'][new_eorder]), (
                f"[iter {i}] gathered edge features differ after sorting by ID")

        print(f"iter {i}: OK ({len(ref_root_ids)} root nodes, {len(ref_nbrs)} neighbor entries, "
              f"{len(ref_edges)} edges)")

    print("\nAll iterations match: union-then-slice reproduces the per-iteration "
          "reference path exactly.")


if __name__ == "__main__":
    main()
