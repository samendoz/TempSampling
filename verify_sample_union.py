"""
Correctness test for ParallelSampler::sample_union (sampler_core.cpp).

Background: sample()/sample_layer() has a real, pre-existing bug for
strategy='recent' (TGN/APAN, the models this project trains): if the SAME node
appears more than once in one sample() call at different timestamps, every
occurrence ends up reading the SAME final ts_ptr cursor value (set by whichever
occurrence update_ts_ptr's separate first pass happened to process last) instead
of its own timestamp-appropriate value. This is deterministic given a fixed
occurrence order, not merely a race -- see the comment above sample_union in
sampler_core.cpp for the full writeup.

sample_union() fixes this by grouping occurrences by node id and interleaving the
pointer-advance step with the neighbor-gather step per occurrence, in per-node
time order.

This script:
  1. Builds a small synthetic bidirectional temporal graph (same T-CSR shape
     ParallelSampler expects: indptr/indices/eid/ts).
  2. Picks a node with two occurrences at different timestamps, interleaved with
     other distinct root nodes in between (to stress node-vs-position grouping).
  3. Computes ground truth for each occurrence independently: a FRESH sampler,
     called with just that one (node, timestamp) pair -- since ts_ptr is a
     persistent cursor across calls, "fresh sampler" is what makes this a valid,
     independent reference for "what should this occurrence's neighbors be".
  4. Confirms sample() (the OLD, still-unfixed-by-design method) gets at least
     one of the two occurrences WRONG on the combined call -- i.e. this test
     would have caught the bug being fixed, it isn't vacuous.
  5. Confirms sample_union() gets BOTH occurrences exactly right on the same
     combined call.

Requires the compiled sampler_core extension. Run on the target machine:

    python verify_sample_union.py
"""
import numpy as np
from sampler_core import ParallelSampler

NUM_THREAD = 4
NUM_NEIGHBORS = 5


def build_synthetic_graph(num_nodes, num_events, seed):
    """
    Bidirectional temporal CSR graph (same shape as gen_graph.py's
    ext_full.npz): for every event i = (src, dst, t=i), both src's and dst's
    adjacency list gets an entry pointing at the other node, tagged eid=i,
    ts=i. Per-node segments come out ts-sorted ascending (events generated in
    increasing time order), matching what ParallelSampler expects.
    """
    rng = np.random.RandomState(seed)
    src = rng.randint(0, num_nodes, size=num_events)
    dst = rng.randint(0, num_nodes, size=num_events)
    mask = src != dst
    src, dst = src[mask], dst[mask]
    num_events = len(src)
    t = np.arange(num_events, dtype=np.float32)

    per_node_nbr = [[] for _ in range(num_nodes)]
    per_node_eid = [[] for _ in range(num_nodes)]
    per_node_ts = [[] for _ in range(num_nodes)]
    for e in range(num_events):
        s, d = int(src[e]), int(dst[e])
        per_node_nbr[s].append(d)
        per_node_eid[s].append(e)
        per_node_ts[s].append(t[e])
        per_node_nbr[d].append(s)
        per_node_eid[d].append(e)
        per_node_ts[d].append(t[e])

    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    indices, eid, ets = [], [], []
    for n in range(num_nodes):
        indptr[n + 1] = indptr[n] + len(per_node_nbr[n])
        indices.extend(per_node_nbr[n])
        eid.extend(per_node_eid[n])
        ets.extend(per_node_ts[n])
    indices = np.array(indices, dtype=np.int32)
    eid = np.array(eid, dtype=np.int32)
    ets = np.array(ets, dtype=np.float32)

    return indptr, indices, eid, ets, num_events


def make_sampler(indptr, indices, eid, ets):
    return ParallelSampler(indptr, indices, eid, ets,
                            NUM_THREAD, 1,          # num_thread_per_worker, num_workers
                            1, [NUM_NEIGHBORS],      # num_layers, num_neighbors
                            True, False,             # recent, prop_time  (strategy='recent', matching TGN/APAN)
                            1, 0.0)                  # num_history, window_duration (duration: 0, matching TGN/APAN)


def edge_set_for_root(ret0, row_filter):
    """Given a TemporalGraphBlock, return the set of (eid) sampled for edges
    whose row matches row_filter (a boolean mask over the row array)."""
    row = np.asarray(ret0.row())
    eid = np.asarray(ret0.eid())
    return set(eid[row_filter(row)].tolist())


def ground_truth_neighbors(indptr, indices, eidarr, ets, node, ts):
    """Independent reference: a FRESH sampler, single (node, ts) call."""
    sampler = make_sampler(indptr, indices, eidarr, ets)
    sampler.sample(np.array([node], dtype=np.int32), np.array([ts], dtype=np.float32))
    ret0 = sampler.get_ret()[0]
    return set(np.asarray(ret0.eid()).tolist())


def main():
    indptr, indices, eidarr, ets, num_events = build_synthetic_graph(num_nodes=30, num_events=600, seed=0)

    # Pick the node with the most incident events, so it has a real choice of
    # neighbors at both an early and a late point in its history.
    degrees = indptr[1:] - indptr[:-1]
    target_node = int(np.argmax(degrees))
    assert degrees[target_node] >= 2 * NUM_NEIGHBORS, "synthetic graph too small; increase num_events"

    node_ts_options = ets[indptr[target_node]:indptr[target_node + 1]]
    t_early = float(node_ts_options[len(node_ts_options) // 4])
    t_late = float(node_ts_options[3 * len(node_ts_options) // 4])
    assert t_early < t_late

    gt_early = ground_truth_neighbors(indptr, indices, eidarr, ets, target_node, t_early)
    gt_late = ground_truth_neighbors(indptr, indices, eidarr, ets, target_node, t_late)
    assert gt_early != gt_late, (
        "ground-truth neighbor sets for the early/late occurrence happened to "
        "match by chance -- pick different t_early/t_late so the test can "
        "actually distinguish correct-per-occurrence handling from the bug."
    )

    # Build a combined root_nodes/root_ts array: target_node at t_early,
    # several distinct filler nodes, target_node again at t_late, more filler.
    # Position indices (rows) of the two target_node occurrences are recorded
    # explicitly rather than assumed.
    rng = np.random.RandomState(1)
    filler = rng.choice(np.delete(np.arange(30), target_node), size=10, replace=False)
    filler_ts = rng.uniform(t_early, t_late, size=10).astype(np.float32)

    root_nodes = np.concatenate([filler[:5], [target_node], filler[5:], [target_node]]).astype(np.int32)
    root_ts = np.concatenate([filler_ts[:5], [t_early], filler_ts[5:], [t_late]]).astype(np.float32)
    pos_early = 5
    pos_late = len(root_nodes) - 1
    assert root_nodes[pos_early] == target_node and root_ts[pos_early] == t_early
    assert root_nodes[pos_late] == target_node and root_ts[pos_late] == t_late

    # --- Step 1: confirm the OLD sample() gets this wrong (test isn't vacuous) ---
    old_sampler = make_sampler(indptr, indices, eidarr, ets)
    old_sampler.sample(root_nodes, root_ts)
    old_ret0 = old_sampler.get_ret()[0]
    old_early = edge_set_for_root(old_ret0, lambda row: row == pos_early)
    old_late = edge_set_for_root(old_ret0, lambda row: row == pos_late)

    bug_reproduced = (old_early != gt_early) or (old_late != gt_late)
    print(f"sample() (old, unfixed): early matches ground truth = {old_early == gt_early}, "
          f"late matches ground truth = {old_late == gt_late}")
    if not bug_reproduced:
        print("WARNING: sample() got both occurrences right on this input -- the test "
              "didn't reproduce the bug on this particular graph/seed. The sample_union "
              "check below is still meaningful, but re-run with a different seed to also "
              "confirm this test can catch the bug it's meant to guard against.")
    else:
        print("Confirmed: sample() is wrong here, as expected -- this input does "
              "exercise the bug sample_union fixes.")

    # --- Step 2: confirm sample_union() gets both occurrences exactly right ---
    union_sampler = make_sampler(indptr, indices, eidarr, ets)
    union_sampler.sample_union(root_nodes, root_ts)
    union_ret0 = union_sampler.get_ret()[0]
    union_early = edge_set_for_root(union_ret0, lambda row: row == pos_early)
    union_late = edge_set_for_root(union_ret0, lambda row: row == pos_late)

    assert union_early == gt_early, (
        f"sample_union early-occurrence mismatch:\n  got={union_early}\n  want={gt_early}")
    assert union_late == gt_late, (
        f"sample_union late-occurrence mismatch:\n  got={union_late}\n  want={gt_late}")

    print("OK: sample_union matches independent ground truth for both the early and "
          "late occurrence of the repeated node.")


if __name__ == "__main__":
    main()
