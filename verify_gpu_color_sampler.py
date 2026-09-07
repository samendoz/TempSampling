"""
Unit parity test: GPUColorBatchSampler vs. ColorBatchSampler (CPU).

Builds a small synthetic temporal graph in the same CSR format TGL/TempSampling
uses (indptr/indices/eid -- see gen_graph.py / ext_full.npz), builds both a CPU
ColorBatchSampler and a GPUColorBatchSampler from it, drives both through an
identical scripted sequence of sample_batch()/update_node_indptr_direct() calls,
and asserts EXACT equality of every returned/mutated value at every step.

This is a re-vectorization of the same discrete algorithm (the node_stable=true
branch of ColoringSampler::sample_batch in color_sampler_core.cpp, plus
update_node_color_ptrs), not an approximation -- any mismatch here is a bug, not
numerical noise.

Requires the compiled ColorSamplerCore extension, torch, and a CUDA device
(GPUColorBatchSampler is GPU-only). Run on the target machine:

    python verify_gpu_color_sampler.py
"""
import numpy as np
import pandas as pd
import torch

from color_sampler import ColorBatchSampler
from color_sampler_gpu import GPUColorBatchSampler


def build_synthetic_graph(num_nodes, num_events, seed):
    """
    Builds a small bidirectional temporal CSR graph: for every event
    i = (src, dst, t=i), both src's and dst's adjacency list gets an entry
    pointing at the other node, tagged with eid=i. Per-node segments come out
    sorted by eid ascending (events are generated in order), matching what
    color_graph()'s min_eid/max_eid windowing assumes about indptr/indices/eid.
    """
    rng = np.random.RandomState(seed)
    src = rng.randint(0, num_nodes, size=num_events)
    dst = rng.randint(0, num_nodes, size=num_events)
    mask = src != dst
    src, dst = src[mask], dst[mask]
    num_events = len(src)

    per_node_nbr = [[] for _ in range(num_nodes)]
    per_node_eid = [[] for _ in range(num_nodes)]
    for e in range(num_events):
        s, d = int(src[e]), int(dst[e])
        per_node_nbr[s].append(d)
        per_node_eid[s].append(e)
        per_node_nbr[d].append(s)
        per_node_eid[d].append(e)

    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    indices, flat_eid = [], []
    for n in range(num_nodes):
        indptr[n + 1] = indptr[n] + len(per_node_nbr[n])
        indices.extend(per_node_nbr[n])
        flat_eid.extend(per_node_eid[n])
    indices = np.array(indices, dtype=np.int32)
    flat_eid = np.array(flat_eid, dtype=np.int32)
    # edge_index is unread by the node_stable=true branch under test here, but
    # the C++ constructor still needs a shape-compatible std::vector<pair<int,int>>.
    edge_index = list(np.stack([indices, indices], axis=0).T)

    train_df = pd.DataFrame({
        'src': src, 'dst': dst,
        'time': np.arange(num_events, dtype=np.float32),
        'Unnamed: 0': np.arange(num_events),
    })

    return list(indptr), edge_index, indices, flat_eid, num_events, train_df


def make_samplers(indptr, edge_index, indices, eid, num_nodes, num_events, num_colors, window_size):
    # IMPORTANT: the constructor's "num_edges" argument must be len(eid) -- the
    # flattened CSR array length -- not the number of actual temporal events.
    # ColoringSampler's C++ constructor ignores the passed-in num_edges and
    # recomputes this->num_edges = eid.size() internally (same pattern as
    # num_nodes = indptr.size()-1); train_simple.py passes num_edges =
    # len(g["eid"]) for exactly this reason. sample_batch()'s end_edge_id /
    # minimal_batch_end_edge_id clamp against this value, so GPUColorBatchSampler
    # (which -- unlike the CPU wrapper -- recomputes that clamp itself in Python)
    # must be constructed with the same len(eid) value the CPU C++ object uses,
    # or the two will silently clamp batch boundaries differently.
    num_edges_csr = len(eid)
    cpu = ColorBatchSampler(indptr, edge_index, indices, eid, num_nodes, num_edges_csr,
                             num_colors=num_colors, num_recent_edges=window_size,
                             num_hops=2, use_full_edge=False)
    gpu = GPUColorBatchSampler(indptr, edge_index, indices, eid, num_nodes, num_edges_csr,
                                num_colors=num_colors, num_recent_edges=window_size,
                                num_hops=2, use_full_edge=False, device='cuda')
    cpu.color_graph(num_events)
    gpu.color_graph(num_events)
    cpu.set_node_stable_mode(True)
    gpu.set_node_stable_mode(True)
    return cpu, gpu


def assert_pointers_match(cpu, gpu, step_label):
    cpu_color_ptrs = np.array(cpu.sampler.current_node_color_ptrs)
    cpu_self_update_ptrs = np.array(cpu.sampler.current_node_self_update_ptrs)
    gpu_color_ptrs = gpu.current_node_color_ptrs.cpu().numpy()
    gpu_self_update_ptrs = gpu.current_node_self_update_ptrs.cpu().numpy()
    assert np.array_equal(cpu_color_ptrs, gpu_color_ptrs), (
        f"[{step_label}] current_node_color_ptrs mismatch:\ncpu={cpu_color_ptrs}\ngpu={gpu_color_ptrs}")
    assert np.array_equal(cpu_self_update_ptrs, gpu_self_update_ptrs), (
        f"[{step_label}] current_node_self_update_ptrs mismatch:\ncpu={cpu_self_update_ptrs}\ngpu={gpu_self_update_ptrs}")


def run_scenario(name, num_nodes, num_events, seed, stable_flag_fn, num_colors=8,
                  window_size=3, num_batches=15, step_size=40, minimal_batch_size=10):
    print(f"--- scenario: {name} ---")
    indptr, edge_index, indices, eid, num_events, train_df = build_synthetic_graph(num_nodes, num_events, seed)
    cpu, gpu = make_samplers(indptr, edge_index, indices, eid, num_nodes, num_events, num_colors, window_size)

    rng = np.random.RandomState(seed + 1)
    start_event_id = 0
    batch_index = 0
    for batch_index in range(num_batches):
        if start_event_id >= num_events:
            break
        stable_flag = stable_flag_fn(num_nodes, rng)
        cpu.update_node_stable_flag(stable_flag.clone())
        gpu.update_node_stable_flag(stable_flag.clone())

        cpu_end, cpu_related = cpu.sample_batch(train_df, start_event_id, batch_index,
                                                 minimal_batch_size=minimal_batch_size, step_size=step_size)
        gpu_end, gpu_related = gpu.sample_batch(train_df, start_event_id, batch_index,
                                                 minimal_batch_size=minimal_batch_size, step_size=step_size)
        assert cpu_end == gpu_end, f"[{name} batch {batch_index}] final_event mismatch: cpu={cpu_end} gpu={gpu_end}"
        assert np.array_equal(cpu_related, gpu_related), f"[{name} batch {batch_index}] related_nodes mismatch"

        cpu.update_node_indptr_direct(cpu_end, cpu_related)
        gpu.update_node_indptr_direct(gpu_end, gpu_related)
        assert_pointers_match(cpu, gpu, f"{name} batch {batch_index}")

        start_event_id = cpu_end
    print(f"    OK ({batch_index + 1} batches)")


def all_stable(num_nodes, rng):
    return torch.ones(num_nodes, dtype=torch.bool)


def all_unstable(num_nodes, rng):
    return torch.zeros(num_nodes, dtype=torch.bool)


def random_stable(num_nodes, rng):
    return torch.from_numpy(rng.rand(num_nodes) < 0.5)


def mostly_stable_long_run(num_nodes, rng):
    # Forces many nodes' pointers deep into their event lists in one batch --
    # exercises the "node runs past the end of its list" (invalid, doesn't
    # affect the min) path in sample_batch.
    return torch.from_numpy(rng.rand(num_nodes) < 0.95)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This test requires a CUDA device (GPUColorBatchSampler is GPU-only).")

    run_scenario("all_stable", num_nodes=40, num_events=800, seed=0, stable_flag_fn=all_stable)
    run_scenario("all_unstable", num_nodes=40, num_events=800, seed=1, stable_flag_fn=all_unstable)
    run_scenario("random_seeded", num_nodes=40, num_events=800, seed=2, stable_flag_fn=random_stable)
    run_scenario("varying_num_colors_small", num_nodes=40, num_events=800, seed=3,
                 stable_flag_fn=random_stable, num_colors=2)
    run_scenario("varying_num_colors_large", num_nodes=40, num_events=800, seed=4,
                 stable_flag_fn=random_stable, num_colors=50)
    run_scenario("exhaustion_overflow_path", num_nodes=25, num_events=600, seed=5,
                 stable_flag_fn=mostly_stable_long_run, num_colors=30, step_size=80)

    print("\nAll scenarios passed: GPUColorBatchSampler matches ColorBatchSampler exactly.")
