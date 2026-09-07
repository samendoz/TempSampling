"""
Concurrency stress test for the GIL-release change (Part 1 of the plan).

Confirms the *_nogil C++ bindings don't crash or corrupt state when genuinely
run from two threads at once -- something that was never actually exercised
before this change, since the GIL previously serialized every call regardless
of which thread made it.

Two things are checked:

1. `sample_batch_nogil` (producer role) and `update_node_color_ptrs_nogil` /
   `update_node_stable_flag` (consumer role) hammered concurrently from two
   threads against the SAME ColorBatchSampler(use_nogil=True) instance and its
   `_state_lock` -- this is exactly the producer/consumer split
   train_simple_v2.py uses. Race conditions are probabilistic, so this runs
   many iterations rather than a handful.
2. A smoke test for the scoped-out risk the design review flagged (ATen calls
   inside a nogil region touching Python's warning machinery): note that the
   node_stable=true branch this repo actually uses (see color_sampler_gpu.py's
   docstring) never calls `stable_flag.item<bool>()` at all -- that only
   happens in the node_stable=false scan branch, which is out of scope (see
   plan). So there is nothing torch::Tensor-related for sample_batch_nogil to
   touch inside its parallel region in the path we ship. This test still
   exercises `update_node_color_ptrs_nogil` (pure std::vector/int state) under
   concurrent load as the real safety net.

Requires the compiled ColorSamplerCore extension. Run on the target machine:

    python verify_nogil_concurrency.py
"""
import threading
import time

import numpy as np
import torch

from color_sampler import ColorBatchSampler
from verify_gpu_color_sampler import build_synthetic_graph

NUM_ITERATIONS = 2000
NUM_NODES = 60
NUM_EVENTS = 3000


def main():
    indptr, edge_index, indices, eid, num_events, train_df = build_synthetic_graph(NUM_NODES, NUM_EVENTS, seed=42)
    sampler = ColorBatchSampler(indptr, edge_index, indices, eid, NUM_NODES, num_events,
                                 num_colors=6, num_recent_edges=3, num_hops=2,
                                 use_full_edge=False, use_nogil=True)
    sampler.color_graph(num_events)
    sampler.set_node_stable_mode(True)

    errors = []
    stop = threading.Event()

    def producer():
        rng = np.random.RandomState(1)
        start_event_id = 0
        for i in range(NUM_ITERATIONS):
            if stop.is_set():
                return
            if start_event_id >= num_events - 1:
                start_event_id = 0  # wrap around, we only care about stress, not epoch semantics
            try:
                end_id, related = sampler.sample_batch(train_df, start_event_id, i,
                                                         minimal_batch_size=5, step_size=30)
                start_event_id = end_id
            except Exception as exc:  # noqa: BLE001 -- want to catch and report, not hide
                errors.append(f"producer iteration {i}: {exc!r}")
                return

    def consumer():
        rng = np.random.RandomState(2)
        for i in range(NUM_ITERATIONS):
            if stop.is_set():
                return
            try:
                related = sampler.related_nodes
                if related is not None and len(related) > 0:
                    sampler.update_node_indptr_direct(min(NUM_EVENTS - 1, i * 3), related)
                stable_flag = torch.from_numpy(rng.rand(NUM_NODES) < 0.5)
                sampler.update_node_stable_flag(stable_flag)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"consumer iteration {i}: {exc!r}")
                return

    t_producer = threading.Thread(target=producer)
    t_consumer = threading.Thread(target=consumer)

    start = time.time()
    t_producer.start()
    t_consumer.start()
    t_producer.join(timeout=120)
    t_consumer.join(timeout=120)
    elapsed = time.time() - start

    if t_producer.is_alive() or t_consumer.is_alive():
        stop.set()
        raise SystemExit(f"FAILED: threads did not finish within 120s (possible deadlock). elapsed={elapsed:.1f}s")

    if errors:
        raise SystemExit("FAILED:\n" + "\n".join(errors))

    color_ptrs = np.array(sampler.sampler.current_node_color_ptrs)
    self_update_ptrs = np.array(sampler.sampler.current_node_self_update_ptrs)
    assert (color_ptrs >= 0).all(), f"corrupted (negative) current_node_color_ptrs: {color_ptrs[color_ptrs < 0]}"
    assert (self_update_ptrs >= 0).all(), f"corrupted (negative) current_node_self_update_ptrs: {self_update_ptrs[self_update_ptrs < 0]}"

    print(f"OK: {NUM_ITERATIONS} concurrent producer/consumer iterations in {elapsed:.1f}s, "
          "no crash, no corrupted pointer state.")


if __name__ == "__main__":
    main()
