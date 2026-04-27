import threading
import numpy as np


class PrefetchManager:
    """
    Overlaps sampler.sample() with other CPU prep work in the same batch.

    Call schedule(nodes, ts) immediately after root_nodes/ts are built to kick
    off a background thread that runs sampler.sample(nodes, ts) + get_ret().
    Call get() just before to_dgl_blocks(ret) to retrieve the result.

    The background thread runs concurrently with unique_pos_root_nodes
    computation and any other CPU prep between schedule() and get(), as well
    as any GPU work that releases the GIL (e.g. backward pass of previous
    iterations cached in CUDA streams).

    If prefetch is not set in the YAML config (defaults to False), PrefetchManager
    is never instantiated and all paths fall back to the original synchronous flow.
    """

    def __init__(self, sampler):
        self._sampler = sampler
        self._result = None
        self._ready = threading.Event()
        self._ready.set()   # starts "ready" — no pending work
        self.n_hits = 0
        self.n_misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(self, nodes, ts):
        """
        Kick off sampler.sample(nodes, ts) in a background thread.
        nodes and ts must already be the final arrays to pass to the sampler
        (caller handles no_neg trimming before calling here).
        """
        self._ready.wait()      # ensure previous thread is done
        self._ready.clear()
        self._result = None
        threading.Thread(target=self._run, args=(nodes, ts), daemon=True).start()

    def get(self):
        """
        Block until the background thread finishes, then return ret.
        Returns None if the thread errored (caller should fall back to
        synchronous sampler.sample()).
        """
        self._ready.wait()
        result = self._result
        self._result = None     # consume — avoid accidental reuse
        if result is not None:
            self.n_hits += 1
        else:
            self.n_misses += 1
        return result

    def clear(self):
        """Reset stats (call at epoch boundaries)."""
        self._ready.wait()
        self._result = None
        self.n_hits = 0
        self.n_misses = 0

    def hit_rate(self):
        total = self.n_hits + self.n_misses
        return self.n_hits / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self, nodes, ts):
        try:
            self._sampler.sample(nodes, ts)
            self._result = self._sampler.get_ret()
        except Exception as exc:
            print(f'[PrefetchManager] background thread error: {exc}')
        finally:
            self._ready.set()
