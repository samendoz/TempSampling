import queue
import threading
from time import perf_counter
from utils import to_dgl_blocks, node_to_dgl_blocks, prepare_input

class PrefetchProducer:
    def __init__(self, sampler, sample_param, gnn_param, node_feats, edge_feats,
                 combine_first=False, all_gpu=True, queue_size=2):
        self.sampler = sampler
        self.sample_param = sample_param
        self.gnn_param = gnn_param
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.combine_first = combine_first
        self.all_gpu = all_gpu

        self.queue = queue.Queue(maxsize=queue_size)
        self.thread = None
        self.stopped = False

        # Producer-side timing. This work used to run (and be timed) on the main/
        # consumer thread; now it runs on this background thread, so its cost is
        # overlapped with the consumer's train+postprocess of the previous batch
        # and would otherwise be invisible in the epoch-level timing summary.
        # queue_put_wait_time is the important one to watch: time this thread spends
        # blocked on queue.put() because the queue is full means the producer is
        # running ahead faster than the consumer can drain it -- i.e. it is NOT the
        # bottleneck and the "hidden" time really is fully hidden. If instead the
        # consumer's queue_wait_time (see train_simple.py) is the one that's
        # nonzero, the producer is the bottleneck and its cost is leaking back into
        # wall-clock time despite the prefetching.
        self.stats_lock = threading.Lock()
        self.stats = {
            'sampling_time': 0.0,
            'to_dgl_blocks_time': 0.0,
            'prepare_input_time': 0.0,
            'queue_put_wait_time': 0.0,
            'batches_produced': 0,
        }

    def get_stats(self):
        with self.stats_lock:
            return dict(self.stats)

    def reset_stats(self):
        with self.stats_lock:
            for k in self.stats:
                self.stats[k] = 0.0 if isinstance(self.stats[k], float) else 0

    def _producer_loop(self, batch_generator):
        for batch_data in batch_generator:
            if self.stopped:
                break

            # Unpack all 6 items yielded by batch_generator_fn(): rows, root_nodes,
            # ts, ptr_end, unique_pos_root_nodes, and related_nodes (the exact node
            # set sample_batch() used to decide ptr_end -- must be carried through
            # explicitly, see color_sampler.update_node_indptr_direct()).
            rows, root_nodes, ts, ptr_end, unique_pos_root_nodes, related_nodes = batch_data

            # 1. Graph Sampling
            if self.sampler is not None:
                t0 = perf_counter()
                if 'no_neg' in self.sample_param and self.sample_param['no_neg']:
                    pos_root_end = root_nodes.shape[0] * 2 // 3
                    self.sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    self.sampler.sample(root_nodes, ts)
                ret = self.sampler.get_ret()
                with self.stats_lock:
                    self.stats['sampling_time'] += perf_counter() - t0
            else:
                ret = None

            # 2. Build DGL Blocks
            t0 = perf_counter()
            if self.gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, self.sample_param['history'], cuda=self.all_gpu)
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=self.all_gpu)
            with self.stats_lock:
                self.stats['to_dgl_blocks_time'] += perf_counter() - t0

            # 3. Input Feature Prep
            t0 = perf_counter()
            mfgs = prepare_input(mfgs, self.node_feats, self.edge_feats, combine_first=self.combine_first)
            with self.stats_lock:
                self.stats['prepare_input_time'] += perf_counter() - t0
                self.stats['batches_produced'] += 1

            # Package items into the item dictionary
            item = {
                'rows': rows,
                'root_nodes': root_nodes,
                'ts': ts,
                'ret': ret,
                'mfgs': mfgs,
                'ptr_end': ptr_end,
                'unique_pos_root_nodes': unique_pos_root_nodes,
                'related_nodes': related_nodes,
            }
            t0 = perf_counter()
            self.queue.put(item)
            with self.stats_lock:
                self.stats['queue_put_wait_time'] += perf_counter() - t0

        self.queue.put(None)  # End sentinel

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


def patch_mfg_mailbox(mailbox, batch_item):
    if mailbox is not None:
        mailbox.prep_input_mails(batch_item['mfgs'][0])
