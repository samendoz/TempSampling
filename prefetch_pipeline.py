import queue
import threading
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

    def _producer_loop(self, batch_generator):
        for batch_data in batch_generator:
            if self.stopped:
                break

            # FIX: Unpack all 5 items yielded by batch_generator_fn()
            rows, root_nodes, ts, ptr_end, unique_pos_root_nodes = batch_data

            # 1. Graph Sampling
            if self.sampler is not None:
                if 'no_neg' in self.sample_param and self.sample_param['no_neg']:
                    pos_root_end = root_nodes.shape[0] * 2 // 3
                    self.sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    self.sampler.sample(root_nodes, ts)
                ret = self.sampler.get_ret()
            else:
                ret = None

            # 2. Build DGL Blocks
            if self.gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, self.sample_param['history'], cuda=self.all_gpu)
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=self.all_gpu)

            # 3. Input Feature Prep
            mfgs = prepare_input(mfgs, self.node_feats, self.edge_feats, combine_first=self.combine_first)

            # Package items into the item dictionary
            item = {
                'rows': rows,
                'root_nodes': root_nodes,
                'ts': ts,
                'ret': ret,
                'mfgs': mfgs,
                'ptr_end': ptr_end,
                'unique_pos_root_nodes': unique_pos_root_nodes
            }
            self.queue.put(item)

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