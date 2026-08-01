import torch
import queue
import threading
from utils import to_dgl_blocks, prepare_input

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
        """
        Background worker that samples and builds DGL blocks for batch n+1.
        """
        for batch_data in batch_generator:
            if self.stopped:
                break

            rows, root_nodes, ts = batch_data

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
                from utils import node_to_dgl_blocks
                mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=self.all_gpu)

            # 3. Initial Input Prep (Features)
            mfgs = prepare_input(mfgs, self.node_feats, self.edge_feats, combine_first=self.combine_first)

            # Put pre-constructed item into queue
            item = {
                'rows': rows,
                'root_nodes': root_nodes,
                'ts': ts,
                'ret': ret,
                'mfgs': mfgs
            }
            self.queue.put(item)

        self.queue.put(None)  # Sentinel to mark end of batches

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


def patch_mfg_mailbox(mailbox, mfg_batch):
    """
    Patches pre-fetched DGL blocks with the latest MailBox states right before training.
    """
    if mailbox is not None:
        # Repopulate updated mailbox features into mfgs[0]
        mailbox.prep_input_mails(mfg_batch['mfgs'][0])