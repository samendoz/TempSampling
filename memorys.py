import torch
import dgl
from layers import TimeEncode
from torch_scatter import scatter
import time

# profile functions
PROFILE = False
EFFICIENT_PROFILE = False
BATCH_FREEZE = False
EPOCH_FREEZE = False
FREEZE_THRESHOLD = 0.9
profile_dict = dict()
prev_profile_dict = dict()
node_update_index_dict = dict()
node_sim_trace_dict = dict()
node_sim_trace_list = list()
cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

MAILBOX_PREP_PROFILE_SUMMARY_PROFILE = False
MAILBOX_PREP_PROFILE_SUMMARY = {
    'mailbox_index_time': 0.0,
    'mailbox_up_index_time': 0.0,
    'mailbox_up_dedup_time': 0.0,
    'mailbox_up_write_time': 0.0,
    'mem_stab_prep_time': 0.0,
    'mem_stab_math_time': 0.0,
    'mem_stab_write_time': 0.0,
}

def set_mailbox_prep_profile(enabled=True):
    global MAILBOX_PREP_PROFILE_SUMMARY_PROFILE
    MAILBOX_PREP_PROFILE_SUMMARY_PROFILE = enabled

def reset_mailbox_prep_profile():
    global MAILBOX_PREP_PROFILE_SUMMARY
    for key in MAILBOX_PREP_PROFILE_SUMMARY:
        MAILBOX_PREP_PROFILE_SUMMARY[key] = 0.0

def get_mailbox_prep_profile_summary():
    global MAILBOX_PREP_PROFILE_SUMMARY
    return MAILBOX_PREP_PROFILE_SUMMARY

def print_mailbox_prep_profile():
    mailbox_index_time = MAILBOX_PREP_PROFILE_SUMMARY['mailbox_index_time']
    mailbox_up_index_time = MAILBOX_PREP_PROFILE_SUMMARY['mailbox_up_index_time']
    mailbox_up_dedup_time = MAILBOX_PREP_PROFILE_SUMMARY['mailbox_up_dedup_time']
    mailbox_up_write_time = MAILBOX_PREP_PROFILE_SUMMARY['mailbox_up_write_time']
    mem_stab_prep_time = MAILBOX_PREP_PROFILE_SUMMARY['mem_stab_prep_time']
    mem_stab_math_time = MAILBOX_PREP_PROFILE_SUMMARY['mem_stab_math_time']
    mem_stab_write_time = MAILBOX_PREP_PROFILE_SUMMARY['mem_stab_write_time']

    print("Mailbox Prep Profile Summary:")
    print("\tMailbox Index Time: {:.6f}s Mailbox Update Index Time: {:.6f}s Mailbox Update Deduplication Time: {:.6f}s Mailbox Update Write Time: {:.6f}s Memory Stability Prep Time: {:.6f}s Memory Stability Math Time: {:.6f}s Memory Stability Write Time: {:.6f}s".format(
        mailbox_index_time, mailbox_up_index_time, mailbox_up_dedup_time, mailbox_up_write_time, mem_stab_prep_time, mem_stab_math_time, mem_stab_write_time))

def enable_profiling():
    global PROFILE
    PROFILE = True

def disable_profiling():
    global PROFILE
    PROFILE = False

def set_efficient_profiling():
    global EFFICIENT_PROFILE
    EFFICIENT_PROFILE = True


def collect_profile(node_id, value, node_memory=None):
    global profile_dict, PROFILE, node_sim_trace_dict
    if not PROFILE:
        return
    
    if EFFICIENT_PROFILE:
        # this slow down the updating process but save memory
        # for i, node in enumerate(node_id):
        #     if node.item() not in profile_dict:
        #         profile_dict[node.item()] = value[i].cpu()
        #     else:
        #         new_feature = value[i].cpu()
        #         update_similarity = cos_sim(profile_dict[node.item()], new_feature)
        #         if node.item() not in node_sim_trace_dict:
        #             node_sim_trace_dict[node.item()] = list()
        #             node_sim_trace_dict[node.item()].append(update_similarity)
        #         else:
        #             node_sim_trace_dict[node.item()].append(update_similarity)
        #         profile_dict[node.item()] = new_feature
        prev_memory = node_memory[node_id]
        update_similarity = cos_sim(prev_memory, value)
        node_sim_trace_list.append(update_similarity.cpu().numpy())
        # for i, node in enumerate(node_id):
        #     if node.item() not in node_sim_trace_dict:
        #         node_sim_trace_dict[node.item()] = list()
        #         node_sim_trace_dict[node.item()].append(update_similarity[i].cpu())
        #     else:
        #         node_sim_trace_dict[node.item()].append(update_similarity[i].cpu())
    else:

        for i,node in enumerate(node_id):
            if node.item() not in profile_dict:
                profile_dict[node.item()] = list()
            profile_dict[node.item()].append(value[i].cpu()) 
    # print("profile_dict", profile_dict)
    # input("Press Enter to continue...")

def reset_profile():
    global profile_dict, node_sim_trace_dict, node_sim_trace_list
    profile_dict = dict()
    node_sim_trace_dict = dict()
    node_sim_trace_list = list()

def reset_n_record_profile():
    global profile_dict, prev_profile_dict
    prev_profile_dict = profile_dict
    profile_dict = dict()

def compare_profile(plot_dir=None, dump_feature=False):
    global profile_dict, prev_profile_dict, PROFILE

    if not PROFILE:
        return

    res_dict = dict()
    cosin_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for node_id, profile_list in profile_dict.items():
        # similar_list = list()
        before_update = prev_profile_dict[node_id]
        after_update = profile_list

        before_update = torch.stack(before_update)
        after_update = torch.stack(after_update)

        # cosin_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity = cosin_similarity(before_update, after_update)

        res_dict[node_id] = similarity.cpu().numpy()
    
    if plot_dir.endswith(".png"):
        import matplotlib.pyplot as plt

        for node_id, similarity in res_dict.items():
            x_axis = list(range(len(similarity)))
            plt.plot(x_axis, similarity)
        
        plt.savefig(plot_dir)
        plt.close()
    elif plot_dir.endswith(".pkl"):
        import pickle
        with open(plot_dir, "wb") as f:
            pickle.dump(res_dict, f)
        
    else:
        print("res_dict", res_dict)

def analysis_profile(plot_dir=None, dump_feature=False):
    global profile_dict, PROFILE
    if not PROFILE:
        return

    if dump_feature:
        import pickle
        with open(plot_dir, "wb") as f:
            pickle.dump(profile_dict, f)
        return
    
    res_dict = dict()
    res_list = None
    if not EFFICIENT_PROFILE:
        cosin_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for node_id, profile_list in profile_dict.items():
            # similar_list = list()

            before_update = profile_list[:-1]
            after_update = profile_list[1:]

            # print("before_update", before_update)
            # print("after_update", after_update)
            # input("Press Enter to continue...")
            # if any of the list is empty, go to next node
            if not before_update or not after_update:
                continue

            before_update = torch.stack(before_update)
            after_update = torch.stack(after_update)

            # cosin_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            similarity = cosin_similarity(before_update, after_update)

            res_dict[node_id] = similarity.cpu().numpy()
    else:
        # for huge dataset, we can use this efficient way to store the similarity
        # res_dict = node_sim_trace_dict
        res_list = node_sim_trace_list
        # res_list = np.concatenate(res_list, axis=0)
    
        if plot_dir.endswith(".pkl"):
            import pickle
            with open(plot_dir, "wb") as f:
                pickle.dump(res_list, f)
        else:
            print("res_list", res_list)
        return


    if plot_dir.endswith(".png"):
        import matplotlib.pyplot as plt

        for node_id, similarity in res_dict.items():
            x_axis = list(range(len(similarity)))
            plt.plot(x_axis, similarity)
        
        plt.savefig(plot_dir)
        plt.close()
    elif plot_dir.endswith(".pkl"):
        import pickle
        with open(plot_dir, "wb") as f:
            pickle.dump(res_dict, f)
        
    else:
        print("res_dict", res_dict)
        # similarity_stats_d


def enable_batch_freeze():
    global BATCH_FREEZE
    BATCH_FREEZE = True

def disable_batch_freeze():
    global BATCH_FREEZE
    BATCH_FREEZE = False

def is_batch_freeze():
    global BATCH_FREEZE
    return BATCH_FREEZE

def get_freeze_threshold():
    global FREEZE_THRESHOLD
    return FREEZE_THRESHOLD

def set_freeze_threshold(threshold):
    global FREEZE_THRESHOLD
    FREEZE_THRESHOLD = threshold

def reset_node_update_index():
    global node_update_index_dict
    node_update_index_dict = dict()

def init_node_update_index(node_id_list):
    global node_update_index_dict
    for node_id in node_id_list:
        node_update_index_dict[node_id] = 0

def inter_batch_has_stable_input(similarity, src_idx, dst_idx, row_idx, sim_batch_size=900, threshold=0.9):
    """
    based on similarity, check if the input of src_idx and dst_idx is stable
    """
    global node_update_index_dict
    # which batch the update is in during profiling
    update_idx = row_idx // sim_batch_size 

    src_sim = similarity[src_idx]
    dst_sim = similarity[dst_idx]

    # which update this event is in
    



class MailBox():

    def __init__(self, memory_param, num_nodes, dim_edge_feat, _node_memory=None, _node_memory_ts=None,_mailbox=None, _mailbox_ts=None, _next_mail_pos=None, _update_mail_pos=None):
        self.memory_param = memory_param
        self.dim_edge_feat = dim_edge_feat
        if memory_param['type'] != 'node':
            raise NotImplementedError
        self.node_memory = torch.zeros((num_nodes, memory_param['dim_out']), dtype=torch.float32) if _node_memory is None else _node_memory
        self.node_memory_ts = torch.zeros(num_nodes, dtype=torch.float32) if _node_memory_ts is None else _node_memory_ts
        self.mailbox = torch.zeros((num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_edge_feat), dtype=torch.float32) if _mailbox is None else _mailbox
        self.mailbox_ts = torch.zeros((num_nodes, memory_param['mailbox_size']), dtype=torch.float32) if _mailbox_ts is None else _mailbox_ts
        self.next_mail_pos = torch.zeros((num_nodes), dtype=torch.long) if _next_mail_pos is None else _next_mail_pos
        self.update_mail_pos = _update_mail_pos
        self.device = torch.device('cpu')
        self.freeze_threshold = 0.9
        self.frozen_node_mask = torch.zeros((num_nodes), dtype=torch.long)
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def reset(self):
        self.node_memory.fill_(0)
        self.node_memory_ts.fill_(0)
        self.mailbox.fill_(0)
        self.mailbox_ts.fill_(0)
        self.next_mail_pos.fill_(0)
        self.frozen_node_mask.fill_(0)

    def move_to_gpu(self):
        self.node_memory = self.node_memory.cuda()
        self.node_memory_ts = self.node_memory_ts.cuda()
        self.mailbox = self.mailbox.cuda()
        self.mailbox_ts = self.mailbox_ts.cuda()
        self.next_mail_pos = self.next_mail_pos.cuda()
        self.frozen_node_mask = self.frozen_node_mask.cuda()
        self.device = torch.device('cuda:0')

    def allocate_pinned_memory_buffers(self, sample_param, batch_size):
        limit = int(batch_size * 3.3)
        if 'neighbor' in sample_param:
            for i in sample_param['neighbor']:
                limit *= i + 1
        self.pinned_node_memory_buffs = list()
        self.pinned_node_memory_ts_buffs = list()
        self.pinned_mailbox_buffs = list()
        self.pinned_mailbox_ts_buffs = list()
        for _ in range(sample_param['history']):
            self.pinned_node_memory_buffs.append(torch.zeros((limit, self.node_memory.shape[1]), pin_memory=True))
            self.pinned_node_memory_ts_buffs.append(torch.zeros((limit,), pin_memory=True))
            self.pinned_mailbox_buffs.append(torch.zeros((limit, self.mailbox.shape[1], self.mailbox.shape[2]), pin_memory=True))
            self.pinned_mailbox_ts_buffs.append(torch.zeros((limit, self.mailbox_ts.shape[1]), pin_memory=True))

    def prep_input_mails(self, mfg, use_pinned_buffers=False):
        global MAILBOX_PREP_PROFILE_SUMMARY
        profile = MAILBOX_PREP_PROFILE_SUMMARY_PROFILE

        for i, b in enumerate(mfg):
            if use_pinned_buffers:

                time_idx_start = time.time()
                idx = b.srcdata['ID'].cpu().long()
                torch.index_select(self.node_memory, 0, idx, out=self.pinned_node_memory_buffs[i][:idx.shape[0]])
                time_idx = time.time() - time_idx_start

                b.srcdata['mem'] = self.pinned_node_memory_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                
                time_idx_start = time.time()
                torch.index_select(self.node_memory_ts,0, idx, out=self.pinned_node_memory_ts_buffs[i][:idx.shape[0]])
                time_idx += time.time() - time_idx_start

                b.srcdata['mem_ts'] = self.pinned_node_memory_ts_buffs[i][:idx.shape[0]].cuda(non_blocking=True)

                time_idx_start = time.time()
                torch.index_select(self.mailbox, 0, idx, out=self.pinned_mailbox_buffs[i][:idx.shape[0]])
                time_idx += time.time() - time_idx_start

                b.srcdata['mem_input'] = self.pinned_mailbox_buffs[i][:idx.shape[0]].reshape(b.srcdata['ID'].shape[0], -1).cuda(non_blocking=True)

                time_idx_start = time.time()
                torch.index_select(self.mailbox_ts, 0, idx, out=self.pinned_mailbox_ts_buffs[i][:idx.shape[0]])
                time_idx += time.time() - time_idx_start

                b.srcdata['mail_ts'] = self.pinned_mailbox_ts_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
            else:
                # print("node memory device", self.node_memory.device, "b.srcdata device", b.srcdata['ID'].device)
                # move index to the device of the node memory and mailbox---this is for no-all-gpu case
                time_idx_start = time.time()
                device = self.node_memory.device
                idx = b.srcdata['ID'].long().to(device)
                time_idx = time.time() - time_idx_start

                b.srcdata['mem'] = self.node_memory[idx].cuda()
                b.srcdata['mem_ts'] = self.node_memory_ts[idx].cuda()
                b.srcdata['mem_input'] = self.mailbox[idx].cuda().reshape(b.srcdata['ID'].shape[0], -1)
                b.srcdata['mail_ts'] = self.mailbox_ts[idx].cuda()

                # b.srcdata['mem'] = self.node_memory[b.srcdata['ID'].long()].cuda()
                # b.srcdata['mem_ts'] = self.node_memory_ts[b.srcdata['ID'].long()].cuda()
                # b.srcdata['mem_input'] = self.mailbox[b.srcdata['ID'].long()].cuda().reshape(b.srcdata['ID'].shape[0], -1)
                # b.srcdata['mail_ts'] = self.mailbox_ts[b.srcdata['ID'].long()].cuda()
            if profile:
                MAILBOX_PREP_PROFILE_SUMMARY['mailbox_index_time'] += time_idx

    def update_memory(self, nid, memory, root_nodes, ts, neg_samples=1):
        if nid is None:
            return
        num_true_src_dst = root_nodes.shape[0] // (neg_samples + 2) * 2
        with torch.no_grad():
            nid = nid[:num_true_src_dst].to(self.device)
            memory = memory[:num_true_src_dst].to(self.device)
            ts = ts[:num_true_src_dst].to(self.device)
            collect_profile(nid.long(), memory, self.node_memory)
            self.node_memory[nid.long()] = memory
            self.node_memory_ts[nid.long()] = ts
            # collect_profile(nid.long(), memory, self.node_memory)

    def get_full_node_stable_flag(self):
        return self.node_stable_flag[:,-1].cpu()

    def update_memory_and_check_stablizing(self, nid, memory, root_nodes, ts, neg_samples=1, threshold=0.9, any=True):

        global MAILBOX_PREP_PROFILE_SUMMARY
        profile = MAILBOX_PREP_PROFILE_SUMMARY_PROFILE

        if nid is None:
            return
        num_true_src_dst = root_nodes.shape[0] // (neg_samples + 2) * 2
        with torch.no_grad():

            # Phase 1 Slice and transfer
            time_prep_start = time.time()

            nid = nid[:num_true_src_dst].to(self.device).long()
            memory = memory[:num_true_src_dst].to(self.device)
            ts = ts[:num_true_src_dst].to(self.device)

            time_prep = time.time() - time_prep_start
            time_math_start = time.time()

            similarity = self.similarity(memory, self.node_memory[nid])
            node_stable = similarity > threshold
            if self.histroy_window_size > 2:
                # prev_stable = self.node_stable_flag[nid]
                self.node_stable_flag[nid, :-1] = self.node_stable_flag[nid, 1:]
            self.node_stable_flag[nid, -1] = node_stable

            time_math = time.time() - time_math_start
            time_write_start = time.time()

            self.node_memory[nid] = memory
            self.node_memory_ts[nid] = ts
            collect_profile(nid, memory, self.node_memory)

            time_write = time.time() - time_write_start

            if profile:
                MAILBOX_PREP_PROFILE_SUMMARY['mem_stab_prep_time'] += time_prep
                MAILBOX_PREP_PROFILE_SUMMARY['mem_stab_math_time'] += time_math
                MAILBOX_PREP_PROFILE_SUMMARY['mem_stab_write_time'] += time_write

    def get_node_stable_flag(self, nid, window_size=2, any=True):

        if nid is None:
            return
        
        with torch.no_grad():
            node_stable = self.node_stable_flag[nid]
            # total check add to number of elements in the similarity matrix
            self.total_check_count_node += node_stable.shape[0] * node_stable.shape[1]
            # total stable add to number of elements in the is_stable matrix
            self.total_stable_count_node += node_stable.sum()

            if any:
                reduce_stable = torch.any(node_stable)
            else:
                reduce_stable = torch.all(node_stable)

            self.total_check_count_event += 1
            self.total_stable_count_event += reduce_stable
            return reduce_stable

    def set_history_recorder(self, window_size):
        self.histroy_window_size = window_size
        self.node_memory_histroy = torch.zeros((self.node_memory.shape[0],window_size, self.node_memory.shape[1]), dtype=torch.float32).to(self.device)
        self.total_check_count_node = 0
        self.total_stable_count_node = 0
        self.total_check_count_event = 0
        self.total_stable_count_event = 0

        self.correct_stable_count_node = 0
        self.accu_node_count = 0

    def set_stablize_recorder(self, window_size):
        self.histroy_window_size = window_size
        self.node_stable_flag = torch.zeros((self.node_memory.shape[0],window_size-1), dtype=torch.bool).to(self.device)
        self.total_check_count_node = 0
        self.total_stable_count_node = 0
        self.total_check_count_event = 0
        self.total_stable_count_event = 0


    def record_recent_memory_history(self, nid, memory, root_nodes, ts, neg_samples=1):
        if nid is None:
            return
        num_true_src_dst = root_nodes.shape[0] // (neg_samples + 2) * 2
        with torch.no_grad():
            nid = nid[:num_true_src_dst].to(self.device)
            memory = memory[:num_true_src_dst].to(self.device)
            #shift the history window left by 1
            nid = nid.long()
            self.node_memory_histroy[nid, :-1, :] = self.node_memory_histroy[nid, 1:, :]
            # overwrite the last one
            self.node_memory_histroy[nid, -1, :] = memory


    def validate_memory_history(self, nid, memory, root_nodes, ts, neg_samples=1, threshold=0.9, any=True, show_log=False):
        if nid is None:
            return
        num_true_src_dst = root_nodes.shape[0] // (neg_samples + 2) * 2
        with torch.no_grad():
            nid = nid[:num_true_src_dst].to(self.device)
            memory = memory[:num_true_src_dst].to(self.device)

            nid = nid.long()
            # cosin_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
            similarity = self.similarity(self.node_memory_histroy[nid, 1:, :], self.node_memory_histroy[nid, :-1, :])
            is_stable = similarity > threshold # nid * window_size
            
            if any:
                reduce_stable = torch.any(is_stable, dim=-1)
            else:
                reduce_stable = torch.all(is_stable, dim=-1)
            
            real_stable = self.similarity(memory, self.node_memory_histroy[nid, -1, :]) > threshold
            # if any:
            #     real_reduce_stable = torch.any(real_stable)
            # else:
            #     real_reduce_stable = torch.all(real_stable)

            # print("nid shape", nid.shape,"is_stable shape", is_stable.shape, "reduce_stable shape", reduce_stable.shape, "real_stable shape", real_stable.shape, "real_reduce_stable shape", real_reduce_stable.shape)
            # input("Press Enter to continue...")
            self.correct_stable_count_node += (reduce_stable == real_stable).sum()
            self.accu_node_count += reduce_stable.shape[0]

            
    
    def get_stablizing_memory_check_accuracy(self):
        # if self.accu_node_count == 0:
        #     return -1
        return self.correct_stable_count_node/self.accu_node_count


    # def get_stablizing_memory_check_accuracy(self):
    #     return self.correct_stable_count_node/self.total_check_count_node, self.correct_stable_count_event/self.total_check_count_event



    def is_node_memory_stable(self, nid, threshold=0.9, any=True, show_log=False):
        if nid is None:
            return
        with torch.no_grad():
            # nid = nid.long()
            # cosin_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
            similarity = self.similarity(self.node_memory_histroy[nid, 1:, :], self.node_memory_histroy[nid, :-1, :])
            is_stable = similarity > threshold # nid * window_size
      
            if any:
                reduce_stable = torch.any(is_stable)
            else:
                reduce_stable = torch.all(is_stable)
 

            # total check add to number of elements in the similarity matrix
            self.total_check_count_node += similarity.shape[0] * similarity.shape[1]
            # total stable add to number of elements in the is_stable matrix
            self.total_stable_count_node += is_stable.sum()

            self.total_check_count_event += 1
            self.total_stable_count_event += reduce_stable

            if show_log:
                print("nid:{}, similarity:{}, is_stable:{}, reduce_stable:{}".format(nid, similarity, is_stable, reduce_stable))
                print("total_check_count_node:{}, total_stable_count_node:{}, total_check_count_event:{}, total_stable_count_event:{}".format(self.total_check_count_node, self.total_stable_count_node, self.total_check_count_event, self.total_stable_count_event))
            # if reduce_stable:
            #     print("********stable during check!!!*******")
                # print("nid:{}, similarity:{}, is_stable:{}, reduce_stable:{}".format(nid, similarity, is_stable, reduce_stable))
                # print("total_check_count_node:{}, total_stable_count_node:{}, total_check_count_event:{}, total_stable_count_event:{}".format(self.total_check_count_node, self.total_stable_count_node, self.total_check_count_event, self.total_stable_count_event))
                # # input("Press Enter to continue...")
            
            # input("Press Enter to continue...")

            return reduce_stable
    def get_node_stable_ratio(self):
        return self.total_stable_count_node,self.total_check_count_node
    def get_event_stable_ratio(self):
        return self.total_stable_count_event,self.total_check_count_event

    def check_stablized_memory(self, nid, memory, root_nodes, ts, neg_samples=1):
        if nid is None:
            return
        num_true_src_dst = root_nodes.shape[0] // (neg_samples + 2) * 2
        with torch.no_grad():
            nid = nid[:num_true_src_dst].to(self.device)
            memory = memory[:num_true_src_dst].to(self.device)
            ts = ts[:num_true_src_dst].to(self.device)
            
            similarity = self.similarity(memory, self.node_memory[nid.long()])
            self.frozen_node_mask[nid.long()] = similarity > self.freeze_threshold

            # we can observe similarity and masks here

            self.node_memory[nid.long()] = memory
            self.node_memory_ts[nid.long()] = ts
    
    def get_frozen_node_mask(self):
        return self.frozen_node_mask

    def update_mailbox(self, nid, memory, root_nodes, ts, edge_feats, block, neg_samples=1):
        global MAILBOX_PREP_PROFILE_SUMMARY
        profile = MAILBOX_PREP_PROFILE_SUMMARY_PROFILE

        with torch.no_grad():

            time_mailbox_prep_start = time.time()
            num_true_edges = root_nodes.shape[0] // (neg_samples + 2)
            memory = memory.to(self.device)
            if edge_feats is not None:
                edge_feats = edge_feats.to(self.device)
            if block is not None:
                block = block.to(self.device)
            # TGN/JODIE
            if self.memory_param['deliver_to'] == 'self':
                src = torch.from_numpy(root_nodes[:num_true_edges]).to(self.device)
                dst = torch.from_numpy(root_nodes[num_true_edges:num_true_edges * 2]).to(self.device)
                mem_src = memory[:num_true_edges]
                mem_dst = memory[num_true_edges:num_true_edges * 2]
                if self.dim_edge_feat > 0:
                    src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
                else:
                    src_mail = torch.cat([mem_src, mem_dst], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src], dim=1)
                mail = torch.cat([src_mail, dst_mail], dim=1).reshape(-1, src_mail.shape[1])
                nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
                mail_ts = torch.from_numpy(ts[:num_true_edges * 2]).to(self.device)

                time_mailbox_prep = time.time() - time_mailbox_prep_start

                time_dedup_start = time.time()

                if mail_ts.dtype == torch.float64:
                    import pdb; pdb.set_trace()
                # find unique nid to update mailbox
                uni, inv = torch.unique(nid, return_inverse=True)
                perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
                perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
                nid = nid[perm]
                mail = mail[perm]
                mail_ts = mail_ts[perm]

                time_dedup = time.time() - time_dedup_start
                time_write_start = time.time()

                if self.memory_param['mail_combine'] == 'last':
                    self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail
                    self.mailbox_ts[nid.long(), self.next_mail_pos[nid.long()]] = mail_ts
                    if self.memory_param['mailbox_size'] > 1:
                        self.next_mail_pos[nid.long()] = torch.remainder(self.next_mail_pos[nid.long()] + 1, self.memory_param['mailbox_size'])
                time_write = time.time() - time_write_start

                if profile:
                    MAILBOX_PREP_PROFILE_SUMMARY['mailbox_up_index_time'] += time_mailbox_prep
                    MAILBOX_PREP_PROFILE_SUMMARY['mailbox_up_dedup_time'] += time_dedup
                    MAILBOX_PREP_PROFILE_SUMMARY['mailbox_up_write_time'] += time_write
            # APAN
            elif self.memory_param['deliver_to'] == 'neighbors':
                mem_src = memory[:num_true_edges]
                mem_dst = memory[num_true_edges:num_true_edges * 2]
                if self.dim_edge_feat > 0:
                    src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
                else:
                    src_mail = torch.cat([mem_src, mem_dst], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src], dim=1)
                mail = torch.cat([src_mail, dst_mail], dim=0)
                mail = torch.cat([mail, mail[block.edges()[0].long()]], dim=0)
                mail_ts = torch.from_numpy(ts[:num_true_edges * 2]).to(self.device)
                mail_ts = torch.cat([mail_ts, mail_ts[block.edges()[0].long()]], dim=0)

                time_mailbox_prep = time.time() - time_mailbox_prep_start
                time_dedup_start = time.time()

                if self.memory_param['mail_combine'] == 'mean':
                    (nid, idx) = torch.unique(block.dstdata['ID'], return_inverse=True)
                    mail = scatter(mail, idx, reduce='mean', dim=0)
                    mail_ts = scatter(mail_ts, idx, reduce='mean')

                    time_dedup = time.time() - time_dedup_start
                    time_write_start = time.time()

                    self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail
                    self.mailbox_ts[nid.long(), self.next_mail_pos[nid.long()]] = mail_ts

                    time_write = time.time() - time_write_start

                elif self.memory_param['mail_combine'] == 'last':
                    nid = block.dstdata['ID']
                    # find unique nid to update mailbox
                    uni, inv = torch.unique(nid, return_inverse=True)
                    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
                    perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
                    nid = nid[perm]
                    mail = mail[perm]
                    mail_ts = mail_ts[perm]

                    time_dedup = time.time() - time_dedup_start
                    time_write_start = time.time()

                    self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail
                    self.mailbox_ts[nid.long(), self.next_mail_pos[nid.long()]] = mail_ts

                    time_write = time.time() - time_write_start

                else:
                    raise NotImplementedError
                
                if profile:
                    MAILBOX_PREP_PROFILE_SUMMARY['mailbox_up_index_time'] += time_mailbox_prep
                    MAILBOX_PREP_PROFILE_SUMMARY['mailbox_up_dedup_time'] += time_dedup
                    MAILBOX_PREP_PROFILE_SUMMARY['mailbox_up_write_time'] += time_write

                if self.memory_param['mailbox_size'] > 1:
                    if self.update_mail_pos is None:
                        self.next_mail_pos[nid.long()] = torch.remainder(self.next_mail_pos[nid.long()] + 1, self.memory_param['mailbox_size'])
                    else:
                        self.update_mail_pos[nid.long()] = 1
            else:
                raise NotImplementedError
            
            
    def update_next_mail_pos(self):
        if self.update_mail_pos is not None:
            nid = torch.where(self.update_mail_pos == 1)[0]
            self.next_mail_pos[nid] = torch.remainder(self.next_mail_pos[nid] + 1, self.memory_param['mailbox_size'])
            self.update_mail_pos.fill_(0)

class GRUMemeoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat):
        super(GRUMemeoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.updater = torch.nn.GRUCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, mfg):
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
            updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
            if is_batch_freeze():
                threshold = get_freeze_threshold()
                similarity = self.cosine_similarity(updated_memory, b.srcdata['mem'])
                # print("similarity(mean/max/min)", similarity.mean(), similarity.max(), similarity.min())
                # input("Press Enter to continue...")
                mask = similarity > threshold
                mask = mask.unsqueeze(1).repeat(1, self.dim_hid)
                updated_memory = torch.where(mask, b.srcdata['mem'], updated_memory)
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            if self.memory_param['combine_node_feature']:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += updated_memory
                    else:
                        b.srcdata['h'] = updated_memory + self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = updated_memory
            # print("===================================== inside GRUMemeoryUpdater ================================")
            # print("updated_memory", updated_memory.shape)
            # print("b.srcdata['h']", b.srcdata['h'].shape)
            # print("b.srcdata['mem']", b.srcdata['mem'].shape)
            # print("b.srcdata['mem_input']", b.srcdata['mem_input'].shape)
            # print("self.last_updated_nid", self.last_updated_nid.shape)
            # collect_profile(self.last_updated_nid, self.last_updated_memory)

class RNNMemeoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat):
        super(RNNMemeoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.updater = torch.nn.RNNCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)

    def forward(self, mfg):
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
            updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            if self.memory_param['combine_node_feature']:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += updated_memory
                    else:
                        b.srcdata['h'] = updated_memory + self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = updated_memory

class TransformerMemoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_out, dim_time, train_param):
        super(TransformerMemoryUpdater, self).__init__()
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.att_h = memory_param['attention_head']
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        self.w_q = torch.nn.Linear(dim_out, dim_out)
        self.w_k = torch.nn.Linear(dim_in + dim_time, dim_out)
        self.w_v = torch.nn.Linear(dim_in + dim_time, dim_out)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
        self.mlp = torch.nn.Linear(dim_out, dim_out)
        self.dropout = torch.nn.Dropout(train_param['dropout'])
        self.att_dropout = torch.nn.Dropout(train_param['att_dropout'])
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None

    def forward(self, mfg):
        for b in mfg:
            Q = self.w_q(b.srcdata['mem']).reshape((b.num_src_nodes(), self.att_h, -1))
            mails = b.srcdata['mem_input'].reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], -1))
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'][:, None] - b.srcdata['mail_ts']).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], -1))
                mails = torch.cat([mails, time_feat], dim=2)
            K = self.w_k(mails).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1))
            V = self.w_v(mails).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1))
            att = self.att_act((Q[:,None,:,:]*K).sum(dim=3))
            att = torch.nn.functional.softmax(att, dim=1)
            att = self.att_dropout(att)
            rst = (att[:,:,:,None]*V).sum(dim=1)
            rst = rst.reshape((rst.shape[0], -1))
            rst += b.srcdata['mem']
            rst = self.layer_norm(rst)
            rst = self.mlp(rst)
            rst = self.dropout(rst)
            rst = torch.nn.functional.relu(rst)
            b.srcdata['h'] = rst
            self.last_updated_memory = rst.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            self.last_updated_ts = b.srcdata['ts'].detach().clone()

