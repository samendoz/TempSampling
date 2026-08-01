import torch
import numpy as np
import ColorSamplerCore
import pandas as pd
import sys, time
import pickle
import os
import threading


class ColorBatchSampler():
    """
    ColorBatchSampler is a class that sample events from a color graph as a batch.
    """
    def __init__(self, 
                    indptr, 
                    edge_index, 
                    indices,
                    eid, 
                    num_nodes, 
                    num_edges, 
                    num_colors=500, 
                    num_hops=2, 
                    num_recent_edges=4, 
                    num_workers=8, 
                    num_threads_per_worker=8,
                    num_nodes_per_thread=1,
                    decay_type = "disable",
                    decay_step = 5,
                    decay_factor = 5,
                    minimum_scale_factor=0.5,
                    use_full_edge=False):
        """
        Constructor of the class.
        :param indptr: indptr is the start index of the edges for each node.
        :param edge_index: edge_index is the index of each edge in the graph in format of (src, dst).
        :param eid: eid is the event id for each edge.
        :param num_nodes: num_nodes is the number of nodes in the graph.
        :param num_edges: num_edges is the number of edges in the graph.
        :param num_colors: num_colors is the number of colors each node can suffer.
        :param num_hops: num_hops is the number of hops an event can propagate.
        :param num_recent_edges: num_recent_edges is the number of recent edges to propagate on each node
        :param batch_size: batch_size is the minimal number of events to sample for each batch.
        :param num_workers: num_workers is the number of workers to use for sampling.
        :param num_threads_per_worker: num_threads_per_worker is the number of threads to use for each worker.
        :param num_nodes_per_thread: num_nodes_per_thread is the number of nodes to use for each thread.
        :param node_stable_flag: node_stable_flag is the flag to indicate whether we use node stable to decide if an event is stable.
        :param decay_type: decay_type is the type of decay to use for the number of colors.
        :param decay_step: decay_step is the step to decay the number of colors.
        :param decay_factor: decay_factor is the factor to decay the number of colors.
        """
        self.indptr = indptr
        self.edge_index = edge_index
        self.eid = eid
        self.indices = indices

        self.num_nodes = num_nodes
        self.num_edges = num_edges

        self.num_colors = num_colors
        self.num_hops = num_hops
        self.num_recent_edges = num_recent_edges

        self.num_workers = num_workers
        self.num_threads_per_worker = num_threads_per_worker

        self.decay_type = decay_type
        self.decay_step = decay_step
        self.decay_factor = decay_factor
        
        self.color_bottom = np.floor(num_colors * minimum_scale_factor).astype(np.int32)
        self.related_nodes = None
        # input("create sampler core...")
        # print("indptr: ", indptr)
        # print("edge_index: ", edge_index)
        # print("eid: ", eid)

        self.use_full_edge = use_full_edge
    
        self.sampler = ColorSamplerCore.ColoringSampler(indptr, 
                                                        edge_index, 
                                                        indices,  
                                                        eid,
                                                        num_nodes,
                                                        num_hops, num_recent_edges,
                                                        num_threads_per_worker, num_workers, num_nodes_per_thread)
        if not self.use_full_edge:
            self.sampler.disable_full_edges()
            
        print("C++ sampler size: ", sys.getsizeof(self.sampler)/1024/1024, "MB")
        
        # print("sampler core created.")
        # input("check sampler core, press enter to continue...")
        # print("indptr: ", self.sampler.indptr)
        # print("edge_index: ", self.sampler.edge_index)
        # print("eid: ", np.equal(np.array(self.sampler.eid), np.array(eid)).all())
        # input("finish checking sampler core, press enter to continue...")
                                                           
        self.prev_node_stable_flag = torch.zeros(num_nodes-1, dtype=torch.bool)
        self.node_stable_flag = torch.zeros(num_nodes-1, dtype=torch.bool)
        # self.prev_node_stable_flag = torch.zeros(num_nodes, dtype=torch.bool)
        self.use_memory = False
        self.node_stable_mode = False
        self.batch_index_list = []
        self.batch_index = 0

        # PREFETCH: guards all reads/writes of node_stable_flag, related_nodes,
        # batch_index_list and the underlying C++ color-ptr state. sample_batch()
        # (called from the producer thread in the prefetch pipeline) and
        # update_node_indptr_direct()/update_node_stable_flag() (called from the
        # consumer thread) touch this shared state from two different threads;
        # this lock makes each individual call atomic (no torn tensor reads/writes),
        # though it does NOT guarantee sample_batch(N+1) sees update_node_stable_flag(N) --
        # that ordering is only as fresh as the producer's queue depth allows.
        self._state_lock = threading.Lock()

    def reset(self):
        self.reset_node_indptr()
        self.record_node_stable_flag()
        self.reset_node_stable_flag()
        if not self.use_memory:
            self.reset_batch_index_list()

    
    def reset_node_stable_flag(self):
        """
        Reset the node stable flag.
        """
        self.node_stable_flag = torch.zeros(self.num_nodes, dtype=torch.bool)

    def record_node_stable_flag(self, new_stable_flag=None):
        """
        Record the node stable flag.
        """
        with torch.no_grad():
            if new_stable_flag is not None:
                # print("record new node stable flag:", new_stable_flag.shape, self.node_stable_flag.shape)
                self.node_stable_flag = new_stable_flag
            else:
                self.prev_node_stable_flag = self.node_stable_flag

    def get_node_stable_flag(self):
        """
        Get the node stable flag.
        :return: node_stable_flag is the flag to indicate whether a node is stable.
        """
        return self.node_stable_flag

    def set_node_stable_mode(self, node_stable_mode):
        """
        Set the node stable mode.
        :param node_stable_mode: node_stable_mode is the mode to use node stable.
        """
        self.node_stable_mode = node_stable_mode

    def reset_batch_index_list(self):
        """
        Reset the batch index list.
        """
        self.batch_index_list = []
        self.batch_index = 0

    def update_node_stable_flag(self, node_stable_flag, root_nodes=None):
        """
        Update the node stable flag.
        :param root_nodes: root_nodes is the nodes to update.
        :param node_stable_flag: node_stable_flag is the flag to update.
        """
        with self._state_lock:
            with torch.no_grad():
                if root_nodes is not None:
                    index = root_nodes.long().cpu()
                    self.node_stable_flag[index] = node_stable_flag
                else:
                    self.node_stable_flag = node_stable_flag
        # print("node stable ratio: ", self.node_stable_flag.sum().item() / self.node_stable_flag.shape[0])

    def set_use_memory(self, use_memory):
        """
        Set the use memory flag.
        :param use_memory: use_memory is the flag to indicate whether to use memory.
        """
        print("use memory disabled")
        self.use_memory = False
        return False

        self.use_memory = use_memory
        if use_memory:
            print("use memory to sample the next epoch.")


    def check_if_use_memory(self, edge_index, recent_node_stable_flag):
        """
        if event stablized pattern is the same as the previous one, we can use memory to sample the next batch.
        """
        print("use memory disabled")
        self.use_memory = False
        return False

        event_stable_flag = recent_node_stable_flag[edge_index[0]] & recent_node_stable_flag[edge_index[1]]
        prev_event_stable_flag = self.prev_node_stable_flag[edge_index[0]] & self.prev_node_stable_flag[edge_index[1]]

        if torch.equal(event_stable_flag, prev_event_stable_flag):
            print("use memory to sample the next epoch.")
            self.use_memory = True
        else:
            self.use_memory = False
    
    def color_decay(self, batch_index):
        if self.decay_type == "disable":
            color_limit = self.num_colors
        elif self.decay_type == "linear":
            color_limit = max(self.num_colors - (batch_index // self.decay_step) * (self.num_colors // self.decay_factor), self.color_bottom)
        elif self.decay_type == "log":
            color_limit = max(self.num_colors - np.log2(batch_index // self.decay_step + 1).astype(np.int32) * (self.num_colors // self.decay_factor), self.color_bottom)
        else:
            color_limit = self.num_colors
        return color_limit

    def color_graph(self, end_event_id=-1, st_event_id=-1):
        """
        Color the graph with the given events.
        """
        # print("====before color graph====")
        # color_table = self.sampler.node_usage_table
        # self_color_table = self.sampler.node_self_update_table
        # print(color_table)
        # input("check color table, press enter to continue...")
        # print(self_color_table)
        # input("check self color table, press enter to continue...")

        # print("color graph with events: ", st_event_id, end_event_id)
        
        self.sampler.color_graph(st_event_id, end_event_id)


        # print("====color graph====")
        # color_table = self.sampler.node_usage_table
        # self_color_table = self.sampler.node_self_update_table
        # print(color_table)
        # input("check color table, press enter to continue...")
        # print(self_color_table)
        # input("check self color table, press enter to continue...")

    def reset_color_graph(self):
        """
        reset the color graph.
        """
        self.sampler.reset_color_table()

    def reset_node_indptr(self):
        """
        Reset the node indptr.
        """
        self.sampler.reset_nodeindptr()
        # print("====reset node color index====")
        # print(self.sampler.current_node_color_ptrs)
        # print("====reset node color index done====")
        # print("====reset node self update index====")
        # print(self.sampler.current_node_self_update_ptrs)
        # print("====reset node self update index done====")
        # input("check reset node indptr, press enter to continue...")

    def update_node_indptr(self, recent_event_id, root_nodes):
        """
        Update the node indptr.

        NOTE: relies on self.related_nodes, a single-slot side channel written by
        sample_batch(). This pairing (one sample_batch() call, then one
        update_node_indptr() call before the next sample_batch()) only holds when
        both are called from the same thread in strict alternation. Do NOT use
        this from the prefetch pipeline (producer/consumer threads) -- use
        update_node_indptr_direct() instead, which takes an explicit root_nodes
        value and never touches the side channel.
        """
        with self._state_lock:
            if self.related_nodes is not None:
                root_nodes = self.related_nodes
                self.sampler.update_node_color_ptrs(recent_event_id, root_nodes)
                self.related_nodes = None
            else:
                self.sampler.update_node_color_ptrs(recent_event_id, root_nodes)
        # print("update node indptr done: ",self.sampler.current_node_color_ptrs, "sum", sum(self.sampler.current_node_color_ptrs))
        # input("check node indptr, press enter to continue...")

    def update_node_indptr_direct(self, recent_event_id, related_nodes):
        """
        Thread-safe variant for the prefetch pipeline: takes the exact root_nodes
        computed by the matching sample_batch() call (threaded through explicitly,
        e.g. via the producer's queue item) instead of relying on the
        self.related_nodes side channel, which can be overwritten by a later
        sample_batch() call before a slower consumer thread gets to it.
        """
        with self._state_lock:
            self.sampler.update_node_color_ptrs(recent_event_id, related_nodes)


    def sample_batch(self, 
                     train_df,
                     start_event_id,
                     batch_index,
                     minimal_batch_size=1000,
                     step_size=8000,):
        """
        Sample a batch of events from the graph, decide the end event id for the next batch.
        :param start_event_id: start_event_id is the start event id for the batch.
        :param stable_flag: stable_flag is the flag to indicate whether we use node stable to decide if an event is stable.
        :param minimal_batch_size: minimal_batch_size is the minimal number of events to sample for each batch.
        :param step_size: step_size is the number of events to sample for each step.
        :return: (end_event_id, root_nodes) -- root_nodes is the exact node set this call
            used to decide the boundary; callers that mutate color state afterward
            (update_node_indptr_direct) should be given this value explicitly rather
            than recomputing/re-reading it, since under the prefetch pipeline another
            sample_batch() call for a later batch may run before a slower consumer
            gets around to consuming this one.
        """
        st_t = time.time()
        checked_df = train_df.loc[start_event_id:start_event_id+step_size]
        # root node are nodes within the checked_df as src or dst
        root_nodes = np.unique(np.concatenate([checked_df['src'].values, checked_df['dst'].values])).astype(np.int32)
        # print("root nodes: ", root_nodes, root_nodes.shape)
        # input("check root nodes, press enter to continue...")

        # print("\t\tbatch", batch_index, "node stable ratio: ", self.node_stable_flag.sum().item() / self.node_stable_flag.shape[0])
        # input("check node stable flag, press enter to continue...")
        ed_t_root_nodes = time.time()

        with self._state_lock:
            self.related_nodes = root_nodes

            if self.use_memory:
                end_event_id = self.batch_index_list[batch_index]
            elif self.node_stable_mode:
                # print("ok node stable mode")
                node_stable_flag = self.node_stable_flag[root_nodes]
                # get the unstable nodes
                unstable_nodes = root_nodes[~node_stable_flag]
                # print("\t\tunstable nodes: ", len(unstable_nodes), "root nodes: ", len(root_nodes))
                num_colors = self.color_decay(batch_index)
                # print("\tnum colors: ", num_colors, "remained_node_ratio: ", len(unstable_nodes) / len(root_nodes))
                end_event_id = self.sampler.sample_batch(unstable_nodes,
                                                        start_event_id,
                                                        self.node_stable_flag,
                                                        num_colors,
                                                        minimal_batch_size,
                                                        step_size,
                                                        self.node_stable_mode)
                self.batch_index_list.append(end_event_id)
            else:
                num_colors = self.color_decay(batch_index)
                # print("num colors: ", num_colors, "self.num_colors: ", self.num_colors)
                end_event_id = self.sampler.sample_batch(root_nodes,
                                                         start_event_id,
                                                         self.node_stable_flag,
                                                         num_colors,
                                                         minimal_batch_size,
                                                         step_size,
                                                         self.node_stable_mode)
                self.batch_index_list.append(end_event_id)

        # print("\tsample batch time: ", time.time() - ed_t_root_nodes, "get root nodes time: ", ed_t_root_nodes - st_t)

        # print("end event id: ", end_event_id)
        # input("check end event id, press enter to continue...")
        return end_event_id, root_nodes

class MultiColorBatchSampler(ColorBatchSampler):
    """
    MultiColorBatchSampler is a class that sample
    """
    def __init__(self, 
                    indptr, 
                    edge_index, 
                    indices,
                    eid, 
                    num_nodes, 
                    num_edges, 
                    num_colors=500, 
                    num_hops=2, 
                    num_recent_edges=4, 
                    num_workers=8, 
                    num_threads_per_worker=8,
                    num_nodes_per_thread=1,
                    decay_type = "disable",
                    decay_step = 5,
                    decay_factor = 5,
                    minimum_scale_factor=0.5,
                    chunk_num=1,
                    cache_dir="sampler_caches",
                    enable_pickle=False,
                    use_full_edge=False):
        super(MultiColorBatchSampler, self).__init__(indptr, 
                                                    edge_index, 
                                                    indices,
                                                    eid, 
                                                    num_nodes, 
                                                    num_edges, 
                                                    num_colors, 
                                                    num_hops=num_hops, 
                                                    num_recent_edges=num_recent_edges, 
                                                    num_workers=num_workers, 
                                                    num_threads_per_worker=num_threads_per_worker,
                                                    num_nodes_per_thread=num_nodes_per_thread,
                                                    decay_type=decay_type,
                                                    decay_step=decay_step,
                                                    decay_factor=decay_factor,
                                                    minimum_scale_factor=minimum_scale_factor,
                                                    use_full_edge=use_full_edge)
        self.sampler.enable_multi_color()
        # multi color map storage
        self.chunk_num = chunk_num
        self.multi_update_table = dict()
        self.multi_usage_table = dict()
        self.enable_pickle = enable_pickle
        self.cache_dir = cache_dir

    def enable_pickle(self, enable_pickle):
        """
        Enable the pickle flag.
        """
        self.enable_pickle = enable_pickle
    
    def disable_pickle(self):
        """
        Disable the pickle flag.
        """
        self.enable_pickle = False

    def cache_multi_color_table(self, chunk_id):
        """
        Cache the multi color table.
        """
        if self.enable_pickle:
            with open(self.cache_dir + "/multi_color_usage_table_" + str(chunk_id) + ".pkl", "wb") as f:
                pickle.dump(self.sampler.get_usage_table(), f)
            with open(self.cache_dir + "/multi_color_update_table_" + str(chunk_id) + ".pkl", "wb") as f:
                pickle.dump(self.sampler.get_update_table(), f)
        else:
            self.multi_update_table[chunk_id] = self.sampler.get_usage_table()
            self.multi_usage_table[chunk_id] = self.sampler.get_update_table()

    def load_multi_color_table(self, chunk_id):
        """
        Set the multi color table.
        """
        if self.enable_pickle:
            with open(self.cache_dir + "/multi_color_usage_table_" + str(chunk_id) + ".pkl", "rb") as f:
                self.sampler.set_usage_table(pickle.load(f))
            with open(self.cache_dir + "/multi_color_update_table_" + str(chunk_id) + ".pkl", "rb") as f:
                self.sampler.set_update_table(pickle.load(f))
        else:
            self.sampler.set_usage_table(self.multi_update_table[chunk_id])
            self.sampler.set_update_table(self.multi_usage_table[chunk_id])
        
    def is_cached(self, chunk_id):
        """
        Check if the multi color table is cached.
        """
        if self.enable_pickle:
            return os.path.exists(self.cache_dir + "/multi_color_usage_table_" + str(chunk_id) + ".pkl") and os.path.exists(self.cache_dir + "/multi_color_update_table_" + str(chunk_id) + ".pkl")
        else:
            return chunk_id in self.multi_update_table and chunk_id in self.multi_usage_table

def resolve_graph(train_df, node_num=0):
    """
    resolve the graph from the given dataframe. Takes node pointers, edge index and event id as input.
    input: train_df: dataframe containing the graph
    output: indptr, edge_index, eid, num_nodes, num_edges
    """
    # print("====resolve graph====")
    # print("====train_df====")
    # print(train_df)
    # sys.exit()
    src_nodes = train_df['src'].astype(np.int64)
    dest_nodes = train_df['dst'].astype(np.int64)
    nodes = np.unique(np.concatenate([src_nodes, dest_nodes]))
    if node_num > 0:
        max_num_nodes = node_num + 1
    else:
        max_num_nodes = nodes.max() + 1
    indptr = torch.zeros(max_num_nodes + 1, dtype=torch.int64)
    edge_index = []
    eid = []
    

    for i, node in enumerate(nodes):
        # find the edges that destinate and originate from the node, and their corresponding indices in the dataframe
        related_edges = train_df[(train_df['src'] == node) | (train_df['dst'] == node)]
        # print("====related_edges on node ", node, "====")
        # print(len(related_edges))
        # print(related_edges.index)
        # sys.exit()

        node_edges_idx = related_edges.index
        node_src = related_edges['src'].astype(np.int64)
        node_dst = related_edges['dst'].astype(np.int64)

        # add src and dst to the edge index
        node_edges = np.stack([node_src, node_dst], axis=0).T
        edge_index.append(node_edges)
        # add the event id as index of the dataframe
        eid += node_edges_idx.tolist()
        # print("====node_edges_idx====")
        # print(node_edges_idx)
        # set the indptr for the node
        indptr[node] = len(node_edges)
        # input("check indptr, press enter to continue...")

    indptr[-1] = len(train_df)
    indptr = indptr.cumsum(0)
    edge_index = np.concatenate(edge_index, axis=0)
    
    indptr = torch.cat([torch.tensor([0]), indptr[:-1]])

    return indptr, edge_index, eid, max_num_nodes, len(edge_index)


def resolve_full_graph(indptr, indices, eid, df=None):
    """
    resolve the full graph from the given indptr, edge index and event id.
    input: indptr, edge_index, eid, num_nodes, num_edges
    output: full graph
    """
    print("====resolve full graph====")
    print("====indptr====", indptr.shape)
    print(indptr)
    print("====indices====", indices.shape)
    print(indices)
    print("====eid====",eid.shape)
    print(eid)
    print("eid max: ", eid.max())
    print("eid min: ", eid.min())
    if df is not None:
        print("====df with eid====")
        eid_df = df.loc[eid]
        print(eid_df)
        print("max df eid: ", eid_df.index.max())
        print("min df eid: ", eid_df.index.min())
    

    # input("check input, press enter to continue...")

    # unfold indptr to source node indices
    src_nodes = []
    for i in range(len(indptr) - 1):
        src_nodes += [i] * (indptr[i + 1] - indptr[i])
    src_nodes = np.array(src_nodes)
    print("====src_nodes====", src_nodes.shape)
    print(src_nodes)

    # input("check src_nodes, press enter to continue...")

    # combine src_nodes and indices to form the full graph
    full_graph = np.stack([src_nodes, indices], axis=0).T

    print("====full_graph====", full_graph.shape)
    print(full_graph)

    # input("check full_graph, press enter to continue...")

    # get new eid based on the eid index
    # new_eid = np.arange(len(eid))

    # get node and edge number
    num_nodes = len(indptr)
    num_edges = len(indices)

    return indptr, full_graph, eid, num_nodes, num_edges