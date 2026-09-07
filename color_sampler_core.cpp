#include <iostream>
#include <string>
#include <cstdlib>
#include <random>
#include <omp.h>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <set>
#include <algorithm>
#include <torch/extension.h>

namespace py = pybind11;

typedef int NodeIDType;
typedef int EdgeIDType;
typedef int ColorIDType;
typedef std::pair<NodeIDType, NodeIDType> EdgeType;
typedef std::set<ColorIDType> ColorSetType;
typedef std::vector<ColorIDType> ColorListType;
typedef std::vector<ColorListType> NodeEdgeTableType;



class ColoringSampler {
    public:
        // class parameters
        // graph information
        std::vector<EdgeIDType> indptr;             // indptr is the start index of the edges for each node     
        std::vector<EdgeType> edge_index;           // edge_index is the index of each edge in the graph in format of (src, dst)
        std::vector<NodeIDType> indices;            // indices is the destniation node of each edge in the graph
        std::vector<EdgeIDType> eid;                // edge id for each edge, refer to the index of the edge (i.e., event id) in the graph

        // coloring table of the graph
        NodeEdgeTableType node_usage_table;
        NodeEdgeTableType node_self_update_table;
        std::vector<ColorIDType> current_node_color_ptrs;       // record the current color pointer for each node
        std::vector<ColorIDType> current_node_self_update_ptrs;       // record the current pointer for each node to the self update edges
        // torch::Tensor node_stable_flag;

        // if large graph and multi color, we leave a trace for each node in indptr
        std::vector<EdgeIDType> node_trace;                // the begining point of the node in the indptr
        // flage if enable multi color features
        bool multi_color_flag;
        bool full_edges;

        // statistics of the graph
        NodeIDType num_nodes;
        EdgeIDType num_edges;

        // hyperparameters
        // int num_colors;
        int num_hops;
        int num_recent_edges;
        // int step_size;
        // int initial_batch_size;

        // specify the system configuration
        int num_thread_per_worker;
        int num_workers;
        int num_threads;
        int num_nodes_per_thread;

        // class methods
        ColoringSampler(std::vector<EdgeIDType> indptr, 
                        std::vector<EdgeType> edge_index,
                        std::vector<NodeIDType> indices,
                        std::vector<EdgeIDType> eid, 
                        int num_nodes,
                        int num_hops, int num_recent_edges,
                        int num_thread_per_worker, int num_workers, int num_nodes_per_thread) {
            this->indptr = indptr;
            this->edge_index = edge_index;
            this->indices = indices;
            this->eid = eid;

            // this->num_colors = num_colors;
            this->num_hops = num_hops;
            this->num_recent_edges = num_recent_edges;

            this->num_thread_per_worker = num_thread_per_worker;
            this->num_workers = num_workers;
            this->num_threads = num_thread_per_worker * num_workers;
            this->num_nodes_per_thread = num_nodes_per_thread;
            // this->num_nodes = num_nodes; 
            this->num_nodes = indptr.size() - 1;    // the number of nodes is the size of indptr minus 1        
            this->num_edges = eid.size();

            this->multi_color_flag = false;
            this->full_edges = true;

            // print the initialized information
            std::cout << "ColoringSampler initialized with " << std::endl;
            std::cout << "num_nodes: " << this->num_nodes << std::endl;
            std::cout << "num_edges: " << this->num_edges << std::endl;
            std::cout << "num_hops: " << this->num_hops << std::endl;
            std::cout << "num_recent_edges: " << this->num_recent_edges << std::endl;
            std::cout << "num_thread_per_worker: " << this->num_thread_per_worker << std::endl;
            std::cout << "num_workers: " << this->num_workers << std::endl;
            std::cout << "num_threads: " << this->num_threads << std::endl;
            std::cout << "num_nodes_per_thread: " << this->num_nodes_per_thread << std::endl;


            std::cout << "initializing the node usage table and the current node color pointers" << std::endl;
            // initialize the node usage table, the initial values are all zeros
            this->node_usage_table = NodeEdgeTableType(this->num_nodes);
            this->node_self_update_table = NodeEdgeTableType(this->num_nodes);
            // show memory cost of the tables in GB
            std::cout << "node_usage_table memory cost: " << this->num_nodes * sizeof(ColorListType) / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;
            std::cout << "node_self_update_table memory cost: " << this->num_nodes * sizeof(ColorListType) / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;
            // for (NodeIDType i = 0; i < this->num_nodes; i++) {
            //     this->node_usage_table[i] = ColorSetType();
            // }

            // initialize the current node indptr
            std::cout << "initializing the current node color pointers" << std::endl;
            this->current_node_color_ptrs = std::vector<ColorIDType>(this->num_nodes, 0);
            this->current_node_self_update_ptrs = std::vector<ColorIDType>(this->num_nodes, 0);
            // show memory cost of the current node color pointers in GB
            std::cout << "current_node_color_ptrs memory cost: " << this->num_nodes * sizeof(ColorIDType) / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;
            std::cout << "current_node_self_update_ptrs memory cost: " << this->num_nodes * sizeof(ColorIDType) / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;


        }

        /*enable multi color
        description: enable multi color feature
        input: None
        output: None
        */
        void enable_multi_color(){
            this->multi_color_flag = true;
            this->node_trace = std::vector<EdgeIDType>(this->num_nodes, 0);
        }

        /*disable full edges
        description: disable full edges feature
        input: None
        output: None
        */  
        void disable_full_edges(){
            this->full_edges = false;
        }

        /*
        method: reset nodeindptr
        description: reset the current node idptr to zeros
        input: None
        output: None
        */ 
        void reset_nodeindptr() {
            this->current_node_color_ptrs = std::vector<ColorIDType>(this->num_nodes, 0);
            this->current_node_self_update_ptrs = std::vector<ColorIDType>(this->num_nodes, 0);
        }

        /*
        method: reset color table
        description: reset the color table to empty
        input: None
        output: None
        */
        void reset_color_table() {
            this->node_usage_table = NodeEdgeTableType(this->num_nodes);
            this->node_self_update_table = NodeEdgeTableType(this->num_nodes);
        }

        /*
        method: return current usage table
        description: return the current usage table and update table to Python
        input: None
        output: vector<vector<int>>, the current usage table and update table
        */
        std::vector<std::vector<int>> get_usage_table() {
            return this->node_usage_table;
        }

        /*
        method: return current self update table
        description: return the current usage table and update table to Python
        input: None
        output: vector<vector<int>>, the current usage table and update table
        */
        std::vector<std::vector<int>> get_update_table() {
            return this->node_self_update_table;
        }

        /*
        method: set current usage table
        description: set the current usage table and update table from Python
        input: vector<vector<int>>, the current usage table and update table
        output: None 
        */
        void set_usage_table(std::vector<std::vector<int>> usage_table) {
            this->node_usage_table = usage_table;
        }

        /*
        method: set current self update table
        description: set the current usage table and update table from Python
        input: vector<vector<int>>, the current usage table and update table
        output: None 
        */
        void set_update_table(std::vector<std::vector<int>> update_table) {
            this->node_self_update_table = update_table;
        }

        /*
        method: color_graph
        description: color every node by the edges it effects
        input: (optional) int, the number of maximum eid to consider for the coloring
        output: None
        */
        void color_graph(int min_eid=-1, int max_eid = -1) {
            // NodeIDType num_connected_nodes = indptr.size() - 1;
            // std::cout << "num_connected_nodes: " << num_connected_nodes << std::endl;
            // color the graph in parallel
if (this->full_edges){
    // if we consider full edges, we need to consider all the edges
    // #pragma omp parallel for num_threads(1) schedule(dynamic)
// #pragma omp parallel for num_threads(1) schedule(dynamic)
#pragma omp parallel for num_threads(this->num_threads) schedule(dynamic)
        for (NodeIDType i = 0; i < this->num_nodes; i++) {
            NodeIDType node = i;
            ColorSetType node_edges = ColorSetType();
            ColorSetType node_self_update_edges = ColorSetType();
            // get the edges related to the node
            EdgeIDType start_edge = this->indptr[node];
            EdgeIDType end_edge = this->indptr[node + 1];

            // for debug
            // if(i >= 3000){
            //     std::cout << "node: " << node << " start_edge: " << start_edge << " end_edge: " << end_edge << " max_eid: "<< max_eid<< std::endl;
            //     std::cout << "Edge id subset: ";
            //         for (auto i = start_edge; i < end_edge && i < eid.size(); ++i) {
            //             std::cout << eid[i] << " ";
            //         }
            //     std::cout << std::endl;
            // }
            if (this->multi_color_flag && this->node_trace[node] != 0){
                // std::cout << "multi color flag is enabled" << std::endl;
                start_edge = this->node_trace[node];
            }

            for (EdgeIDType j = start_edge; j < end_edge; j++) {
                if (min_eid != -1 && eid[j] < min_eid) {
                    // if the edge id is smaller than the minimum edge id, we do not consider it
                    // for debug
                    // if(i >= 3000){
                    //     std::cout << "skip edge: " << eid[j] << " since it is smaller than the minimum edge id " << min_eid << std::endl; 
                    // }
                    continue;
                }

                if (max_eid != -1 && eid[j] > max_eid) {
                    // if the edge id is larger than the maximum edge id, we do not consider it
                    // for debug
                    // if(i >= 3000){
                    //     std::cout << "skip edge: " << eid[j] << " since it is larger than the maximum edge id " << max_eid << std::endl; 
                    // }
                    // continue;
                    // this indicates that we only consider the edges that are smaller than the maximum edge id
                    if (this->multi_color_flag){
                        // if we enable multi color, we need to update the node trace
                        this->node_trace[node] = j;
                    }
                    // this->node_trace[node] = j;
                    break;
                }
                // insert the edge into the node edges
                node_edges.insert(eid[j]);
                // insert the edge into the node self update edges, this edge updates the node itself
                node_self_update_edges.insert(eid[j]);
                // find the neighbors of the node in the edge
                // if full edges then find from the second, else directly find from the edge index
                NodeIDType neighbor = this->edge_index[j].second;
                // if (this->full_edges){
                //     neighbor = this->edge_index[j].second;
                // }
                // else{
                //     neighbor = this->indices[j];
                // }
                // find the neighbor of the node from the idx
                // get the edges related to the neighbor
                EdgeIDType start_edge_other = this->indptr[neighbor];
                EdgeIDType end_edge_other = this->indptr[neighbor + 1];
                // find the most recent edges (most this->num_recent_edges) edges of the neighbor after the current edge and insert them into the node edges
                int count = 0;

                // for debug    
                // if(i >= 3000){
                //     std::cout << "Neighbor: " << neighbor << " start_edge: " << start_edge_other << " end_edge: " << end_edge_other << " max_eid: "<< max_eid<< std::endl;
                //     std::cout << "Edge id Neighbor's subset: ";
                //     for (auto i = start_edge_other; i < end_edge_other && i < eid.size(); ++i) {
                //         std::cout << eid[i] << " ";
                //     }
                //     std::cout << std::endl;
                // }

                for (EdgeIDType k = start_edge_other; k < end_edge_other; k++) {
                    if (min_eid != -1 && eid[k] < min_eid) {
                        // if the edge id is smaller than the minimum edge id, we do not consider it
                        // for debug
                        // if(i >= 3000){
                        //     std::cout << "skip edge: " << eid[k] << " since it is smaller than the minimum edge id " << min_eid << std::endl; 
                        // }
                        continue;
                    }
                    if(max_eid != -1 && eid[k] > max_eid) {
                        // if the edge id is larger than the maximum edge id, we do not consider it
                        // for debug
                        // if(i >= 3000){
                        //     std::cout << "skip edge: " << eid[k] << " since it is larger than the maximum edge id " << max_eid << std::endl; 
                        // }
                        break;
                    }
                    if (eid[k] > eid[j]) {
                        // we only consider the edges that happen after the current edge
                        node_edges.insert(eid[k]);
                        count++;
                    }
                    if (count >= this->num_recent_edges) {
                        break;
                    }
                }
            }
            // convert the node edges into a list
            std::vector<ColorIDType> node_edges_list(node_edges.begin(), node_edges.end());
            // insert the node edges into the node usage table
            this->node_usage_table[node] = node_edges_list;

            // convert the node self update edges into a list
            std::vector<ColorIDType> node_self_update_edges_list(node_self_update_edges.begin(), node_self_update_edges.end());
            // insert the node self update edges into the node self update table
            this->node_self_update_table[node] = node_self_update_edges_list;

            // for debug
            // if(i >= 3000){
            //     std::cout << "node: " << node << " node_edges_list: ";
            //     std::copy(node_edges_list.begin(), node_edges_list.end(), std::ostream_iterator<int>(std::cout, " "));
            //     std::cout << std::endl;

            //     std::cout << "Press Enter to continue...";
            //     std::cin.get();  // Waits for a single character input.
            //     std::cout << "Continuing execution...\n";
            // }

            // show size of the tables after coloring
            
        }

    }
    else{
        // if we do not consider full edges, we only consider the edges that are smaller than the maximum edge id
            // if we consider full edges, we need to consider all the edges
    // #pragma omp parallel for num_threads(1) schedule(dynamic)
// #pragma omp parallel for num_threads(1) schedule(dynamic)
#pragma omp parallel for num_threads(this->num_threads) schedule(dynamic)
        for (NodeIDType i = 0; i < this->num_nodes; i++) {
            NodeIDType node = i;
            ColorSetType node_edges = ColorSetType();
            ColorSetType node_self_update_edges = ColorSetType();
            // get the edges related to the node
            EdgeIDType start_edge = this->indptr[node];
            EdgeIDType end_edge = this->indptr[node + 1];

            // for debug
            // if(i >= 3000){
            //     std::cout << "node: " << node << " start_edge: " << start_edge << " end_edge: " << end_edge << " max_eid: "<< max_eid<< std::endl;
            //     std::cout << "Edge id subset: ";
            //         for (auto i = start_edge; i < end_edge && i < eid.size(); ++i) {
            //             std::cout << eid[i] << " ";
            //         }
            //     std::cout << std::endl;
            // }
            if (this->multi_color_flag && this->node_trace[node] != 0){
                // std::cout << "multi color flag is enabled" << std::endl;
                start_edge = this->node_trace[node];
            }

            for (EdgeIDType j = start_edge; j < end_edge; j++) {
                if (min_eid != -1 && eid[j] < min_eid) {
                    // if the edge id is smaller than the minimum edge id, we do not consider it
                    // for debug
                    // if(i >= 3000){
                    //     std::cout << "skip edge: " << eid[j] << " since it is smaller than the minimum edge id " << min_eid << std::endl; 
                    // }
                    continue;
                }

                if (max_eid != -1 && eid[j] > max_eid) {
                    // if the edge id is larger than the maximum edge id, we do not consider it
                    // for debug
                    // if(i >= 3000){
                    //     std::cout << "skip edge: " << eid[j] << " since it is larger than the maximum edge id " << max_eid << std::endl; 
                    // }
                    // continue;
                    // this indicates that we only consider the edges that are smaller than the maximum edge id
                    if (this->multi_color_flag){
                        // if we enable multi color, we need to update the node trace
                        this->node_trace[node] = j;
                    }
                    // this->node_trace[node] = j;
                    break;
                }
                // insert the edge into the node edges
                node_edges.insert(eid[j]);
                // insert the edge into the node self update edges, this edge updates the node itself
                node_self_update_edges.insert(eid[j]);
                // find the neighbors of the node in the edge
                // if full edges then find from the second, else directly find from the edge index
                NodeIDType neighbor = this->indices[j];
                // find the neighbor of the node from the idx
                // get the edges related to the neighbor
                EdgeIDType start_edge_other = this->indptr[neighbor];
                EdgeIDType end_edge_other = this->indptr[neighbor + 1];
                // find the most recent edges (most this->num_recent_edges) edges of the neighbor after the current edge and insert them into the node edges
                int count = 0;

                // for debug    
                // if(i >= 3000){
                //     std::cout << "Neighbor: " << neighbor << " start_edge: " << start_edge_other << " end_edge: " << end_edge_other << " max_eid: "<< max_eid<< std::endl;
                //     std::cout << "Edge id Neighbor's subset: ";
                //     for (auto i = start_edge_other; i < end_edge_other && i < eid.size(); ++i) {
                //         std::cout << eid[i] << " ";
                //     }
                //     std::cout << std::endl;
                // }

                for (EdgeIDType k = start_edge_other; k < end_edge_other; k++) {
                    if (min_eid != -1 && eid[k] < min_eid) {
                        // if the edge id is smaller than the minimum edge id, we do not consider it
                        // for debug
                        // if(i >= 3000){
                        //     std::cout << "skip edge: " << eid[k] << " since it is smaller than the minimum edge id " << min_eid << std::endl; 
                        // }
                        continue;
                    }
                    if(max_eid != -1 && eid[k] > max_eid) {
                        // if the edge id is larger than the maximum edge id, we do not consider it
                        // for debug
                        // if(i >= 3000){
                        //     std::cout << "skip edge: " << eid[k] << " since it is larger than the maximum edge id " << max_eid << std::endl; 
                        // }
                        break;
                    }
                    if (eid[k] > eid[j]) {
                        // we only consider the edges that happen after the current edge
                        node_edges.insert(eid[k]);
                        count++;
                    }
                    if (count >= this->num_recent_edges) {
                        break;
                    }
                }
            }
            // convert the node edges into a list
            std::vector<ColorIDType> node_edges_list(node_edges.begin(), node_edges.end());
            // insert the node edges into the node usage table
            this->node_usage_table[node] = node_edges_list;

            // convert the node self update edges into a list
            std::vector<ColorIDType> node_self_update_edges_list(node_self_update_edges.begin(), node_self_update_edges.end());
            // insert the node self update edges into the node self update table
            this->node_self_update_table[node] = node_self_update_edges_list;

            // for debug
            // if(i >= 3000){
            //     std::cout << "node: " << node << " node_edges_list: ";
            //     std::copy(node_edges_list.begin(), node_edges_list.end(), std::ostream_iterator<int>(std::cout, " "));
            //     std::cout << std::endl;

            //     std::cout << "Press Enter to continue...";
            //     std::cin.get();  // Waits for a single character input.
            //     std::cout << "Continuing execution...\n";
            // }

            // show size of the tables after coloring
            
        }
    }
}


        /*
        method: sample_batch
        description: sample a batch of edges from the graph depending on the coloring table, the sampling is done in parallel
                        First, we take a chunk of edges from the graph, and retrive the corresponding nodes.
                        Next, we check the latest event for each node, then get a global latest event as the final event for the batch.
                        Finally, we return the index of the final event.
        input: 
            root_nodes: a list of root nodes for the sampling
            start_edge_id: the start index of the edges for the batch
            step_size: the last observed edges start from start_edge_id to start_edge_id + step_size
            node_stable: bool, if True, ignore those stable events whose both nodes are stable, else ignore those stable events whose any node is stable
        output: int, the index of the final event
        */
        int sample_batch(
            std::vector<NodeIDType> root_nodes,
            EdgeIDType start_edge_id,
            torch::Tensor stable_flag,
            int num_colors,
            int minimal_batch_size,
            int step_size,
            bool node_stable = false
        ) {
            // get the chunk of edges from the graph
            auto end_edge_id = std::min(start_edge_id + step_size, this->num_edges);

            // get the minimal batch end edge id
            auto minimal_batch_end_edge_id = std::min(start_edge_id + minimal_batch_size, this->num_edges);

            // get the final event for each node by moving the current node indptr to its maximum limit defined by the num_colors, 
            // if and_stable ignore those stable events whose both nodes are stable, else ignore those stable events whose any node is stable
            auto final_event = end_edge_id;
            // std::cout << "final_event before: " << final_event << std::endl;

            int num_node_chunks = (root_nodes.size() + this->num_nodes_per_thread - 1) / this->num_nodes_per_thread;
            // declear a vector to store the last event for each node
            // std::vector<EdgeIDType> last_node_events(this->num_nodes, end_edge_id);

            if (node_stable){
                // in this case, we ignore those stable events on nodes that are stable, and the root nodes only contain the unstable nodes
                // we can do this in parallel, note that the final_event is a shared variable
// #pragma omp parallel for num_threads(1) reduction(min:final_event) schedule(dynamic)
#pragma omp parallel for num_threads(this->num_threads) reduction(min:final_event) schedule(dynamic)
                for (int chunk_id = 0; chunk_id < num_node_chunks; chunk_id++) {
                    int start_node_id = chunk_id * this->num_nodes_per_thread;
                    int end_node_id = std::min((chunk_id + 1) * this->num_nodes_per_thread, (int)root_nodes.size());
                    // for (size_t i = 0; i < root_nodes.size(); i++) {
                    for (int i = start_node_id; i < end_node_id; i++) {
                        NodeIDType node = root_nodes[i];
                        ColorListType node_edges = this->node_usage_table[node];
                        ColorIDType last_event_ptr = this->current_node_color_ptrs[node];
                        int node_edges_size = node_edges.size();
                        // std::cout << "node: " << node 
                        //                 << " start from " << last_event_ptr 
                        //                 << " as edge " << node_edges[last_event_ptr] 
                        //                 << " to at most " << last_event_ptr + num_colors 
                        //                 << " as edge " << node_edges[last_event_ptr + num_colors] 
                        //                 << " with size " << node_edges_size
                        //                 << std::endl;
                        //     std::copy(node_edges.begin(), node_edges.end(), std::ostream_iterator<int>(std::cout, " "));
                        //     std::cout << std::endl;

                        last_event_ptr = last_event_ptr + num_colors - 1;
                        if (last_event_ptr >= node_edges_size) {
                            // if the node is fully used, we do not need to use its last one to limit batch
                            continue;
                        }
                        // last_event_ptr = std::min(last_event_ptr + num_colors - 1, (ColorIDType)node_edges.size()-1);
                        // get the last event for the node
                        EdgeIDType last_node_event = node_edges[last_event_ptr];

                        // update final_event if the node_event is smaller
                        final_event = std::min(final_event, last_node_event);
                        // for debug
                        // std::cout << "last_event_ptr: " << last_event_ptr << " last_node_event: " << last_node_event << " final_event: " << final_event << std::endl;
                        // std::cout << "Press Enter to continue...";
                        // std::cin.get();  // Waits for a single character input.
                        // std::cout << "Continuing execution...\n";

                    }
                }
            }
            else{
                // in this case, we ignore those stable events whose both nodes are stable
                // we can do this in parallel, note that the final_event is a shared variable
#pragma omp parallel for num_threads(this->num_threads) reduction(min:final_event) schedule(dynamic)
// #pragma omp parallel for num_threads(1) reduction(min:final_event) schedule(dynamic)
                for (int chunk_id = 0; chunk_id < num_node_chunks; chunk_id++) {
                    int start_node_id = chunk_id * this->num_nodes_per_thread;
                    int end_node_id = std::min((chunk_id + 1) * this->num_nodes_per_thread, (int)root_nodes.size());
                    EdgeIDType final_event_chunk = end_edge_id;
                    // for (size_t i = 0; i < root_nodes.size(); i++) {
                    for (int i = start_node_id; i < end_node_id; i++) {
                        NodeIDType node = root_nodes[i];
                        ColorListType node_edges = this->node_usage_table[node];
                        ColorIDType last_event_ptr = this->current_node_color_ptrs[node];
                        // std::cout << "node: " << node 
                        //             << " start from " << last_event_ptr 
                        //             << " as edge " << node_edges[last_event_ptr] 
                        //             << " to at most " << last_event_ptr + num_colors 
                        //             << " as edge " << node_edges[last_event_ptr + num_colors] 
                        //             << " with size " << node_edges.size()
                        //             << std::endl;
                        // std::copy(node_edges.begin(), node_edges.end(), std::ostream_iterator<int>(std::cout, " "));
                        // std::cout << std::endl;
                        int count = 0;
                        bool node_done = false;
                        while (count < num_colors) {
                            if (static_cast<size_t>(last_event_ptr) >= node_edges.size()) {
                                // std::cout << "break due to size, the node is fully used no need to use its last one to limit batch" << std::endl;
                                node_done = true;
                                break;
                            }
                            EdgeIDType node_event = node_edges[last_event_ptr];
                            if (node_event >= end_edge_id) {
                                // std::cout << "break due to end_edge_id, the batch size exceed maximum step size" << std::endl;
                                break;
                            }
                            auto src = this->edge_index[node_event].first;
                            auto dst = this->edge_index[node_event].second;
                            if (stable_flag[src].item<bool>() && stable_flag[dst].item<bool>()) {
                                // if both nodes are stable, we do not take the event into account, but just move the pointer
                                // std::cout << "both nodes are stable, move the pointer but not take the event into account---" 
                                //             << " event_ptr: " << last_event_ptr
                                //             << " count: " << count << std::endl;
                                last_event_ptr++;
                                // count++;
                            } else {
                                // if any node is unstable, we take the event into account
                                // std::cout << "at least one node is unstable, move the pointer and count---" 
                                //             << " event_ptr: " << last_event_ptr
                                //             << " count: " << count << std::endl;
                                last_event_ptr++;
                                count++;
                            }
                        }
                        if (node_done) {
                            continue;
                        }
                        // get the last event for the node
                        last_event_ptr = std::min(last_event_ptr, (ColorIDType)node_edges.size()-1);
                        EdgeIDType last_node_event = node_edges[last_event_ptr];
                        // final_event = std::min(final_event, last_node_event);
                        final_event_chunk = std::min(final_event_chunk, last_node_event);
                    }
                    final_event = std::min(final_event, final_event_chunk);
                        // std::cout << "last_event_ptr: " << last_event_ptr << " last_node_event: " << last_node_event << " final_event: " << final_event << std::endl;
                        // std::cout << "Press Enter to continue...";
                        // std::cin.get();  // Waits for a single character input.
                        // std::cout << "Continuing execution...\n";
                }
            }
            
            // final_event = *std::min_element(last_node_events.begin(), last_node_events.end());
            // if the final event is smaller than the minimal batch end edge id, we need to move the final event to the minimal batch end edge id
            final_event = std::max(final_event, minimal_batch_end_edge_id);
            
            return final_event;
        }
        
        /*
        method: update_node_color_ptrs
        description: update the current node color pointers based on the final event
        input: 
            final_event: the index of the final event
            root_nodes: a list of root nodes to be updated
        output: None
        */
        void update_node_color_ptrs(EdgeIDType final_event, std::vector<NodeIDType> root_nodes) {
            
            // for debug: print the final event and the root nodes
            // std::cout << "final_event: " << final_event << std::endl;
            // std::cout << "root_nodes: ";
            // std::copy(root_nodes.begin(), root_nodes.end(), std::ostream_iterator<int>(std::cout, " "));
            // std::cout << std::endl;

            // update the current node color pointers, the update is done in parallel
#pragma omp parallel for num_threads(this->num_threads) schedule(dynamic)
// #pragma omp parallel for num_threads(1) schedule(dynamic)
        for (size_t i = 0; i < root_nodes.size(); i++) {
            NodeIDType node = root_nodes[i];
            ColorListType node_edges = this->node_usage_table[node];
            ColorListType node_self_update_edges = this->node_self_update_table[node];
            int node_edges_size = node_edges.size();
            int node_self_update_edges_size = node_self_update_edges.size();
            if (node_edges_size == 0) {
                // if the node has no edges, we set the current node color pointer to the end
                this->current_node_color_ptrs[node] = node_edges_size;
                // for debug
                // std::cout << "node: " << node << " current_node_color_ptrs[node]: " << this->current_node_color_ptrs[node] << " set to the end" << std::endl;
                continue;
            }
            // for debug
            // std::cout << "index: " << i 
            // << " node: " << node 
            // << " node_edges_size: " <<  node_edges_size 
            // << " start from color event " << this->current_node_color_ptrs[node] << " event id " << node_edges[this->current_node_color_ptrs[node]]
            // << " self update event " << this->current_node_self_update_ptrs[node] << " event id " << node_self_update_edges[this->current_node_self_update_ptrs[node]]
            // << " final_event: " << final_event
            // << std::endl;

            // now we set the self update pointer to the earliest event that is larger than the final event
            while (node_self_update_edges[this->current_node_self_update_ptrs[node]] < final_event && this->current_node_self_update_ptrs[node] < node_self_update_edges_size) {
                // for debug
                // std::cout << "node: " << node 
                //             << " update event: " << node_self_update_edges[this->current_node_self_update_ptrs[node]] 
                //             << " final_event: " << final_event << " increased" <<std::endl;
                this->current_node_self_update_ptrs[node]++;
            }

            // then we set the current node color pointer pointing to the color indicated by the self update pointer
            if(this->current_node_self_update_ptrs[node] >= node_self_update_edges_size){
                // there is no update later, we can set the current_node_color_ptrs to the end
                this->current_node_color_ptrs[node] = node_edges_size;
                // for debug
                // std::cout << "node: " << node << " current_node_color_ptrs[node]: " << this->current_node_color_ptrs[node] << " set to the end" << std::endl;
            } else {
                // we set the current_node_color_ptrs to the next self update pointer
                EdgeIDType self_update_event = node_self_update_edges[this->current_node_self_update_ptrs[node]]; // this is the eid of the updates to the node itself
                // for debug
                // std::cout << "Moving node: " << node << " to self_update_event: " << self_update_event << std::endl;
                while (node_edges[this->current_node_color_ptrs[node]] <= self_update_event && this->current_node_color_ptrs[node] < node_edges_size) {
                    this->current_node_color_ptrs[node]++;
                }
            }

            // avoid the case that the current node color pointer is larger than the node edges size
            this->current_node_color_ptrs[node] = std::min(this->current_node_color_ptrs[node], (ColorIDType)node_edges_size);

            // for debug
            // std::cout << "node: " << node << " current_node_color_ptrs[node]: " << this->current_node_color_ptrs[node] << std::endl;
            // std::cout << "Press Enter to continue...";
            // std::cin.get();  // Waits for a single character input.
            // std::cout << "Continuing execution...\n";

            // show memory cost of the tables in GB
        }
    }


        // ~ColoringSampler() {
        //     delete[] this->ts_ptr_lock;
        // }
        

};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Coloring Sampler for Dynamic Graphs";
    py::class_<ColoringSampler>(m, "ColoringSampler")
        .def(py::init<std::vector<EdgeIDType>, std::vector<EdgeType>, std::vector<NodeIDType>, std::vector<EdgeIDType>, int, int, int, int, int, int>())
        .def("color_graph", &ColoringSampler::color_graph)
        .def("sample_batch", &ColoringSampler::sample_batch)
        .def("sample_batch_nogil", &ColoringSampler::sample_batch, py::call_guard<py::gil_scoped_release>())
        .def("update_node_color_ptrs", &ColoringSampler::update_node_color_ptrs)
        .def("update_node_color_ptrs_nogil", &ColoringSampler::update_node_color_ptrs, py::call_guard<py::gil_scoped_release>())
        .def("reset_nodeindptr", &ColoringSampler::reset_nodeindptr)
        .def("reset_color_table", &ColoringSampler::reset_color_table)
        .def("get_usage_table", &ColoringSampler::get_usage_table)
        .def("get_update_table", &ColoringSampler::get_update_table)
        .def("set_usage_table", &ColoringSampler::set_usage_table)
        .def("set_update_table", &ColoringSampler::set_update_table)
        .def("enable_multi_color", &ColoringSampler::enable_multi_color)
        .def("disable_full_edges", &ColoringSampler::disable_full_edges)
        .def_readwrite("indptr", &ColoringSampler::indptr)
        .def_readwrite("edge_index", &ColoringSampler::edge_index)
        .def_readwrite("eid", &ColoringSampler::eid)
        .def_readwrite("node_usage_table", &ColoringSampler::node_usage_table)
        .def_readwrite("node_self_update_table", &ColoringSampler::node_self_update_table)
        .def_readwrite("current_node_color_ptrs", &ColoringSampler::current_node_color_ptrs)
        .def_readwrite("current_node_self_update_ptrs", &ColoringSampler::current_node_self_update_ptrs)
        .def_readwrite("num_nodes", &ColoringSampler::num_nodes)
        .def_readwrite("num_edges", &ColoringSampler::num_edges)
        .def_readwrite("num_hops", &ColoringSampler::num_hops)
        .def_readwrite("num_recent_edges", &ColoringSampler::num_recent_edges);
}