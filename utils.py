import torch
import os
import yaml
import dgl
import time
import pandas as pd
import numpy as np
import psutil
from time import perf_counter


TO_DGL_BLOCKS_PROFILE = False
TO_DGL_BLOCKS_PROFILE_SUMMARY = {
    'count': 0,
    'create_block_time': 0.0,
    'cuda_copy_time': 0.0,
    'total_time': 0.0,
    'combine_first_time': 0.0,
    'node_index_time': 0.0,
    'edge_index_time': 0.0,
    'node_cuda_time': 0.0,
    'edge_cuda_time': 0.0,
    'create_dgl_block_time': 0.0,
    'src_id_time': 0.0,
    'edge_dt_time': 0.0,
    'src_ts_time': 0.0,
    'id_time': 0.0,
}

def set_to_dgl_blocks_profiling(enabled=True, aggressive_profiling=False):
    global TO_DGL_BLOCKS_PROFILE
    TO_DGL_BLOCKS_PROFILE = enabled
    global AGGRESSIVE_PROFILING
    AGGRESSIVE_PROFILING = aggressive_profiling



def reset_to_dgl_blocks_profile():
    global TO_DGL_BLOCKS_PROFILE_SUMMARY
    TO_DGL_BLOCKS_PROFILE_SUMMARY = {
        'count': 0,
        'create_block_time': 0.0,
        'cuda_copy_time': 0.0,
        'total_time': 0.0,
        'combine_first_time': 0.0,
        'node_index_time': 0.0,
        'edge_index_time': 0.0,
        'node_cuda_time': 0.0,
        'edge_cuda_time': 0.0,
        'create_dgl_block_time': 0.0,
        'src_id_time': 0.0,
        'edge_dt_time': 0.0,
        'src_ts_time': 0.0,
        'id_time': 0.0,
    }


def print_to_dgl_blocks_profile():
    count = TO_DGL_BLOCKS_PROFILE_SUMMARY['count']
    create_block_time = TO_DGL_BLOCKS_PROFILE_SUMMARY['create_block_time']
    cuda_copy_time = TO_DGL_BLOCKS_PROFILE_SUMMARY['cuda_copy_time']
    total_time = TO_DGL_BLOCKS_PROFILE_SUMMARY['total_time']
    combine_first_time = TO_DGL_BLOCKS_PROFILE_SUMMARY['combine_first_time']
    node_index_time = TO_DGL_BLOCKS_PROFILE_SUMMARY['node_index_time']
    edge_index_time = TO_DGL_BLOCKS_PROFILE_SUMMARY['edge_index_time']
    node_cuda_time = TO_DGL_BLOCKS_PROFILE_SUMMARY['node_cuda_time']
    edge_cuda_time = TO_DGL_BLOCKS_PROFILE_SUMMARY['edge_cuda_time']
    create_dgl_block_time = TO_DGL_BLOCKS_PROFILE_SUMMARY['create_dgl_block_time']
    src_id_time = TO_DGL_BLOCKS_PROFILE_SUMMARY['src_id_time']
    edge_dt_time = TO_DGL_BLOCKS_PROFILE_SUMMARY['edge_dt_time']
    src_ts_time = TO_DGL_BLOCKS_PROFILE_SUMMARY['src_ts_time']
    id_time = TO_DGL_BLOCKS_PROFILE_SUMMARY['id_time']

    print('\t=== to_dgl_blocks profiling summary ===')
    print('\tblocks created: {:d}'.format(count))
    print('\tcreate_block_time: {:.6f}s cuda_copy_time: {:.6f}s total_to_dgl_blocks_time: {:.6f}s combine_first_time: {:.6f}s node_index_time: {:.6f}s edge_index_time: {:.6f}s node_cuda_time: {:.6f}s edge_cuda_time: {:.6f}s create_dgl_block_time: {:.6f}s src_id_time: {:.6f}s edge_dt_time: {:.6f}s src_ts_time: {:.6f}s id_time: {:.6f}s'.format(
        create_block_time, 
        cuda_copy_time, 
        total_time, 
        combine_first_time, 
        node_index_time, 
        edge_index_time, 
        node_cuda_time, 
        edge_cuda_time, 
        create_dgl_block_time, 
        src_id_time, 
        edge_dt_time, 
        src_ts_time, 
        id_time
    ))


def get_to_dgl_blocks_profile_summary():
    return TO_DGL_BLOCKS_PROFILE_SUMMARY


def check_memory_usage():
    # Getting % usage of virtual_memory ( 3rd field) and usage of virtual_memory in GB ( 4th field)
    print('RAM memory % used:', psutil.virtual_memory()[2], 'GB used:', psutil.virtual_memory()[3]/(1024*1024*1024))
    
def load_feat(d, rand_de=0, rand_dn=0, num_nodes=None, num_edges=None):
    node_feats = None
    if os.path.exists('DATA/{}/node_features.pt'.format(d)) and d != "MAG":
        node_feats = torch.load('DATA/{}/node_features.pt'.format(d))
        # if node_feats.dtype == torch.bool:
        if node_feats.dtype == torch.bool and d != "GDELT":
            print("convert node feature to float 32...")
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists('DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('DATA/{}/edge_features.pt'.format(d))
        # if edge_feats.dtype == torch.bool:
        if edge_feats.dtype == torch.bool and d != "GDELT" and d != "MAG":
            print("convert edge feature to float 32...")
            edge_feats = edge_feats.type(torch.float32)
    if rand_de > 0:
        print("random edge feature...")
        if d == 'LASTFM':
            edge_feats = torch.randn(1293103, rand_de)
            # edge_feats = torch.zeros(1293103, rand_de)
        elif d == 'MOOC':
            # edge_feats = torch.randn(411749, rand_de)
            edge_feats = torch.zeros(411749, rand_de)
        else:
            edge_feats = torch.zeros(num_edges, rand_de, dtype=torch.bool)
    if rand_dn > 0:
        print("random node feature...")
        if d == 'LASTFM':
            node_feats = torch.randn(1980, rand_dn)
            # node_feats = torch.zeros(1980, rand_dn)
        elif d == 'MOOC':
            # edge_feats = torch.randn(7144, rand_dn)
            node_feats = torch.zeros(7144, rand_dn)
        else:
            edge_feats = torch.zeros(num_nodes, rand_dn, dtype=torch.bool)
    return node_feats, edge_feats

def load_graph(d):
    df = pd.read_csv('DATA/{}/edges.csv'.format(d))
    g = np.load('DATA/{}/ext_full.npz'.format(d))
    return g, df

def parse_config(f):
    conf = yaml.safe_load(open(f, 'r'))
    sample_param = conf['sampling'][0]
    memory_param = conf['memory'][0]
    gnn_param = conf['gnn'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, train_param

def to_dgl_blocks(ret, hist, reverse=False, cuda=True):
    global TO_DGL_BLOCKS_PROFILE_SUMMARY
    mfgs = list()
    profile = TO_DGL_BLOCKS_PROFILE
    aggresive_profiling = AGGRESSIVE_PROFILING

    for r in ret:
        start = perf_counter()
        if not reverse:
            
            create_block_start = perf_counter()
            b = dgl.create_block((r.col(), r.row()), num_src_nodes=r.dim_in(), num_dst_nodes=r.dim_out())
            create_block_end = perf_counter()
            create_dgl_block_time = create_block_end - create_block_start

            src_id_start_time = perf_counter()
            b.srcdata['ID'] = torch.from_numpy(r.nodes())
            src_id_end_time = perf_counter()
            src_id_time = src_id_end_time - src_id_start_time


            edge_dt_start_time = perf_counter()
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():]
            edge_dt_end_time = perf_counter()
            edge_dt_time = edge_dt_end_time - edge_dt_start_time


            src_ts_start_time = perf_counter()
            b.srcdata['ts'] = torch.from_numpy(r.ts())
            src_ts_end_time = perf_counter()
            src_ts_time = src_ts_end_time - src_ts_start_time

        else:

            create_block_start = perf_counter()
            b = dgl.create_block((r.row(), r.col()), num_src_nodes=r.dim_out(), num_dst_nodes=r.dim_in())
            create_block_end = perf_counter()
            create_dgl_block_time = create_block_end - create_block_start

            src_id_start_time = perf_counter()
            b.dstdata['ID'] = torch.from_numpy(r.nodes())
            src_id_end_time = perf_counter()
            src_id_time = src_id_end_time - src_id_start_time

            edge_dt_start_time = perf_counter()
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_src_nodes():]
            edge_dt_end_time = perf_counter()
            edge_dt_time = edge_dt_end_time - edge_dt_start_time

            src_ts_start_time = perf_counter()
            b.dstdata['ts'] = torch.from_numpy(r.ts())
            src_ts_end_time = perf_counter()
            src_ts_time = src_ts_end_time - src_ts_start_time

        id_time_start = perf_counter()
        b.edata['ID'] = torch.from_numpy(r.eid())
        id_time_end = perf_counter()
        id_time = id_time_end - id_time_start

        create_block_end_time = perf_counter()

        cuda_start = perf_counter()
        if cuda:
            b = b.to('cuda:0')
            if aggresive_profiling and torch.cuda.is_available():
                torch.cuda.synchronize()
            mfgs.append(b)
        else:
            mfgs.append(b)
        cuda_end = perf_counter()
        end = perf_counter()


        if profile:
            TO_DGL_BLOCKS_PROFILE_SUMMARY['count'] += 1
            TO_DGL_BLOCKS_PROFILE_SUMMARY['create_block_time'] += (create_block_end_time - start)
            TO_DGL_BLOCKS_PROFILE_SUMMARY['cuda_copy_time'] += (cuda_end - cuda_start)
            TO_DGL_BLOCKS_PROFILE_SUMMARY['total_time'] += (end - start)
            TO_DGL_BLOCKS_PROFILE_SUMMARY['create_dgl_block_time'] += create_dgl_block_time
            TO_DGL_BLOCKS_PROFILE_SUMMARY['src_id_time'] += src_id_time
            TO_DGL_BLOCKS_PROFILE_SUMMARY['edge_dt_time'] += edge_dt_time
            TO_DGL_BLOCKS_PROFILE_SUMMARY['src_ts_time'] += src_ts_time
            TO_DGL_BLOCKS_PROFILE_SUMMARY['id_time'] += id_time

    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs

def to_dgl_blocks_ob(ret, hist, reverse=False, cuda=True):
    mfgs = list()
    print("len of ret: ", len(ret))
    p1 = time.time()
    for r in ret:
        p1_0 = time.time()
        if not reverse:
            b = dgl.create_block((r.col(), r.row()), num_src_nodes=r.dim_in(), num_dst_nodes=r.dim_out())
            b.srcdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():]
            b.srcdata['ts'] = torch.from_numpy(r.ts())
        else:
            b = dgl.create_block((r.row(), r.col()), num_src_nodes=r.dim_out(), num_dst_nodes=r.dim_in())
            b.dstdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_src_nodes():]
            b.dstdata['ts'] = torch.from_numpy(r.ts())
        b.edata['ID'] = torch.from_numpy(r.eid())
        p1_1 = time.time()
        if cuda:
            mfgs.append(b.to('cuda:0'))
        else:
            mfgs.append(b)
        p1_2 = time.time()
    p2 = time.time()
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    p3 = time.time()
    print("\tTime to create blocks: P1: ", p2 - p1, " P1_0: ", p1_1 - p1, " P1_1: ", p1_2 - p1_1, "P2:", p3 - p2)
    return mfgs


def node_to_dgl_blocks(root_nodes, ts, cuda=True):
    global TO_DGL_BLOCKS_PROFILE_SUMMARY
    mfgs = list()
    profile = TO_DGL_BLOCKS_PROFILE
    aggresive_profiling = AGGRESSIVE_PROFILING


    start = perf_counter()
    create_block_start = perf_counter()
    b = dgl.create_block(([],[]), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0])
    create_block_end = perf_counter()
    create_dgl_block_time = create_block_end - create_block_start

    src_id_start_time = perf_counter()
    b.srcdata['ID'] = torch.from_numpy(root_nodes)
    src_id_end_time = perf_counter()
    src_id_time = src_id_end_time - src_id_start_time

    edge_dt_start_time = perf_counter()
    b.edata['dt'] = torch.from_numpy(ts)
    edge_dt_end_time = perf_counter()
    edge_dt_time = edge_dt_end_time - edge_dt_start_time

    create_block_end_time = perf_counter()

    cuda_start = perf_counter()
    if cuda:
        mfgs.insert(0, [b.to('cuda:0')])
        if aggresive_profiling and torch.cuda.is_available():
                torch.cuda.synchronize()
    else:
        mfgs.insert(0, [b])
    cuda_end = perf_counter()
    end = perf_counter()

    
    if profile:
        TO_DGL_BLOCKS_PROFILE_SUMMARY['count'] += 1
        TO_DGL_BLOCKS_PROFILE_SUMMARY['create_block_time'] += (create_block_end_time - start)
        TO_DGL_BLOCKS_PROFILE_SUMMARY['cuda_copy_time'] += (cuda_end - cuda_start)
        TO_DGL_BLOCKS_PROFILE_SUMMARY['total_time'] += (end - start)
        TO_DGL_BLOCKS_PROFILE_SUMMARY['create_dgl_block_time'] += create_dgl_block_time
        TO_DGL_BLOCKS_PROFILE_SUMMARY['src_id_time'] += src_id_time
        TO_DGL_BLOCKS_PROFILE_SUMMARY['edge_dt_time'] += edge_dt_time

    return mfgs

def mfgs_to_cuda(mfgs):
    for mfg in mfgs:
        for i in range(len(mfg)):
            mfg[i] = mfg[i].to('cuda:0')
    return mfgs

def prepare_input(mfgs, node_feats, edge_feats, combine_first=False, pinned=False, nfeat_buffs=None, efeat_buffs=None, nids=None, eids=None):
    global TO_DGL_BLOCKS_PROFILE_SUMMARY
    profile = TO_DGL_BLOCKS_PROFILE
    aggresive_profiling = AGGRESSIVE_PROFILING

    # 1st step: combine first block source nodes
    if combine_first:
        combine_first_start = perf_counter()
        for i in range(len(mfgs[0])):
            if mfgs[0][i].num_src_nodes() > mfgs[0][i].num_dst_nodes():
                num_dst = mfgs[0][i].num_dst_nodes()
                ts = mfgs[0][i].srcdata['ts'][num_dst:]
                nid = mfgs[0][i].srcdata['ID'][num_dst:].float()
                nts = torch.stack([ts, nid], dim=1)
                unts, idx = torch.unique(nts, dim=0, return_inverse=True)
                uts = unts[:, 0]
                unid = unts[:, 1]
                # import pdb; pdb.set_trace()
                torch 
                b = dgl.create_block((idx + num_dst, mfgs[0][i].edges()[1]), num_src_nodes=unts.shape[0] + num_dst, num_dst_nodes=num_dst, device=torch.device('cuda:0'))
                b.srcdata['ts'] = torch.cat([mfgs[0][i].srcdata['ts'][:num_dst], uts], dim=0)
                b.srcdata['ID'] = torch.cat([mfgs[0][i].srcdata['ID'][:num_dst], unid], dim=0)
                b.edata['dt'] = mfgs[0][i].edata['dt']
                b.edata['ID'] = mfgs[0][i].edata['ID']
                mfgs[0][i] = b
        combine_first_end = perf_counter()
        combine_first_time = combine_first_end - combine_first_start

    # 2nd step: prepare input features
    t_idx = 0
    t_cuda = 0
    i = 0

    total_node_idx_time = 0.0
    total_node_cuda_time = 0.0
    if node_feats is not None:
        for b in mfgs[0]:
            if pinned:
                idx_time_start = perf_counter()
                if nids is not None:
                    idx = nids[i]
                else:
                    idx = b.srcdata['ID'].cpu().long()
                torch.index_select(node_feats, 0, idx, out=nfeat_buffs[i][:idx.shape[0]])
                idx_time_end = perf_counter()
                node_idx_time = idx_time_end - idx_time_start
                total_node_idx_time += node_idx_time

                cuda_start_time = perf_counter()

                #Fix to Run GDELT on CPU
                target_device = b.device
                is_cuda = (target_device.type == 'cuda')
                #b.srcdata['h'] = nfeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                b.srcdata['h'] = nfeat_buffs[i][:idx.shape[0]].to(target_device, non_blocking=is_cuda)
                i += 1

                # Synchronize to get accurate CUDA time measurement
                if aggresive_profiling and torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                cuda_end_time = perf_counter()
                node_cuda_time = cuda_end_time - cuda_start_time
                total_node_cuda_time += node_cuda_time

            else:
                idx_time_start = perf_counter()
                idx = b.srcdata['ID'].long().to(node_feats.device)
                srch = node_feats[idx].float()
                idx_time_end = perf_counter()
                node_idx_time = idx_time_end - idx_time_start
                total_node_idx_time += node_idx_time
                # srch = node_feats[b.srcdata['ID'].long()].float()
                # print("index device: ", b.srcdata['ID'].device, "node_feats device: ", node_feats.device)
                # print("srch shape: ", srch.shape, "device: ", srch.device)
                
                #Fix to Run GDELT on CPU
                cuda_start_time = time.time()
                target_device = b.device
                #b.srcdata['h'] = srch.cuda()
                b.srcdata['h'] = srch.to(target_device)


                if aggresive_profiling and torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                cuda_end_time = time.time()
                node_cuda_time = cuda_end_time - cuda_start_time
                total_node_cuda_time += node_cuda_time

    # 3rd step: prepare edge features
    i = 0
    total_edge_idx_time = 0.0
    total_edge_cuda_time = 0.0

    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                if b.num_src_nodes() > b.num_dst_nodes():
                    if pinned:

                        idx_time_start = perf_counter()

                        if eids is not None:
                            idx = eids[i]
                        else:
                            idx = b.edata['ID'].cpu().long()
                        torch.index_select(edge_feats, 0, idx, out=efeat_buffs[i][:idx.shape[0]])

                        idx_time_end = perf_counter()
                        
                        edge_idx_time = idx_time_end - idx_time_start
                        total_edge_idx_time += edge_idx_time

                        cuda_start_time = perf_counter()

                        #GDELT CPU fix
                        target_device = b.device
                        is_cuda = (target_device.type == 'cuda')
                        #b.edata['f'] = efeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                        b.edata['f'] = efeat_buffs[i][:idx.shape[0]].to(target_device, non_blocking=is_cuda)


                        i += 1

                        if aggresive_profiling and torch.cuda.is_available():
                            torch.cuda.synchronize()
                            
                        cuda_end_time = perf_counter()
                        edge_cuda_time = cuda_end_time - cuda_start_time
                        total_edge_cuda_time += edge_cuda_time
                    else:
                        # edge_feats_device = edge_feats.device
                        idx_time_start = perf_counter()
                        idx = b.edata['ID'].long().to(edge_feats.device)
                        srch = edge_feats[idx].float()
                        idx_time_end = perf_counter()

                        edge_idx_time = idx_time_end - idx_time_start
                        total_edge_idx_time += edge_idx_time

                        cuda_start_time = perf_counter()
                        # srch = edge_feats[b.edata['ID'].long()].float()

                        #GDELT CPU fix
                        target_device = b.device
                        #b.edata['f'] = srch.cuda()
                        b.edata['f'] = srch.to(target_device)

                        if aggresive_profiling and torch.cuda.is_available():
                            torch.cuda.synchronize()
                            
                        cuda_end_time = perf_counter()
                        edge_cuda_time = cuda_end_time - cuda_start_time

                        total_edge_cuda_time += edge_cuda_time

    if profile:
        TO_DGL_BLOCKS_PROFILE_SUMMARY['combine_first_time'] += combine_first_time if combine_first else 0.0
        TO_DGL_BLOCKS_PROFILE_SUMMARY['node_index_time'] += total_node_idx_time if node_feats is not None else 0.0
        TO_DGL_BLOCKS_PROFILE_SUMMARY['edge_index_time'] += total_edge_idx_time if edge_feats is not None else 0.0
        TO_DGL_BLOCKS_PROFILE_SUMMARY['node_cuda_time'] += total_node_cuda_time if node_feats is not None else 0.0
        TO_DGL_BLOCKS_PROFILE_SUMMARY['edge_cuda_time'] += total_edge_cuda_time if edge_feats is not None else 0.0

    return mfgs

def get_ids(mfgs, node_feats, edge_feats):
    nids = list()
    eids = list()
    if node_feats is not None:
        for b in mfgs[0]:
            nids.append(b.srcdata['ID'].long())
    if 'ID' in mfgs[0][0].edata:
        if edge_feats is not None:
            for mfg in mfgs:
                for b in mfg:
                    eids.append(b.edata['ID'].long())
    else:
        eids = None
    return nids, eids

def get_pinned_buffers(sample_param, batch_size, node_feats, edge_feats):
    pinned_nfeat_buffs = list()
    pinned_efeat_buffs = list()
    limit = int(batch_size * 3.3)
    if 'neighbor' in sample_param:
        for i in sample_param['neighbor']:
            limit *= i + 1
            if edge_feats is not None:
                for _ in range(sample_param['history']):
                    pinned_efeat_buffs.insert(0, torch.zeros((limit, edge_feats.shape[1]), pin_memory=True))
    if node_feats is not None:
        for _ in range(sample_param['history']):
            pinned_nfeat_buffs.insert(0, torch.zeros((limit, node_feats.shape[1]), pin_memory=True))
    return pinned_nfeat_buffs, pinned_efeat_buffs


def unique_based_on_last_appearance(lst, window_size):
    seen = set()
    result = []
    for item in reversed(lst):
        if item not in seen:
            result.append(item)
            seen.add(item)
        if len(result) == window_size:
            break
    return list(reversed(result))


def unique_based_on_last_appearance_prev(lst):
    seen = set()
    result = []
    for item in reversed(lst):
        if item not in seen:
            result.append(item)
            seen.add(item)
    return list(reversed(result))