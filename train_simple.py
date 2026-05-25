import argparse
import os, sys, math, yaml


parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--rand_edge_features', type=int, default=0, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=0, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
parser.add_argument('--ratio', type=float, default=0.5, help='ratio of values per batch to total values')
parser.add_argument('--epoch', type=int, default=-1, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=-1, help='batch size')

parser.add_argument('--chunk_num', type=int, default=1, help='number of chunks used for super larger dataset, the training will be divided into chunks')
parser.add_argument('--sampler_for_each_chunk', action='store_true', help='whether to use individual sampler for each chunk')
parser.add_argument('--enable_pickle', action='store_true', help='whether to pickle the color sampler')
parser.add_argument('--cache_dir', type=str, default='sampler_caches', help='cache directory for color sampler')
parser.add_argument('--use_full_edge', action='store_true', help='whether to use full edge index for color sampler')


parser.add_argument('--node_count', type=int, default=500, help='maximum number of concurrent event for a node')
parser.add_argument('--mode', type=str, default='baseline', help='optimization modes: baseline, degree, node_count')
parser.add_argument('--window_size', type=int, default=10, help='window size for heatmap')
parser.add_argument('--color_limit', type=int, default=-1, help='color limit for heatmap')
parser.add_argument('--color_stack_per_node', type=int, default=1, help='number of cycles for heatmap')
parser.add_argument('--color_num_hops', type=int, default=2, help='number of hops for colording range')

parser.add_argument('--adaptive_node_count', type=str, default="disable", help='adaptive count of each batch, the count will raise if mant nodes are stable')
parser.add_argument('--step_scale', type=int, default=10, help='step scale for adaptive node count')
parser.add_argument('--step_size', type=int, default=10, help='step size for adaptive node count')
parser.add_argument('--minimum_scale_factor', type=float, default=0.5, help='minimum scale factor for adaptive node count')
parser.add_argument('--max_batch_size', type=int, default=6000, help='maximum batch size for adaptive node count')

parser.add_argument('--freeze_similarity', type=float, default=0.75, help='freeze if similarity is higher than this value')
parser.add_argument('--freeze_any', action='store_true', help='freeze if any similarity is higher than threshold')
parser.add_argument('--freeze_window_size', type=int, default=2, help='window size for similarity history check')

parser.add_argument('--lr_scale_factor', type=float, default=1, help='learning rate scale factor while scaling')
parser.add_argument('--lr_scale_ceiling', type=float, default=100, help='learning rate scale ceiling')

parser.add_argument('--observing', action='store_true', help='whether to observe the data')
parser.add_argument('--lr_scale', action='store_true', help='whether to scale the learning rate')
parser.add_argument('--check_similarity', action='store_true', help='whether to check similarity between node updates')
parser.add_argument('--check_similarity_epoch', action='store_true', help='whether to check similarity between node updates')
parser.add_argument('--batch_freeze', action='store_true', help='whether to freeze if similarity is high')

parser.add_argument('--cheat_code', type=str, default='disabled', help='cheat code for coloring')
parser.add_argument('--ignore_stable', action='store_true', help='ignore stable nodes')
parser.add_argument('--save_count', action='store_true', help='save count of each batch')

parser.add_argument('--observe_batch_latency', action='store_true', help='observe batch latency')
parser.add_argument('--observe_batch_utilization', action='store_true', help='observe batch utilization')

parser.add_argument('--batch_level_log', action='store_true', help='whether to log batch level information')
parser.add_argument('--profile_stable', action='store_true', help='whether to profile stable nodes')
parser.add_argument('--profile_to_dgl_blocks', action='store_true', help='whether to enable fine-grained to_dgl_blocks profiling')
parser.add_argument('--adaptive_update', action='store_true', help='whether to use adaptive update based on node similarity')
parser.add_argument('--extra_config', type=str, default='', help='path to extra config parameters for the trainer/batching, e.g., --extra_config "config/adapt_exp/WIKI/TGN.yml"')

args=parser.parse_args()


# TRAIN_SIMPLE: use yaml config file to override parameters
if args.extra_config != '':
    if not os.path.exists(args.extra_config):
        print("extra config file does not exist:", args.extra_config)
        sys.exit(1)
    with open(args.extra_config, 'r') as f:
        extra_config = yaml.safe_load(f)
    for key, value in extra_config.items():
        if not hasattr(args, key):
            print(f"[INFO] Adding new attribute from YAML: '{key}' = {value}")
        setattr(args, key, value)
    print("extra config loaded:", extra_config)


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

print("importing necessary packages...")

import torch
import time
import random
import dgl
import csv
import pickle
import numpy as np

print("importing customized packages...")

from modules import *
from sampler import *
from utils import *
from memorys import *
from color_sampler import *
from adaptive_updater import *
from prefetch_sampler import PrefetchManager
from sklearn.metrics import average_precision_score, roc_auc_score

import nvtx
from matplotlib import pyplot as plt
from tqdm import tqdm
# from node_freeze import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

print("setting seed...")
set_seed(0)

if args.profile_to_dgl_blocks:
    print("[INFO] Enabling fine-grained to_dgl_blocks profiling")
    set_to_dgl_blocks_profiling(True)
    reset_to_dgl_blocks_profile()

g, df = load_graph(args.data)
print("graph loaded...")

PROFILE_INTERVAL = 20
# EFFICIENT_PROFILE = False
if args.data == 'WIKI_TALK':
    PROFILE_INTERVAL = 10
elif args.data == 'STACK_OVERFLOW':
    PROFILE_INTERVAL = 5
    set_efficient_profiling()
else:
    PROFILE_INTERVAL = 20



if args.data == 'WIKI_TALK' or args.data == 'STACK_OVERFLOW' or args.data == 'MAG':
    num_edges = len(df)
    num_nodes = max(df['src'].max(), df['dst'].max()) + 1
    node_feats, edge_feats = load_feat(args.data, args.rand_edge_features, args.rand_node_features, num_nodes, num_edges)
else:
    num_edges = len(df)
    num_nodes = max(df['src'].max(), df['dst'].max()) + 1
    node_feats, edge_feats = load_feat(args.data, args.rand_edge_features, args.rand_node_features)
print("feature loaded...")
print("check memory usage after loading features...")
check_memory_usage()

sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]
# print("train edge intermediate", df[df['ext_roll'].gt(0)])
# print("val edge intermediate", df[df['ext_roll'].gt(1)])
# print("train_edge_end", train_edge_end, "val_edge_end", val_edge_end)
# sys.exit()
if 'validation_batch_size' in train_param:
    val_batch_size = train_param['validation_batch_size']
else:
    val_batch_size = train_param['batch_size']


if args.epoch != -1:
    train_param['epoch'] = args.epoch
if args.batch_size != -1:
    train_param['batch_size'] = args.batch_size


# CHEAT_CODE = "baseline" # "baseline"---make coloring behave like baseline; otherwise, make coloring behave like island
CHEAT_CODE = args.cheat_code
IGNORE_STABLE = args.ignore_stable
SAVE_COUNT = args.save_count
LR_SCALE_FACTOR = args.lr_scale_factor
LR_SCALE_CEILING = args.lr_scale_ceiling
SIM_THRESHOLD = args.freeze_similarity
SIM_WINDOW = args.freeze_window_size
SIM_ANY = args.freeze_any
MAX_BACTH_SIZE = args.max_batch_size
NUM_COLORS = args.node_count if memory_param['type'] != 'none' else args.node_count
total_coloring_time = 0
if args.step_size == 0:
    num_edges = len(df)
    num_batches = num_edges // train_param['batch_size']
    STEP_SIZE = num_batches // args.step_scale
else:
    STEP_SIZE = args.step_size
print("STEP_SIZE", STEP_SIZE, "STEP_SCALE", args.step_scale, "MINIMUM_SCALE_FACTOR", args.minimum_scale_factor)
count_path ="../results/counts/"
batch_max_color_counts_total = dict()

# if args.observing and SAVE_COUNT:
#     batch_max_color_counts_total = {"test":{}, "train":{}}
#     file_name = "test_0.1.pkl"
#     file_path = count_path + file_name
#     print("saving batch_max_color_counts to", file_path)
#     with open(file_path, 'wb') as f:
#         pickle.dump(batch_max_color_counts_total, f)

# sys.exit()

def get_inductive_links(df, train_edge_end, val_edge_end):
    train_df = df[:train_edge_end]
    test_df = df[val_edge_end:]
    
    total_node_set = set(np.unique(np.hstack([df['src'].values, df['dst'].values])))
    train_node_set = set(np.unique(np.hstack([train_df['src'].values, train_df['dst'].values])))
    new_node_set = total_node_set - train_node_set
    
    del total_node_set, train_node_set

    inductive_inds = []
    for index, (_, row) in enumerate(test_df.iterrows()):
        if row.src in new_node_set or row.dst in new_node_set:
            inductive_inds.append(val_edge_end+index)
    
    print('Inductive links', len(inductive_inds), len(test_df))
    return [i for i in range(val_edge_end)] + inductive_inds

if args.use_inductive:
    inductive_inds = get_inductive_links(df, train_edge_end, val_edge_end)
    df = df.iloc[inductive_inds]


print("creating model...")
gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
combine_first = False
print("gnn_dim_node", gnn_dim_node, "gnn_dim_edge", gnn_dim_edge)
if 'combine_neighs' in train_param and train_param['combine_neighs']:
    combine_first = True
model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first).cuda()
mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None
creterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
# ALL_GPU = False
ALL_GPU = True
if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
    if node_feats is not None:
        node_feats = node_feats.cuda()
    if edge_feats is not None:
        edge_feats = edge_feats.cuda()
    if mailbox is not None:
        mailbox.move_to_gpu()
    ALL_GPU = True

print("check memory usage after creating model...")
check_memory_usage()

sampler = None
if not ('no_sample' in sample_param and sample_param['no_sample']):
    print('Creating sampler...')
    sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy']=='recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))
print("sampler is None?", sampler is None)
print("g['indptr'].shape", g['indptr'].shape, "g['indices'].shape", g['indices'].shape, "g['eid'].shape", g['eid'].shape, "g['ts'].shape", g['ts'].shape)
print("training index", train_edge_end, "validation index", val_edge_end, "total index", df.shape[0])

print("check memory usage after creating edge sampler...")
check_memory_usage()

# print("==================node index pointer==================")
# print(g['indptr']) 
# input("Press Enter to continue...")
# print("==================dest node index==================")
# print(g['indices'])
# input("Press Enter to continue...")
# print("==================edge index==================")
# print(g['eid'])
# input("Press Enter to continue...")

color_sampler = None
if args.sampler_for_each_chunk:
        sampler_dict = dict()
if args.mode == "observing" or args.mode == "batch_stable_freezing" or args.mode == "batch_stable_freezing_large" or args.mode == "observing_large":
    training_df = df[:train_edge_end]

    # for debugging
    real_node_count = g['indptr'][1:] - g['indptr'][:-1]
    print("real_node_count", real_node_count.shape, "max", np.max(real_node_count), "min", np.min(real_node_count), "mean", np.mean(real_node_count))
    
    print("training_df", training_df.shape, "preparing graph data...")
    # indptr, edge_index, eid, num_nodes, num_edges = resolve_graph(training_df, node_num=g["indptr"].shape[0])

    if args.use_full_edge:
        indptr, edge_index, eid, num_nodes, num_edges = resolve_full_graph(g["indptr"], g["indices"], g["eid"], df=df)
        # print("indptr", indptr.shape, "edge_index", edge_index.shape, "eid", len(eid), "num_nodes", num_nodes, "num_edges", num_edges)
        indices = np.array([0,0], dtype=np.int32)
    else:
        print("using original graph data, no need to resolve full graph")
        indptr = g["indptr"]
        # create fake edge index, we do not use it just for convenience...
        src_nodes = np.array([0,0], dtype=np.int32)
        edge_index = np.stack([src_nodes,src_nodes], axis=0).T
        indices = g["indices"]
        eid = g["eid"]
        num_nodes = g["indptr"].shape[0]
        num_edges = len(g["eid"]) 
    # for debugging
    # print("indptr", indptr)
    # my_node_count = indptr[1:] - indptr[:-1]
    # my_node_count = my_node_count.numpy()
    # print("my_node_count", my_node_count)
    # print("my_node_count", my_node_count.shape, "max", np.max(my_node_count), "min", np.min(my_node_count), "mean", np.mean(my_node_count))
    # non_zero_node_count = my_node_count[my_node_count > 0]
    # print("non_zero_node_count", non_zero_node_count.shape, "max", np.max(non_zero_node_count), "min", np.min(non_zero_node_count), "mean", np.mean(non_zero_node_count))
    # input("Press Enter to continue...")

    num_nodes = g["indptr"].shape[0]
    print("indptr", indptr.shape, "edge_index", edge_index.shape, "indices", indices.shape,"eid", len(eid), "num_nodes", num_nodes, "num_edges", num_edges, "use full list?", args.use_full_edge)
    edge_index = list(edge_index)
    # indptr = list(indptr.numpy())
    indptr = list(indptr)
    indices = list(indices)
    eid = list(eid)
    num_nodes = int(num_nodes)
    num_edges = int(num_edges)

    # print("edge_index", edge_index)
    # input("building color batch sampler...")
    # print("building color batch sampler...")
    if args.sampler_for_each_chunk:
        print("build parallel color batch sampler...")
        if args.enable_pickle:
            print("enabling pickle...")
            if not os.path.isdir(args.cache_dir):
                os.mkdir(args.cache_dir)
                print("cache directory created at:", args.cache_dir)
            else:
                print("cache directory exists at:", args.cache_dir)
        color_sampler = MultiColorBatchSampler(indptr, edge_index, indices, eid, num_nodes, num_edges, 
                                      num_colors=NUM_COLORS, 
                                      num_recent_edges=args.window_size,
                                      num_hops=args.color_num_hops,
                                      decay_type=args.adaptive_node_count,
                                      decay_step=STEP_SIZE,
                                      decay_factor=args.step_scale,
                                      minimum_scale_factor=args.minimum_scale_factor,
                                      chunk_num=args.chunk_num,
                                      enable_pickle=args.enable_pickle,
                                      cache_dir=args.cache_dir)
    else:
        print("use original color batch sampler...")
        color_sampler = ColorBatchSampler(indptr, edge_index, indices, eid, num_nodes, num_edges, 
                                        num_colors=NUM_COLORS, 
                                        num_recent_edges=args.window_size,
                                        num_hops=args.color_num_hops,
                                        decay_type=args.adaptive_node_count,
                                        decay_step=STEP_SIZE,
                                        decay_factor=args.step_scale,
                                        minimum_scale_factor=args.minimum_scale_factor,
                                        use_full_edge=args.use_full_edge)


    if args.mode != "batch_stable_freezing_large" and args.mode != "observing_large":
        # for large scale, we color on the fly
        print("coloring...")
        t_color_st = time.time()
        # color_sampler.color_graph()
        # end_edge_index = df.loc[train_edge_end-1, "Unnamed: 0"]
        # print("color graph with edges before event id: ", end_edge_index, "at row index: ", train_edge_end-1)
        color_sampler.color_graph(train_edge_end)
        # print("unlimited coloring!")
        # color_sampler.color_graph()
        t_color_ed = time.time()
        total_coloring_time += t_color_ed - t_color_st
        print("coloring time", t_color_ed - t_color_st)
    # elif args.sampler_for_each_chunk:
    #     print("build parallel color batch sampler...")
    #     if args.enable_pickle:
    #         print("enabling pickle...")
    #         if not os.path.isdir(args.cache_dir):
    #             os.mkdir(args.cache_dir)
    #             print("cache directory created at:", args.cache_dir)
    #         else:
    #             print("cache directory exists at:", args.cache_dir)
    #     color_sampler = MultiColorBatchSampler(indptr, edge_index, eid, num_nodes, num_edges, 
    #                                   num_colors=NUM_COLORS, 
    #                                   num_recent_edges=args.window_size,
    #                                   num_hops=args.color_num_hops,
    #                                   decay_type=args.adaptive_node_count,
    #                                   decay_step=STEP_SIZE,
    #                                   decay_factor=args.step_scale,
    #                                   minimum_scale_factor=args.minimum_scale_factor,
    #                                   chunk_num=args.chunk_num,
    #                                   enable_pickle=args.enable_pickle,
    #                                   cache_dir=args.cache_dir)
    else:
        print("coloring on the fly...")

    if args.freeze_any:
        color_sampler.set_node_stable_mode(True)

print("check memory usage after creating batch sampler...")
check_memory_usage()

if args.use_inductive:
    print('Creating inductive sampler...')
    test_df = df[val_edge_end:]
    inductive_nodes = set(test_df.src.values).union(test_df.src.values)
    print("inductive nodes", len(inductive_nodes))
    neg_link_sampler = NegLinkInductiveSampler(inductive_nodes)
else:
    print('Creating negative sampler...')
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)

def eval(mode='val'):
    neg_samples = 1
    model.eval()
    # val_losses = list()
    aps = list()
    aucs_mrrs = list()
    if mode == 'val':
        eval_df = df[train_edge_end:val_edge_end]
    elif mode == 'test':
        eval_df = df[val_edge_end:]
        neg_samples = args.eval_neg_samples
    elif mode == 'train':
        eval_df = df[:train_edge_end]
    with torch.no_grad():
        total_loss = 0
        print("eval_batch_size", val_batch_size)
        
        for _, rows in eval_df.groupby(eval_df.index // val_batch_size):
            root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows) * neg_samples)]).astype(np.int32)
            ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = len(rows) * 2
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'], cuda=ALL_GPU)
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=ALL_GPU)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            pred_pos, pred_neg = model(mfgs, neg_samples=neg_samples)
            total_loss += creterion(pred_pos, torch.ones_like(pred_pos)) * pred_pos.size(0)
            total_loss += creterion(pred_neg, torch.zeros_like(pred_neg)) * pred_neg.size(0)
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            aps.append(average_precision_score(y_true, y_pred))
            if neg_samples > 1:
                aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float))
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))
            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True, cuda=ALL_GPU)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=neg_samples)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)
        if mode == 'val':
            val_losses.append(float(total_loss))
    ap = float(torch.tensor(aps).mean())
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs).mean())
    return ap, auc_mrr

if not os.path.isdir('models'):
    os.mkdir('models')
if args.model_name == '':
    path_saver = 'models/{}_{}.pkl'.format(args.data, time.time())
else:
    path_saver = 'models/{}.pkl'.format(args.model_name)
best_ap = 0
best_e = 0
best_auc = 0
val_losses = list()
group_indexes = list()
epoch_group_indexes = list()
# baseline = True
# TRAIN_VALUE_PER_BATCH = train_param['batch_size'] * 80 * 2
# create group indexes
# if baseline:
group_indexes.append(np.array(df[:train_edge_end].index // train_param['batch_size']))
print("group_indexes", group_indexes[0])


############################################
# get training index
############################################
training_df = df[:train_edge_end]
src_index = np.array(training_df.src.values.tolist()).astype(np.int32)
dst_index = np.array(training_df.dst.values.tolist()).astype(np.int32)
train_edge_index = np.stack((src_index, dst_index))



total_train_time = 0
# average_batch_size = 0
total_batch_count = 0
total_batch_sum = 0



observing = args.observing
observing = False
# print("observing?", observing)
print("observing?", observing, "SIM_THRESHOLD", SIM_THRESHOLD, "SIM_WINDOW", SIM_WINDOW, "SIM_ANY", SIM_ANY, "MAX_BACTH_SIZE", MAX_BACTH_SIZE)

if args.batch_freeze:
    print("enabling batch freeze...")
    enable_batch_freeze()

# node_number = int(max(np.array(df[:train_edge_end].src.values.tolist() + df[:train_edge_end].dst.values.tolist())))
node_number = num_nodes if num_nodes is not None else int(max(np.array(df[:train_edge_end].src.values.tolist() + df[:train_edge_end].dst.values.tolist())))


#########################################
# EXPERIMENTAL: adaptive_updater---enable/disable adaptive updater
#########################################
adaptive_updater = None
if args.adaptive_update:
    adaptive_updater = Adaptive_Update_Controller(node_num=node_number,
                                      freeze_threshold=args.adaptive_update_similarity)
#########################################

if 'reorder' in train_param:
    # random chunk shceduling
    reorder = train_param['reorder']
    group_idx = list()
    for i in range(reorder):
        group_idx += list(range(0 - i, reorder - i))
    group_idx = np.repeat(np.array(group_idx), train_param['batch_size'] // reorder)
    group_idx = np.tile(group_idx, train_edge_end // train_param['batch_size'] + 1)[:train_edge_end]
    group_indexes.append(group_indexes[0] + group_idx)
    base_idx = group_indexes[0]
    for i in range(1, train_param['reorder']):
        additional_idx = np.zeros(train_param['batch_size'] // train_param['reorder'] * i) - 1
        group_indexes.append(np.concatenate([additional_idx, base_idx])[:base_idx.shape[0]])


for e in range(train_param['epoch']):
    print('Epoch {:d}:'.format(e))
    print("check memory usage before each epoch...")
    check_memory_usage()
    # input("Press Enter to continue...")
    time_sample = 0
    time_prep = 0
    time_model = 0
    time_mem = 0
    time_tot = 0
    total_loss = 0
    if args.profile_to_dgl_blocks:
        reset_to_dgl_blocks_profile()

    ########################################
    # EXPERIMENTAL: adaptive_updater---enable/disable adaptive updater
    ########################################
    if args.adaptive_update and adaptive_updater is not None:
        adaptive_updater.enable()
    #########################################


    ########################################
    # stats init
    ########################################
    unique_average = []
    unique_max = []
    unique_std = []
    loss_list = []
    batch_latency = []
    model_latency = []
    other_latency = []
    batch_sizes = []
    prep_time_breakdown = {
        "to_dgl_blocks": 0,
        "prepare_input": 0,
        "mailbox_prep": 0,
        "mailbox_update": 0,
        "batch_postprocessing": 0,
    }
    color_time_breakdown = {"forming_batch": 0, "coloring": 0, "model training": 0, "record memory": 0, "others": 0}
    batching_time_breakdown = {"sampling":0, "updating_indptr":0, "updating_stable_flag":0, "others":0}
    estimated_prep_times = {"initial_to_dgl_blocks": 0, 
                            "prepare_input": 0, 
                            "mailbox_prep": 0, 
                            "post_to_dgl_blocks": 0,
                            "mailbox_update": 0, 
                            "updating_indptr_and_stable_flag": 0}
    #Variables to determine the stable flag ratio
    flip_ratio_log = []
    prev_stable_flag = None


    ########################################
    # batching sampler purpose: check if we can use memorized results
    ########################################
    training_df = df[:train_edge_end]
    if e != 0 and color_sampler is not None and args.mode != 'batch_stable_freezing_large' and args.mode != 'observing_large':
        if mailbox is not None:
            recent_node_stable = mailbox.get_full_node_stable_flag()
            recent_node_stable = recent_node_stable.cpu()
            color_sampler.check_if_use_memory(train_edge_index, recent_node_stable)
            color_sampler.record_node_stable_flag(new_stable_flag=recent_node_stable)
        else:
            color_sampler.set_use_memory(True)
    if color_sampler is not None:
        color_sampler.reset()
    # if e != 0 and args.mode == 'batch_stable_freezing_large':
    #     for key, color_sampler in sampler_dict.items():
    #         color_sampler.reset()
            

    # if args.check_similarity and e % 20 == 0:
    if args.check_similarity and e % PROFILE_INTERVAL == 0:
        enable_profiling()
        reset_profile()

    # if args.check_similarity_epoch and e < 10:
    if args.check_similarity_epoch:
        enable_profiling()
        reset_profile()


    # training
    model.train()
    if sampler is not None:
        sampler.reset()
    if mailbox is not None:
        mailbox.reset()
        model.memory_updater.last_updated_nid = None


    group_idx = group_indexes[random.randint(0, len(group_indexes) - 1)]
    
    if args.mode == 'batch_stable_freezing':
        # mailbox.set_history_recorder(window_size=SIM_WINDOW)
        if mailbox is not None:
            mailbox.set_stablize_recorder(window_size=SIM_WINDOW)
        ptr_start = 0
        ptr_end = ptr_start
        batch_count = 0
        final_group_idx = np.zeros(train_edge_end)


        training_df = df[:train_edge_end]
        cur_batch = 0
        cur_batch_count = 0
        cur_color_dict = dict()
        cur_color_list = []

        color_counts_dict = dict()
        batch_max_color_counts = list()

        _use_prefetch = getattr(args, 'prefetch', False) and sampler is not None
        prefetch = PrefetchManager(sampler) if _use_prefetch else None

        while ptr_start < train_edge_end:
            ########################################
            # batching block: moving ptrs and sample batch
            ########################################
            t_forming_batch_s = time.time()
            # color_sampler.update_node_indptr(ptr_start, model.memory_updater.last_updated_nid)
            ptr_end = color_sampler.sample_batch(training_df,
                                                 start_event_id=ptr_start,
                                                 batch_index=cur_batch,
                                                 minimal_batch_size=train_param['batch_size'],
                                                 step_size=MAX_BACTH_SIZE)
            color_time_breakdown["forming_batch"] += time.time() - t_forming_batch_s
            batching_time_breakdown["sampling"] += time.time() - t_forming_batch_s


            #########################################
            # model training block
            #########################################

            #########################################
            # EXPERIMENTAL: adaptive_updater (record up to date stable flag)---get node stable flag indicating whether the node is stable
            ########################################
            if args.adaptive_update and adaptive_updater is not None:
                node_stable_flag = mailbox.get_full_node_stable_flag() if mailbox is not None else None
                if node_stable_flag is not None and args.batch_level_log:
                    print("node_stable_flag shape", node_stable_flag.shape, "stable count", torch.sum(node_stable_flag).item(), "total nodes", node_stable_flag.shape[0])
                adaptive_updater.set_stable_record(node_stable_flag)
            #########################################
            
            
            cur_batch += 1
            # print("batch {}, cur batch count {}".format(cur_batch, ptr_end - ptr_start))
            t_training_start = time.time()
            
            # update batch count
            rows = df.iloc[ptr_start:ptr_end]
            final_group_idx[ptr_start:ptr_end] = batch_count
            batch_count += 1
            total_batch_count += 1
            total_batch_sum += len(rows)
            batch_sizes.append(len(rows))
            # print("batch {}, cur batch count {}".format(cur_batch, len(rows)))
            if args.lr_scale:
                lr = min(train_param['lr'] * len(rows) / train_param['batch_size'] * LR_SCALE_FACTOR, train_param['lr'] * LR_SCALE_CEILING)
                # print("batch size", len(rows), "preset batch size", train_param['batch_size'], "lr", lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            color_time_breakdown["others"] += time.time() - t_training_start

            # model forward
            t_tot_s = time.time()
            #########################################
            # EXPERIMENTAL: adaptive_updater(reduce stable root nodes)---based on node stable flag, we can reduce the number of root nodes
            ########################################
            if args.adaptive_update and adaptive_updater is not None:
                try:
                    rows = adaptive_updater.elastic_row_filer(rows)
                except Exception as e:
                    print("adaptive_updater error:", e)
                    rows = rows
            #########################################

            root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
            ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)

            #########################################
            # EXPERIMENTAL: adaptive_updater(not used for now)---based on node stable flag, we can reduce the number of root nodes
            ########################################
            # if args.adaptive_update is not None:
            #     full_root_nodes = root_nodes.copy()
            #     try:
            #         root_nodes, pos_root_end_reduced = adaptive_updater(root_nodes)
            #     except Exception as e:
            #         print("adaptive_updater error:", e)
            #         root_nodes = root_nodes
            #########################################

            pos_root_end = root_nodes.shape[0] * 2 // 3
            # EXPERIMENTAL(not used for now): adaptive_updater---minors
            # if pos_root_end_reduced is not None:
            #     pos_root_end = pos_root_end_reduced

            # kick off sampler in background — unique_pos computation below overlaps with it
            if prefetch is not None:
                _no_neg = 'no_neg' in sample_param and sample_param['no_neg']
                prefetch.schedule(
                    root_nodes[:pos_root_end] if _no_neg else root_nodes,
                    ts[:pos_root_end] if _no_neg else ts,
                )

            unique_pos_root_nodes = np.unique(root_nodes[:pos_root_end])
            if sampler is not None:
                ret = prefetch.get() if prefetch is not None else None
                if ret is None:
                    if 'no_neg' in sample_param and sample_param['no_neg']:
                        # EXPERIMENTAL(not used for now): adaptive_updater---minors
                        # if pos_root_end_reduced is not None:
                        #     pos_root_end = pos_root_end_reduced
                        # else:
                        pos_root_end = root_nodes.shape[0] * 2 // 3
                        sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                    else:
                        sampler.sample(root_nodes, ts)
                    ret = sampler.get_ret()
                # time_sample += ret[0].sample_time()
                time_sample += time.time() - t_tot_s
            t_prep_s = time.time()
            # if e == 15:
            #     to_dgl_blocks_ob(ret, sample_param['history'])
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'], cuda=ALL_GPU)
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=ALL_GPU)
            
            #Valiating the time for to_dgl_blocks conversion
            estimated_prep_times["initial_to_dgl_blocks"] += time.time() - t_prep_s

            prep_time_breakdown["to_dgl_blocks"] += time.time() - t_prep_s
            t_prepare_s = time.time()
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            prep_time_breakdown["prepare_input"] += time.time() - t_prepare_s

            #Valiating the time for prepare_input conversion
            estimated_prep_times["prepare_input"] += time.time() - t_prepare_s

            t_mailbox_prep_s = time.time()
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            prep_time_breakdown["mailbox_prep"] += time.time() - t_mailbox_prep_s

            #valuating the time for mailbox preparation
            estimated_prep_times["mailbox_prep"] += time.time() - t_mailbox_prep_s

            time_prep += time.time() - t_prep_s
            ########################################
            # check input mails
            ########################################
            t_model_s = time.time()
            optimizer.zero_grad()
            rng = nvtx.start_range(message="train")
            pred_pos, pred_neg = model(mfgs)
            loss = creterion(pred_pos, torch.ones_like(pred_pos))
            loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            # total_loss += float(loss) * train_param['batch_size']
            total_loss += float(loss) * len(rows)
            loss.backward()
            optimizer.step()
            nvtx.end_range(rng)
            time_model += time.time() - t_model_s

            #restart the timer for post processing after model update
            t_prep_s = time.time()
            model_latency.append(time.time() - t_tot_s)


            ########################################
            # observing the batch---size, index and losses
            ########################################
            if args.batch_level_log:
                print("\tbatch {}, cur batch count {}".format(cur_batch, len(rows)), "loss", float(loss))


            ########################################
            # batch level processing observation
            ########################################

            t_batch_post_s = time.time()
            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True, cuda=ALL_GPU)[0][0]
                    prep_time_breakdown["to_dgl_blocks"] += time.time() - t_batch_post_s
                else:
                    block = None
                estimated_prep_times["post_to_dgl_blocks"] += time.time() - t_prep_s

                t_prep_mailbox_s = time.time()
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                mailbox.update_memory_and_check_stablizing(model.memory_updater.last_updated_nid,
                                                           model.memory_updater.last_updated_memory,
                                                           root_nodes,
                                                           model.memory_updater.last_updated_ts,
                                                           threshold=SIM_THRESHOLD, any=SIM_ANY)
                stable_flag = mailbox.get_full_node_stable_flag()
                if prev_stable_flag is not None:
                    flips = (stable_flag != prev_stable_flag).sum().item()
                    flip_ratio_log.append(flips / stable_flag.shape[0])
                prev_stable_flag = stable_flag.clone()
                prep_time_breakdown["mailbox_update"] += time.time() - t_prep_mailbox_s
                
                estimated_prep_times["mailbox_update"] += time.time() - t_prep_mailbox_s

                t_forming_batch_s = time.time()
                # print("in memory", model.memory_updater.last_updated_nid.shape, "root_nodes", root_nodes.shape, "stable_flag", stable_flag)
                # input("Press Enter to continue...")
                color_sampler.update_node_indptr(ptr_end, unique_pos_root_nodes)
                batching_time_breakdown["updating_indptr"] += time.time() - t_forming_batch_s
                t_flag_update_s = time.time()
                color_sampler.update_node_stable_flag(stable_flag)
                # mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts)
                color_time_breakdown["forming_batch"] += time.time() - t_forming_batch_s
                batching_time_breakdown["updating_stable_flag"] += time.time() - t_flag_update_s

                estimated_prep_times["updating_indptr_and_stable_flag"] += time.time() - t_forming_batch_s
            else:
                t_forming_batch_s = time.time()
                color_sampler.update_node_indptr(ptr_end, unique_pos_root_nodes)
                color_time_breakdown["forming_batch"] += time.time() - t_forming_batch_s
                batching_time_breakdown["updating_indptr"] += time.time() - t_forming_batch_s
                estimated_prep_times["updating_indptr_and_stable_flag"] += time.time() - t_forming_batch_s
            
            time_prep += time.time() - t_prep_s
            batch_time = time.time() - t_tot_s
            time_tot += batch_time
            color_time_breakdown["model training"] += batch_time

            batch_latency.append(batch_time)
            other_latency.append(batch_time - model_latency[-1])

            t_record_memory_s = time.time()
            # record recent memory
            # if mailbox is not None:
            #     if args.observing:
            #         mailbox.validate_memory_history(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, threshold=SIM_THRESHOLD, any=SIM_ANY)
            #         # cur_node_acc = mailbox.get_stablizing_memory_check_accuracy()
            #         # print("****************cur_node_acc****************", cur_node_acc)
            #     mailbox.record_recent_memory_history(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts)

            # # update ptrs
            ptr_start = ptr_end
            t_training_end = time.time()
            color_time_breakdown["record memory"] += t_training_end - t_record_memory_s
            # ptr_end = min(ptr_start + train_param['batch_size'], train_edge_end)
            # # check stablized input
            # while ptr_end < train_edge_end:
            #     src_idx = df.iloc[ptr_end].src.astype(np.int32)
            #     dst_idx = df.iloc[ptr_end].dst.astype(np.int32)
            #     nid = torch.tensor([src_idx, dst_idx], dtype=torch.int64)
            #     if mailbox.is_node_memory_stable(nid,threshold=SIM_THRESHOLD,any=SIM_ANY):
            #         ptr_end += 1
            #     else:
            #         # print("batch size", ptr_end - ptr_start)
            #         break
            
        # node_stable, node_check = mailbox.get_node_stable_ratio()
        # event_stable, event_check = mailbox.get_event_stable_ratio()
        if args.observing:
            node_stable_acc = mailbox.get_stablizing_memory_check_accuracy()
        if args.profile_stable:
            # get node stable ratio
            node_stable, node_check = mailbox.get_node_stable_ratio()
            event_stable, event_check = mailbox.get_event_stable_ratio()
            print("node_stable_ratio", node_stable / node_check, "event_stable_ratio", event_stable / event_check, "total_check", event_check, "break_point", event_check - event_stable)
            print("node_stable", node_stable, "node_check", node_check, "event_stable", event_stable, "event_check", event_check)
        print("***********************************************")
        print("final_group_idx", final_group_idx, "batch count", batch_count)
        # print("final_group_idx", final_group_idx, 
        # "batch count", batch_count, 
        # "node_stable_ratio", node_stable/node_check, 
        # "event_stable_ratio", event_stable/event_check,
        # "total_check", event_check, "break_point", event_check-event_stable)
        print("\tform_batch time: {:.2f}s, coloring time: {:.2f}s, model training time: {:.2f}s, recording mem time: {:.2f}s, other time: {:.2f}s".format(color_time_breakdown["forming_batch"], color_time_breakdown["coloring"], color_time_breakdown["model training"], color_time_breakdown["record memory"], color_time_breakdown["others"]))
        print("\tsampling time: {:.2f}s, updating indptr time: {:.2f}s, updating stable flag time: {:.2f}s".format(batching_time_breakdown["sampling"], batching_time_breakdown["updating_indptr"], batching_time_breakdown["updating_stable_flag"]))
        #estimated_prep_times
        print("\testimated initial to_dgl_blocks time: {:.2f}s, prepare_input time: {:.2f}s, mailbox prep time: {:.2f}s, post to_dgl_blocks time: {:.2f}s, mailbox update time: {:.2f}s, updating indptr and stable flag time: {:.2f}s".format(estimated_prep_times["initial_to_dgl_blocks"], estimated_prep_times["prepare_input"], 
            estimated_prep_times["mailbox_prep"], estimated_prep_times["post_to_dgl_blocks"], estimated_prep_times["mailbox_update"], estimated_prep_times["updating_indptr_and_stable_flag"]))
        print('\t Captured prep time ratio: {:.2f}%'.format(100 * sum(estimated_prep_times.values()) / time_prep if time_prep > 0 else 0))
        if prefetch is not None:
            print("\tprefetch hit rate: {:.1%}  (hits={}, misses={})".format(prefetch.hit_rate(), prefetch.n_hits, prefetch.n_misses))
            prefetch.clear()
        if args.observing:
            print("****************node_stable_acc****************", node_stable_acc)
        print("***********************************************")

        if args.observing and SAVE_COUNT:
            batch_max_color_counts = np.array(batch_max_color_counts)
            batch_max_color_counts_total[e] = batch_max_color_counts


        # batch_sizes.append(batch_count)
    elif args.mode == 'observing':
        print("Entering observing mode...")
        moving_node_usage_stats = {"max": [], "min": [], "mean": []}
        st_row = 0
        ed_row = 0
        for i, rows in df[:train_edge_end].groupby(group_idx):
            ed_row = st_row + len(rows)
            # print("="*20, "batch", i, "start row", st_row, "end row", ed_row, "="*20)
            unique_pos_root_nodes = list(np.unique(np.concatenate([rows.src.values, rows.dst.values])).astype(np.int32))
            # node_usage_table = color_sampler.sampler.node_usage_table
            prev_node_usage_ptr = color_sampler.sampler.current_node_color_ptrs
            prev_node_usage_ptr = np.array(prev_node_usage_ptr)
            color_sampler.update_node_indptr(ed_row, unique_pos_root_nodes)
            cur_node_usage_ptr = color_sampler.sampler.current_node_color_ptrs
            cur_node_usage_ptr = np.array(cur_node_usage_ptr)
            moving_node_usage_ptr = cur_node_usage_ptr - prev_node_usage_ptr
            moving_max = np.max(moving_node_usage_ptr)
            moving_min = np.min(moving_node_usage_ptr)
            moving_mean = np.mean(moving_node_usage_ptr)
            print("\t batch {} node color ptr moving distance max {} min {} mean {}".format(i, moving_max, moving_min, moving_mean))
            moving_node_usage_stats["max"].append(moving_max)
            moving_node_usage_stats["min"].append(moving_min)
            moving_node_usage_stats["mean"].append(moving_mean)
            st_row = ed_row
            # plot the ptr moving diatance
            # image_dir = "results/tuning/"
            # image_fn = image_dir + "node_color_ptr_moving_distance_" + args.data + "_temp.png"
            # plt.clf()
            # plt.hist(moving_node_usage_ptr, bins=100)
            # plt.xlabel("node color ptr moving distance")
            # plt.ylabel("count")
            # plt.savefig(image_fn)

            
            # input("Press Enter to continue...")
    elif args.mode == 'observing_large':

        print("Entering large observing mode...")
        moving_node_usage_stats = {"max": [], "min": [], "mean": []}


        # chunk_num = args.chunk_num
        chunk_size = train_param['batch_size'] * 500    # 500 batches per chunk
        # chunk_size = (train_edge_end + chunk_num - 1) // chunk_num
        chunk_num = (train_edge_end + chunk_size - 1) // chunk_size
        batch_count = 0
        final_group_idx = np.zeros(train_edge_end)
        training_df = df[:train_edge_end]
        cur_batch = 0
        cur_chunk_idx = -1

        print("chunk size", chunk_size, "chunk num", chunk_num)


        st_row = 0
        ed_row = 0
        total_coloring_time = 0

        for i, rows in df[:train_edge_end].groupby(group_idx):
            chunk_idx = i // 500
            if chunk_idx != cur_chunk_idx:
                # we need to color the chunk
                cur_chunk_idx = chunk_idx
                color_st_row = i * train_param['batch_size']
                color_ed_row = min(color_st_row + chunk_size, train_edge_end)
                print("="*20, "chunk", cur_chunk_idx, "start row", color_st_row, "end row", color_ed_row, "="*20)

                if args.sampler_for_each_chunk:
                    if e == 0:
                        print("coloring chunk...")
                        color_st = time.time()
                        color_sampler.color_graph(st_event_id=color_st_row, end_event_id=color_ed_row)
                        color_ed = time.time()
                        print("coloring time", color_ed - color_st)
                        total_coloring_time += color_ed - color_st
                        color_sampler.cache_multi_color_table(chunk_id=i)
                        color_sampler.reset_node_indptr()
                    else:
                        print("reuse colored graph...")
                        color_sampler.reset_node_indptr()
                        assert color_sampler.is_cached(i), "chunk {} not cached".format(i)
                        color_sampler.load_multi_color_table(chunk_id=i)
                        
                else:
                    print("reset dependency table...")
                    color_sampler.reset_color_graph()
                    color_sampler.reset_node_indptr()
                    print("coloring chunk...")
                    color_st = time.time()
                    color_sampler.color_graph(st_event_id=color_st_row, end_event_id=color_ed_row)
                    color_ed = time.time()
                    print("coloring time", color_ed - color_st)
                    if e == 0:
                        # for the rest we can reuse the colored graph
                        total_coloring_time += color_ed - color_st

            ed_row = st_row + len(rows)
            # print("="*20, "batch", i, "start row", st_row, "end row", ed_row, "="*20)
            unique_pos_root_nodes = list(np.unique(np.concatenate([rows.src.values, rows.dst.values])).astype(np.int32))
            # node_usage_table = color_sampler.sampler.node_usage_table
            prev_node_usage_ptr = color_sampler.sampler.current_node_color_ptrs
            prev_node_usage_ptr = np.array(prev_node_usage_ptr)
            color_sampler.update_node_indptr(ed_row, unique_pos_root_nodes)
            cur_node_usage_ptr = color_sampler.sampler.current_node_color_ptrs
            cur_node_usage_ptr = np.array(cur_node_usage_ptr)
            moving_node_usage_ptr = cur_node_usage_ptr - prev_node_usage_ptr
            moving_max = np.max(moving_node_usage_ptr)
            moving_min = np.min(moving_node_usage_ptr)
            moving_mean = np.mean(moving_node_usage_ptr)
            print("\t batch {} node color ptr moving distance max {} min {} mean {}".format(i, moving_max, moving_min, moving_mean))
            moving_node_usage_stats["max"].append(moving_max)
            moving_node_usage_stats["min"].append(moving_min)
            moving_node_usage_stats["mean"].append(moving_mean)
            st_row = ed_row 
    elif args.mode == 'batch_stable_freezing_large':
        """
        This mode is used to run the large scale experiment
        1. break the training data into small chunks
        2. run the table coloring for each chunk in sequence
        3. form batch within each chunk
        """
        if mailbox is not None:
            mailbox.set_stablize_recorder(window_size=SIM_WINDOW)
        # chunk_size = 10000000 # 10 million
        # chunk_num = train_edge_end // chunk_size
        # chunk_num = args.chunk_num
        chunk_size = train_param['batch_size'] * 100
        chunk_num = (train_edge_end + chunk_size - 1) // chunk_size
        batch_count = 0
        final_group_idx = np.zeros(train_edge_end)
        training_df = df[:train_edge_end]
        cur_batch = 0

        print("chunk size", chunk_size, "chunk num", chunk_num)

        for i in range(chunk_num):
            total_train_time_chunk = 0
            total_coloring_time_chunk = 0
            st_row = i * chunk_size
            ed_row = min((i+1) * chunk_size, train_edge_end)
            print("="*20, "chunk", i, "start row", st_row, "end row", ed_row, "="*20)

            if args.sampler_for_each_chunk:
                if e == 0:
                    print("coloring chunk...")
                    color_st = time.time()
                    color_sampler.color_graph(st_event_id=st_row, end_event_id=ed_row)
                    color_ed = time.time()
                    print("coloring time", color_ed - color_st)
                    total_coloring_time += color_ed - color_st
                    total_coloring_time_chunk = color_ed - color_st
                    color_sampler.cache_multi_color_table(chunk_id=i)
                    color_sampler.reset_node_indptr()
                else:
                    print("reuse colored graph...")
                    color_sampler.reset_node_indptr()
                    assert color_sampler.is_cached(i), "chunk {} not cached".format(i)
                    color_sampler.load_multi_color_table(chunk_id=i)

                    
            else:
                print("reset dependency table...")
                color_sampler.reset_color_graph()
                color_sampler.reset_node_indptr()
                print("coloring chunk...")
                color_st = time.time()
                color_sampler.color_graph(st_event_id=st_row, end_event_id=ed_row)
                color_ed = time.time()
                print("coloring time", color_ed - color_st)
                if e == 0:
                    # for the rest we can reuse the colored graph
                    total_coloring_time += color_ed - color_st
                    total_coloring_time_chunk = color_ed - color_st

            ptr_start = st_row
            ptr_end = ptr_start
            

            while ptr_start < ed_row:
                ########################################
                # moving ptrs and sample batch
                ########################################
                t_forming_batch_s = time.time()
                # color_sampler.update_node_indptr(ptr_start, model.memory_updater.last_updated_nid)
                ptr_end = color_sampler.sample_batch(training_df,
                                                    start_event_id=ptr_start,
                                                    batch_index=cur_batch,
                                                    minimal_batch_size=train_param['batch_size'],
                                                    step_size=MAX_BACTH_SIZE)
                ptr_end = min(ptr_end, ed_row) # make sure the ptr_end is within the chunk
                # print("batch {}, cur batch count {}".format(cur_batch, ptr_end - ptr_start), "ptr_start", ptr_start, "ptr_end", ptr_end, "start row", st_row, "end row", ed_row)
                color_time_breakdown["forming_batch"] += time.time() - t_forming_batch_s
                batching_time_breakdown["sampling"] += time.time() - t_forming_batch_s
                
                #########################################
                # EXPERIMENTAL: adaptive_updater (record up to date stable flag)---get node stable flag indicating whether the node is stable
                ########################################
                if args.adaptive_update and adaptive_updater is not None:
                    node_stable_flag = mailbox.get_full_node_stable_flag() if mailbox is not None else None
                    if node_stable_flag is not None and args.batch_level_log:
                        print("node_stable_flag shape", node_stable_flag.shape, "stable count", torch.sum(node_stable_flag).item(), "total nodes", node_stable_flag.shape[0])
                    adaptive_updater.set_stable_record(node_stable_flag)
                #########################################


                cur_batch += 1
                # print("batch {}, cur batch count {}".format(cur_batch, ptr_end - ptr_start))
                t_training_start = time.time()
                
                # update batch count
                rows = df.iloc[ptr_start:ptr_end]
                final_group_idx[ptr_start:ptr_end] = batch_count
                batch_count += 1
                total_batch_count += 1
                total_batch_sum += len(rows)
                batch_sizes.append(len(rows))
                # print("batch {}, cur batch count {}".format(cur_batch, len(rows)))
                if args.lr_scale:
                    lr = min(train_param['lr'] * len(rows) / train_param['batch_size'] * LR_SCALE_FACTOR, train_param['lr'] * LR_SCALE_CEILING)
                    # print("batch size", len(rows), "preset batch size", train_param['batch_size'], "lr", lr)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                color_time_breakdown["others"] += time.time() - t_training_start

                # model forward
                t_tot_s = time.time()
                #########################################
                # EXPERIMENTAL: adaptive_updater(reduce stable root nodes)---based on node stable flag, we can reduce the number of root nodes
                ########################################
                if args.adaptive_update and adaptive_updater is not None:
                    try:
                        rows = adaptive_updater.elastic_row_filer(rows)
                    except Exception as e:
                        print("adaptive_updater error:", e)
                        rows = rows
                #########################################
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
                ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
                pos_root_end = root_nodes.shape[0] * 2 // 3
                unique_pos_root_nodes = np.unique(root_nodes[:pos_root_end])
                if sampler is not None:
                    if 'no_neg' in sample_param and sample_param['no_neg']:
                        pos_root_end = root_nodes.shape[0] * 2 // 3
                        sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                    else:
                        sampler.sample(root_nodes, ts)
                    ret = sampler.get_ret()
                    # time_sample += ret[0].sample_time()
                    time_sample += time.time() - t_tot_s
                t_prep_s = time.time()
                # if e == 15:
                #     to_dgl_blocks_ob(ret, sample_param['history'])
                if gnn_param['arch'] != 'identity':
                    mfgs = to_dgl_blocks(ret, sample_param['history'], cuda=ALL_GPU)
                else:
                    mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=ALL_GPU)
                prep_time_breakdown["to_dgl_blocks"] += time.time() - t_prep_s

                #Valiating the time for to_dgl_blocks conversion
                estimated_prep_times["initial_to_dgl_blocks"] += time.time() - t_prep_s

                t_prepare_s = time.time()
                mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
                prep_time_breakdown["prepare_input"] += time.time() - t_prepare_s

                #Valiating the time for prepare_input conversion
                estimated_prep_times["prepare_input"] += time.time() - t_prepare_s

                if mailbox is not None:
                    t_mailbox_prep_s = time.time()
                    mailbox.prep_input_mails(mfgs[0])
                    prep_time_breakdown["mailbox_prep"] += time.time() - t_mailbox_prep_s
                time_prep += time.time() - t_prep_s
                
                #valuating the time for mailbox preparation
                estimated_prep_times["mailbox_prep"] += time.time() - t_mailbox_prep_s
                ########################################
                # check input mails
                ########################################
                t_model_s = time.time()
                optimizer.zero_grad()
                rng = nvtx.start_range(message="train")
                pred_pos, pred_neg = model(mfgs)
                loss = creterion(pred_pos, torch.ones_like(pred_pos))
                loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                # total_loss += float(loss) * train_param['batch_size']
                total_loss += float(loss) * len(rows)
                loss.backward()
                optimizer.step()
                nvtx.end_range(rng)
                time_model += time.time() - t_model_s
                t_prep_s = time.time()
                model_latency.append(time.time() - t_tot_s)

                ########################################
                # observing the batch---size, index and losses
                ########################################
                if args.batch_level_log:
                    print("\tbatch {}, cur batch count {}".format(cur_batch, len(rows)), "loss", float(loss))


                ########################################
                # batch level processing observation
                ########################################

                if mailbox is not None:
                    eid = rows['Unnamed: 0'].values
                    mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                    block = None
                    if memory_param['deliver_to'] == 'neighbors':
                        block = to_dgl_blocks(ret, sample_param['history'], reverse=True, cuda=ALL_GPU)[0][0]
                    prep_time_breakdown["to_dgl_blocks"] += time.time() - t_prep_s

                    estimated_prep_times["post_to_dgl_blocks"] += time.time() - t_prep_s

                    t_prep_mailbox_s = time.time()
                    mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                    mailbox.update_memory_and_check_stablizing(model.memory_updater.last_updated_nid,
                                                            model.memory_updater.last_updated_memory,
                                                            root_nodes,
                                                            model.memory_updater.last_updated_ts,
                                                            threshold=SIM_THRESHOLD, any=SIM_ANY)
                    stable_flag = mailbox.get_full_node_stable_flag()
                    if prev_stable_flag is not None:
                        flips = (stable_flag != prev_stable_flag).sum().item()
                        flip_ratio_log.append(flips / stable_flag.shape[0])
                    prev_stable_flag = stable_flag.clone()
                    prep_time_breakdown["mailbox_update"] += time.time() - t_prep_mailbox_s

                    estimated_prep_times["mailbox_update"] += time.time() - t_prep_mailbox_s

                    t_batch_post_s = time.time()

                    t_forming_batch_s = time.time()
                    # print("in memory", model.memory_updater.last_updated_nid.shape, "root_nodes", root_nodes.shape, "stable_flag", stable_flag)
                    # input("Press Enter to continue...")
                    color_sampler.update_node_indptr(ptr_end, unique_pos_root_nodes)
                    batching_time_breakdown["updating_indptr"] += time.time() - t_forming_batch_s
                    t_flag_update_s = time.time()
                    color_sampler.update_node_stable_flag(stable_flag)
                    # mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts)
                    color_time_breakdown["forming_batch"] += time.time() - t_forming_batch_s
                    batching_time_breakdown["updating_stable_flag"] += time.time() - t_flag_update_s

                    estimated_prep_times["updating_indptr_and_stable_flag"] += time.time() - t_forming_batch_s
                else:
                    t_forming_batch_s = time.time()
                    color_sampler.update_node_indptr(ptr_end, unique_pos_root_nodes)
                    color_time_breakdown["forming_batch"] += time.time() - t_forming_batch_s
                    batching_time_breakdown["updating_indptr"] += time.time() - t_forming_batch_s
                    estimated_prep_times["updating_indptr_and_stable_flag"] += time.time() - t_forming_batch_s
                    
                prep_time_breakdown["batch_postprocessing"] += time.time() - t_batch_post_s
                time_prep += time.time() - t_prep_s
                batch_time = time.time() - t_tot_s
                time_tot += batch_time
                total_train_time_chunk += batch_time
                color_time_breakdown["model training"] += batch_time

                batch_latency.append(batch_time)
                other_latency.append(batch_time - model_latency[-1])

                t_record_memory_s = time.time()

                # # update ptrs
                ptr_start = ptr_end
                t_training_end = time.time()
                color_time_breakdown["record memory"] += t_training_end - t_record_memory_s
            # show chunk level stats
            
            if args.profile_stable:
                # get node stable ratio
                node_stable, node_check = mailbox.get_node_stable_ratio()
                event_stable, event_check = mailbox.get_event_stable_ratio()
                print("node_stable_ratio", node_stable / node_check, "event_stable_ratio", event_stable / event_check, "total_check", event_check, "break_point", event_check - event_stable)
                print("node_stable", node_stable, "node_check", node_check, "event_stable", event_stable, "event_check", event_check)

            print('\tChunk total training time:{:.2f}s, chunk coloring time {}, total coloring time {}'.format(total_train_time_chunk, total_coloring_time_chunk, total_coloring_time))
            print("\tform_batch time: {:.2f}s, coloring time: {:.2f}s, model training time: {:.2f}s, recording mem time: {:.2f}s, other time: {:.2f}s".format(color_time_breakdown["forming_batch"], color_time_breakdown["coloring"], color_time_breakdown["model training"], color_time_breakdown["record memory"], color_time_breakdown["others"]))
            print("\tsampling time: {:.2f}s, updating indptr time: {:.2f}s, updating stable flag time: {:.2f}s".format(batching_time_breakdown["sampling"], batching_time_breakdown["updating_indptr"], batching_time_breakdown["updating_stable_flag"]))
            print("\testimated initial to_dgl_blocks time: {:.2f}s, prepare_input time: {:.2f}s, mailbox prep time: {:.2f}s, post to_dgl_blocks time: {:.2f}s, mailbox update time: {:.2f}s, updating indptr and stable flag time: {:.2f}s".format(estimated_prep_times["initial_to_dgl_blocks"], estimated_prep_times["prepare_input"], 
            estimated_prep_times["mailbox_prep"], estimated_prep_times["post_to_dgl_blocks"], estimated_prep_times["mailbox_update"], estimated_prep_times["updating_indptr_and_stable_flag"]))
            print('\t Captured prep time ratio: {:.2f}%'.format(100 * sum(estimated_prep_times.values()) / time_prep if time_prep > 0 else 0))
    else:
        # for i, rows in df[:train_edge_end].groupby(group_indexes[random.randint(0, len(group_indexes) - 1)]):
        for i, rows in df[:train_edge_end].groupby(group_idx):
            ########################################
            # set lr based on batch size
            ########################################
            if args.lr_scale:
                lr = min(train_param['lr'] * len(rows) / train_param['batch_size'] * LR_SCALE_FACTOR, train_param['lr'] * LR_SCALE_CEILING)
                # print("batch size", len(rows), "preset batch size", train_param['batch_size'], "lr", lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            total_batch_count += 1
            total_batch_sum += len(rows)
            batch_sizes.append(len(rows))

            t_tot_s = time.time()
            root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
            ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = root_nodes.shape[0] * 2 // 3
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
                # time_sample += ret[0].sample_time()
                time_sample += time.time() - t_tot_s
            t_prep_s = time.time()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'], cuda=ALL_GPU)
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=ALL_GPU)
            prep_time_breakdown["to_dgl_blocks"] += time.time() - t_prep_s
            t_prepare_s = time.time()
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            prep_time_breakdown["prepare_input"] += time.time() - t_prepare_s
            if mailbox is not None:
                t_mailbox_prep_s = time.time()
                mailbox.prep_input_mails(mfgs[0])
                prep_time_breakdown["mailbox_prep"] += time.time() - t_mailbox_prep_s
            time_prep += time.time() - t_prep_s
            ########################################
            # check input mails
            ########################################
            # print("=====================================","batch",i,"=====================================")
            # print("root_nodes", root_nodes.shape, "ts", ts.shape)
            # real_root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
            # real_src_nodes = rows.src.values.astype(np.int32)
            # real_dst_nodes = rows.dst.values.astype(np.int32)
            # print("real_root_nodes", real_root_nodes.shape, min(real_root_nodes), max(real_root_nodes))
            # print("real_src_nodes", real_src_nodes.shape, min(real_src_nodes), max(real_src_nodes))
            # print("real_dst_nodes", real_dst_nodes.shape, min(real_dst_nodes), max(real_dst_nodes))
            # mfg = mfgs[0]
            # for i, b in enumerate(mfg):
            #     print("block", i)
            #     src_nodes, dst_nodes = b.edges()
            #     print("src", src_nodes, src_nodes.shape, min(src_nodes), max(src_nodes))
            #     print("dst", dst_nodes, dst_nodes.shape, min(dst_nodes), max(dst_nodes))
            #     src_nodes_id = b.srcdata['ID']
            #     print("src_nodes_id", src_nodes_id, src_nodes_id.shape, min(src_nodes_id), max(src_nodes_id))
            # input("Press Enter to continue...")
            

            t_model_s = time.time()
            optimizer.zero_grad()
            rng = nvtx.start_range(message="train")
            pred_pos, pred_neg = model(mfgs)
            loss = creterion(pred_pos, torch.ones_like(pred_pos))
            loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            # total_loss += float(loss) * train_param['batch_size']
            total_loss += float(loss) * len(rows)
            loss.backward()
            optimizer.step()
            nvtx.end_range(rng)
            t_prep_s = time.time()
            time_model += time.time() - t_model_s
            model_latency.append(time.time() - t_tot_s)

            ########################################
            # batch level processing observation
            ########################################
            if args.batch_level_log:
                print("\tbatch {}, cur batch count {}".format(i, len(rows)), "loss", float(loss))
            
            # print("=====================================","batch",i,"=====================================")
            # # print(rows)
            # true_root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
            # unique_nodes, node_counts = np.unique(true_root_nodes, return_counts=True)
            # print("batch",i, "root_nodes", true_root_nodes.shape, "unique_nodes", unique_nodes.shape,
            # "node count max:", np.max(node_counts),"node count mean:", np.mean(node_counts),"node count std:", np.std(node_counts), "loss", loss/len(rows))

            # unique_average.append(np.mean(node_counts))
            # unique_max.append(np.max(node_counts))
            # unique_std.append(np.std(node_counts))
            # loss_list.append((loss/len(rows)).cpu().detach().numpy())

            # sys.exit(0)
            if mailbox is not None:
                t_batch_post_s = time.time()
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True, cuda=ALL_GPU)[0][0]
                    prep_time_breakdown["to_dgl_blocks"] += time.time() - t_batch_post_s
                    t_prep_mailbox_s = time.time()
                else:
                    block = None
                    t_prep_mailbox_s = time.time()
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts)
                prep_time_breakdown["mailbox_update"] += time.time() - t_prep_mailbox_s
                t_batch_post_s = time.time()
            time_prep += time.time() - t_prep_s
            if mailbox is not None:
                prep_time_breakdown["batch_postprocessing"] += time.time() - t_batch_post_s
            batch_time = time.time() - t_tot_s
            batch_time = time.time() - t_tot_s
            time_tot += batch_time

            batch_latency.append(batch_time)
            other_latency.append(batch_time - model_latency[-1])

    if args.mode == 'observing' or args.mode == 'observing_large':
        print("moving_node_usage_stats", moving_node_usage_stats)
        print("average max", np.mean(moving_node_usage_stats["max"][:-1]), "average min", np.mean(moving_node_usage_stats["min"][:-1]), "average mean", np.mean(moving_node_usage_stats["mean"][:-1]))
        print("max max", np.max(moving_node_usage_stats["max"][:-1]), "min max", np.min(moving_node_usage_stats["max"][:-1]), "mean max", np.mean(moving_node_usage_stats["max"][:-1]))
        print("exiting observing mode...")
        break

    ap, auc = eval('val')
    if e > 2 and ap > best_ap:
        best_e = e
        best_ap = ap
        best_auc = auc
        torch.save(model.state_dict(), path_saver)

    print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f} val loss:{:4f} ave_val loss:{:4f} final_val loss:{:4f}'.format(total_loss, ap, auc, sum(val_losses), sum(val_losses) / len(val_losses), val_losses[-1]))
    print('\ttotal time:{:.2f}s sample time:{:.2f}s prep time:{:.2f}s model time:{:.2f}s'.format(time_tot, time_sample, time_prep, time_model))
    print('\tprep time details: to_dgl_blocks:{:.2f}s prepare_input:{:.2f}s mailbox_prep:{:.2f}s mailbox_update:{:.2f}s batch_postprocessing:{:.2f}s'.format(
        prep_time_breakdown["to_dgl_blocks"],
        prep_time_breakdown["prepare_input"],
        prep_time_breakdown["mailbox_prep"],
        prep_time_breakdown["mailbox_update"],
        prep_time_breakdown["batch_postprocessing"],
    ))
    print('\t Initial Captured prep time ratio: {:.2f}%'.format(100 * sum(prep_time_breakdown.values()) / time_prep if time_prep > 0 else 0))
    print('\t Estimated Captured prep time ratio: {:.2f}%'.format(100 * sum(estimated_prep_times.values()) / time_prep if time_prep > 0 else 0))

    if args.profile_to_dgl_blocks:
        print_to_dgl_blocks_profile()
        reset_to_dgl_blocks_profile()
    if flip_ratio_log:
        flip_arr = np.array(flip_ratio_log)
        print('\tstable flag flip ratio — mean:{:.4f}  std:{:.4f}  min:{:.4f}  max:{:.4f}  batches:{:d}'.format(flip_arr.mean(), flip_arr.std(), flip_arr.min(), flip_arr.max(), len(flip_arr)))
        print('\tstable flag flip list: ' + ' '.join('{:.6f}'.format(r) for r in flip_ratio_log))
    # batch_latency.append(time_tot)
    


    if e != 0:
        total_train_time += time_tot
    ########################################
    # epoch level processing observation
    ########################################
    if args.check_similarity and e % PROFILE_INTERVAL == 0:
        # fn = 'profile/similarities/inter_batch_{}_batch_{}_epoch_{}.pkl'.format(args.data, train_param['batch_size'], e)
        fn = 'profile/similarities/{}_inter_batch_{}_batch_{}_epoch_{}.pkl'.format(args.model_name, args.data, train_param['batch_size'], e)
        # fn = None
        analysis_profile(plot_dir=fn)
        disable_profiling()
        reset_profile()


    if args.check_similarity_epoch:
        
        if e == 0:
            reset_n_record_profile()
        else:
            fn = 'profile/similarities/inter_epoch_{}_batch_{}_epoch_{}.pkl'.format(args.data, train_param['batch_size'], e)
            # fn = None
            compare_profile(plot_dir=fn)
            reset_n_record_profile()

        disable_profiling()

    if args.observe_batch_latency and e == 15:
        file_name = "results/batch_latency_" + args.data + "_" + args.mode + "_epoch_" + str(e) + ".png"
        plt.clf()
        plt.plot(batch_latency, label='batch_latency', color='r')
        # plt.xlabel('Batch Index')
        # plt.ylabel('Latency (s)')
        # plt.savefig(file_name)
        # print("exiting due to batch latency observation mode...")

        # file_name = "results/batch_model_latency_" + args.data + "_" + args.mode + "_epoch_" + str(e) + ".png"
        # plt.clf()
        plt.plot(model_latency, label='model_latency', color='b')
        plt.plot(other_latency, label='other_latency', color='g')
        plt.legend()
        plt.xlabel('Batch Index')
        plt.ylabel('Latency (s)')
        plt.savefig(file_name)

        plt.clf()
        file_name = "results/batch_size_" + args.data + "_" + args.mode + "_epoch_" + str(e) + ".png"
        plt.clf()
        plt.plot(batch_sizes)
        plt.xlabel('Batch Index')
        plt.ylabel('Size')
        plt.savefig(file_name)
        print("exiting due to batch latency observation mode...")

        sys.exit(0)


########################################
# plot unique node stats
########################################
'''
unique_average = unique_average[:-1]
unique_max = unique_max[:-1]
unique_std = unique_std[:-1]
loss_list = loss_list[:-1]


batch_list = np.arange(len(unique_average))
unique_average = np.array(unique_average)
unique_max = np.array(unique_max)
unique_std = np.array(unique_std)
loss_list = np.array(loss_list)

file_name = "results/batch_stats_with_mean_" + args.data + ".png"
plt.clf()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# scatter plot the time trace, x-axis is time, y-axis is node index, color is blue, save to results/temporal_clustering.png
ax1.plot(batch_list, unique_average, 'g-')
ax2.plot(batch_list, loss_list, 'b-')
ax1.set_xlabel('Batch Index')
ax1.set_ylabel('Node Count', color='g')
ax2.set_ylabel('Loss value', color='b')
plt.savefig(file_name)

file_name = "results/batch_stats_with_std_" + args.data + ".png"
plt.clf()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# scatter plot the time trace, x-axis is time, y-axis is node index, color is blue, save to results/temporal_clustering.png
ax1.plot(batch_list, unique_std, 'g-')
ax2.plot(batch_list, loss_list, 'b-')
ax1.set_xlabel('Batch Index')
ax1.set_ylabel('Node Count', color='g')
ax2.set_ylabel('Loss value', color='b')
plt.savefig(file_name)

file_name = "results/batch_stats_with_max_" + args.data + ".png"
plt.clf()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# scatter plot the time trace, x-axis is time, y-axis is node index, color is blue, save to results/temporal_clustering.png
ax1.plot(batch_list, unique_max, 'g-')
ax2.plot(batch_list, loss_list, 'b-')
ax1.set_xlabel('Batch Index')
ax1.set_ylabel('Node Count', color='g')
ax2.set_ylabel('Loss value', color='b')
plt.savefig(file_name)
'''

# sys.exit(0)

if args.observing and SAVE_COUNT:
    file_name = "data_{}_mode_{}_cheat_{}_simTR_{}_simANY_{}_ignoreStable.pkl".format(args.data, args.mode, CHEAT_CODE, SIM_THRESHOLD, SIM_ANY, IGNORE_STABLE)
    file_path = count_path + file_name
    print("saving batch_max_color_counts to", file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(batch_max_color_counts_total, f)


print('=====================================end of training=====================================')
print("data", args.data, 
        "mode", args.mode, 
        "model", args.config,
        "cheat", CHEAT_CODE, "simTR", SIM_THRESHOLD, "simANY", SIM_ANY, "ignoreStable", IGNORE_STABLE)
# print('Loading model at epoch {}...'.format(best_e))
# model.load_state_dict(torch.load(path_saver))
# model.eval()
# if sampler is not None:
#     sampler.reset()
# if mailbox is not None:
#     mailbox.reset()
#     model.memory_updater.last_updated_nid = None
#     eval('train')
#     eval('val')
# ap, auc = eval('test')
# if args.eval_neg_samples > 1:
#     print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))
# else:
#     print('\ttest AP:{:4f}  test AUC:{:4f}'.format(ap, auc))
if args.mode == 'batch_stable_freezing' or args.mode == 'batch_stable_freezing_large':
    print('\tTotal training time:{:.2f}s, average batch size:{:.2f}, color count:{}, minimum node count:{}, coloring time {}'.format(total_train_time, total_batch_sum / total_batch_count, NUM_COLORS, color_sampler.color_bottom, total_coloring_time))
else:
    print('\tTotal training time:{:.2f}s, average batch size:{:.2f}'.format(total_train_time, total_batch_sum / total_batch_count))
print('\tBest epoch:{:d}  Best AP:{:4f}  Best AUC:{:4f}'.format(best_e, best_ap, best_auc), 
      "ave val loss {:.4f}, min val loss {:.4f}".format(sum(val_losses) / len(val_losses), min(val_losses)))