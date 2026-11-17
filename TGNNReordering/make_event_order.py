"""Build a non-temporal processing order for a dataset's TRAINING events.

The order is consumed by train_simple.py (--event_order <file>): Cascade then
trains on the training events in this order instead of time order. Validation
and test are never reordered.

    python TGNNReordering/make_event_order.py --data WIKI --ordering gcod --window global
    python TGNNReordering/make_event_order.py --data WIKI --ordering gcod --window 10000
    python TGNNReordering/make_event_order.py --data WIKI --ordering shuffle --window 10000

Orderings (all applied inside consecutive windows of `--window` time-ordered
training events; `global` = one window holding the whole training set):

  gcod          GCoD node blocks of the final training snapshot (gcod_reorder.py).
                Each event gets the block of its row in the adjacency (its src,
                like a static GNN processing the reordered adjacency row block by
                row block); events are sorted by block, keeping time order inside
                a block. With --offdiag end, events whose endpoints fall in
                different blocks (sparse region) go after all diagonal blocks.
  random_block  Same procedure and block sizes, but nodes are assigned to blocks
                at random: separates "GCoD structure" from "any block grouping".
  shuffle       Events uniformly shuffled inside each window: breaks time order
                with no structure at all.

Output: <data_root>/<data>/event_orders/<tag>.npz with
  perm    int64 [num_train_events]: perm[new_position] = original row index
  meta    json string (ordering, window, block info, disorder statistics)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from snapshot import build_snapshot
from gcod_reorder import (gcod_partition, degree_split_from_quantiles, labels_from_order,
                          baseline_order)

ORDERINGS = ('gcod', 'random_block', 'shuffle')


def order_tag(ordering, window, offdiag, window_frac=None):
    if window_frac is not None and window > 0:
        w = f'f{window_frac:g}'
    else:
        w = 'global' if window <= 0 else str(window)
    tag = f'{ordering}_w{w}'
    if ordering != 'shuffle':
        tag += f'_{offdiag}'
    return tag


def event_block_keys(src, dst, node_block, offdiag):
    """Sort key per event: block of its src row; `end` sends off-diagonal events last."""
    num_blocks = int(node_block.max()) + 1
    key = node_block[src].copy()
    key[key < 0] = num_blocks          # src never in a block: only self-loop events
    if offdiag == 'end':
        key[node_block[src] != node_block[dst]] = num_blocks
    return key


def windowed_order(n, window, key=None, rng=None):
    """perm over range(n): sort by key (stable) or shuffle, within windows."""
    if window <= 0 or window >= n:
        bounds = [(0, n)]
    else:
        bounds = [(s, min(n, s + window)) for s in range(0, n, window)]
    perm = np.empty(n, dtype=np.int64)
    for s, e in bounds:
        if key is None:
            perm[s:e] = s + rng.permutation(e - s)
        else:
            perm[s:e] = s + np.argsort(key[s:e], kind='stable')
    return perm


def disorder_stats(perm, ts):
    """How far the order is from time order (ts are the original, time-sorted stamps)."""
    t = ts[perm]
    n = len(perm)
    disp = np.abs(perm - np.arange(n))
    running_max = np.maximum.accumulate(t)
    return {
        'num_events': int(n),
        'mean_abs_displacement': float(disp.mean()),
        'max_abs_displacement': int(disp.max()),
        # adjacent pairs where time goes backwards
        'adjacent_inversion_rate': float((t[1:] < t[:-1]).mean()) if n > 1 else 0.0,
        # events processed after a strictly later event was already processed
        'behind_running_max_rate': float((t[1:] < running_max[:-1]).mean()) if n > 1 else 0.0,
        'mean_time_lag_behind_max': float((running_max - t).mean()),
    }


def build_event_order(src, dst, ts, num_nodes, ordering, window, offdiag='src', degree_split=None,
                      degree_quantiles=(0.5, 0.8), total_subgraphs=12, num_groups=2, seed=0,
                      partition_fn=None, log=print, window_frac=None):
    """Returns (perm, meta) for time-sorted training events src/dst/ts."""
    meta = {'ordering': ordering, 'window': window, 'window_frac': window_frac, 'offdiag': offdiag,
            'seed': seed, 'tag': order_tag(ordering, window, offdiag, window_frac)}
    num_train = len(src)
    rng = np.random.default_rng(seed)
    if ordering == 'shuffle':
        perm = windowed_order(num_train, window, rng=rng)
    elif ordering in ('gcod', 'random_block'):
        snap = build_snapshot(src, dst, num_nodes)
        if degree_split:
            degree_split = sorted(degree_split)
        else:
            degree_split = degree_split_from_quantiles(snap.degree, list(degree_quantiles))
        kwargs = {} if partition_fn is None else {'partition_fn': partition_fn}
        part = gcod_partition(snap, degree_split, total_subgraphs, num_groups, log=log, **kwargs)
        if ordering == 'gcod':
            node_order = part['order']
        else:
            node_order = baseline_order('random', snap, part['num_isolated'], seed=seed)
        node_block, _ = labels_from_order(node_order, part['block_sizes'], part['block_group'], num_nodes)
        key = event_block_keys(src, dst, node_block, offdiag)
        perm = windowed_order(num_train, window, key=key)
        meta.update({'degree_split': degree_split, 'num_blocks': int(len(part['block_sizes'])),
                     'block_sizes': part['block_sizes'].tolist(),
                     'diag_event_frac': float((node_block[src] == node_block[dst]).mean())})
        meta['_event_key'] = key  # for tests; dropped before saving
    else:
        raise ValueError(f'unknown ordering {ordering}')
    assert np.array_equal(np.sort(perm), np.arange(num_train))
    meta['disorder'] = disorder_stats(perm, ts)
    return perm, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--data_root', default='DATA')
    p.add_argument('--ordering', choices=ORDERINGS, required=True)
    p.add_argument('--window', default='global',
                   help="events per reordering window, or 'global' for the whole training set")
    p.add_argument('--window_frac', type=float, default=None,
                   help='window as a fraction of the training events (overrides --window; >= 1 means global)')
    p.add_argument('--offdiag', choices=('src', 'end'), default='src')
    p.add_argument('--degree_split', type=int, nargs='*', default=None)
    p.add_argument('--degree_quantiles', type=float, nargs='+', default=[0.5, 0.8])
    p.add_argument('--total_subgraphs', type=int, default=12)
    p.add_argument('--num_groups', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', default=None, help='default: <data_root>/<data>/event_orders')
    p.add_argument('--force', action='store_true', help='rebuild even if the output file exists')
    args = p.parse_args()

    window = -1 if str(args.window).lower() == 'global' else int(args.window)
    data_root = args.data_root if os.path.isabs(args.data_root) else os.path.join(REPO_ROOT, args.data_root)
    out_dir = args.out_dir or os.path.join(data_root, args.data, 'event_orders')
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    df = pd.read_csv(os.path.join(data_root, args.data, 'edges.csv'), usecols=['src', 'dst', 'time', 'ext_roll'])
    roll = df['ext_roll'].to_numpy()
    num_train = int((roll == 0).sum())
    # train_simple.py takes rows [0, train_edge_end) as the training set
    assert np.all(roll[:num_train] == 0), 'training events are not a prefix of edges.csv'
    src = df['src'].to_numpy()[:num_train].astype(np.int64)
    dst = df['dst'].to_numpy()[:num_train].astype(np.int64)
    ts = df['time'].to_numpy()[:num_train]
    assert np.all(np.diff(ts) >= 0), 'edges.csv is not sorted by time'
    num_nodes = int(max(df['src'].max(), df['dst'].max())) + 1
    del df
    print(f'[{args.data}] {num_train} train events, {num_nodes} nodes ({time.time() - t0:.1f}s)')
    if args.window_frac is not None:
        window = -1 if args.window_frac >= 1 else max(1, int(np.ceil(args.window_frac * num_train)))
    path = os.path.join(out_dir, order_tag(args.ordering, window, args.offdiag,
                                           args.window_frac if window > 0 else None) + '.npz')
    if os.path.exists(path) and not args.force:
        print(f'[{args.data}] exists, reusing (pass --force to rebuild): {path}')
        print(f'[{args.data}] wrote {path}')
        return

    perm, meta = build_event_order(src, dst, ts, num_nodes, args.ordering, window, args.offdiag,
                                   degree_split=args.degree_split, degree_quantiles=args.degree_quantiles,
                                   total_subgraphs=args.total_subgraphs, num_groups=args.num_groups,
                                   seed=args.seed,
                                   window_frac=args.window_frac if window > 0 else None)
    assert path.endswith(meta['tag'] + '.npz')
    meta.pop('_event_key', None)
    meta['data'] = args.data
    np.savez(path, perm=perm, meta=json.dumps(meta))
    print(f'[{args.data}] wrote {path}')
    print(json.dumps(meta['disorder'], indent=2))


if __name__ == '__main__':
    main()
