"""GCoD-style reordering of a TGNN dataset's final training snapshot + per-batch
error-edge (future knowledge) analysis.

    python TGNNReordering/run_analysis.py --config TGNNReordering/configs/WIKI.yml
    python TGNNReordering/run_analysis.py --config TGNNReordering/configs/*.yml --batch_size 900

Relative `data_root` is resolved against the TempSampling repo root, relative
`output_dir` against this directory.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from snapshot import load_snapshot
from gcod_reorder import (gcod_partition, degree_split_from_quantiles, labels_from_order,
                          baseline_order)
from error_edges import batch_error_counts, summarize, UNITS

DEFAULTS = {
    'data_root': 'DATA',
    'batch_size': [900, 8000],
    'degree_split': None,          # explicit thresholds; overrides degree_quantiles
    'degree_quantiles': [0.5, 0.8],
    'total_subgraphs': 12,         # GCoD defaults (train.py / tune.py)
    'num_groups': 2,
    'balance_edges': True,
    'baselines': ['identity', 'random'],
    'seed': 0,
    'output_dir': None,            # default: results/<data>
    'save_per_batch': True,
    'chunk_events': 20_000_000,
}


def load_config(path, overrides):
    import yaml
    conf = dict(DEFAULTS)
    conf.update(yaml.safe_load(open(path, 'r')) or {})
    conf.update({k: v for k, v in overrides.items() if v is not None})
    if isinstance(conf['batch_size'], int):
        conf['batch_size'] = [conf['batch_size']]
    if conf['output_dir'] is None:
        conf['output_dir'] = os.path.join('results', conf['data'])
    if not os.path.isabs(conf['data_root']):
        conf['data_root'] = os.path.join(REPO_ROOT, conf['data_root'])
    if not os.path.isabs(conf['output_dir']):
        conf['output_dir'] = os.path.join(HERE, conf['output_dir'])
    return conf


def block_stats(snapshot, node_block, node_group):
    bu, bv = node_block[snapshot.u], node_block[snapshot.v]
    in_block = bu == bv
    nblocks = int(node_block.max()) + 1
    return {
        'block_diag_edge_frac': float(in_block.mean()),
        'group_diag_edge_frac': float((node_group[snapshot.u] == node_group[snapshot.v]).mean()),
        'edges_per_block': np.bincount(bu[in_block], minlength=nblocks).tolist(),
    }


def run(conf, partition_fn=None, log=print):
    os.makedirs(conf['output_dir'], exist_ok=True)
    t0 = time.time()
    snap = load_snapshot(conf['data_root'], conf['data'])
    log(f"[{conf['data']}] {snap.num_nodes} nodes, {snap.num_events} train events, "
        f"{snap.num_static_edges} static edges ({time.time() - t0:.1f}s)")

    if conf['degree_split'] is not None:
        degree_split = sorted(conf['degree_split'])
    else:
        degree_split = degree_split_from_quantiles(snap.degree, conf['degree_quantiles'])
    log(f"[{conf['data']}] degree_split = {degree_split}")

    t0 = time.time()
    kwargs = {} if partition_fn is None else {'partition_fn': partition_fn}
    part = gcod_partition(snap, degree_split, conf['total_subgraphs'], conf['num_groups'],
                          balance_edges=conf['balance_edges'], log=log, **kwargs)
    log(f"[{conf['data']}] GCoD partition: {len(part['block_sizes'])} blocks ({time.time() - t0:.1f}s)")

    orders = {'gcod': part['order']}
    for kind in conf['baselines']:
        orders[kind] = baseline_order(kind, snap, part['num_isolated'], seed=conf['seed'])

    reorder_info = {}
    scope_labels = {'all': None}
    for name, order in orders.items():
        node_block, node_group = labels_from_order(order, part['block_sizes'], part['block_group'], snap.num_nodes)
        reorder_info[name] = block_stats(snap, node_block, node_group)
        scope_labels[f'{name}/block_diag'] = node_block
        scope_labels[f'{name}/group_diag'] = node_group

    np.savez_compressed(os.path.join(conf['output_dir'], 'reorder.npz'),
                        **{f'{n}_order': o for n, o in orders.items()},
                        block_sizes=part['block_sizes'], block_group=part['block_group'],
                        block_bucket=part['block_bucket'])

    results = {}
    rows = []
    for bs in conf['batch_size']:
        t0 = time.time()
        nodes_per_batch, counts, totals = batch_error_counts(snap, bs, scope_labels, conf['chunk_events'])
        # off-diagonal (sparse region) = all - diagonal, per batch
        for unit in UNITS:
            for name in orders:
                for level in ('block', 'group'):
                    diag, off = (unit, f'{name}/{level}_diag'), (unit, f'{name}/{level}_offdiag')
                    counts[off] = {k: counts[(unit, 'all')][k] - counts[diag][k] for k in counts[diag]}
                    totals[off] = totals[(unit, 'all')] - totals[diag]
        results[bs] = {}
        for unit, scope in sorted(counts):
            s = summarize(nodes_per_batch, counts[(unit, scope)], totals[(unit, scope)])
            results[bs][f'{unit}:{scope}'] = s
            ordering, _, level = scope.partition('/')
            rows.append({'data': conf['data'], 'batch_size': bs, 'unit': unit,
                         'ordering': ordering if level else '-', 'scope': level or 'all', **s})
        if conf['save_per_batch']:
            np.savez_compressed(os.path.join(conf['output_dir'], f'per_batch_bs{bs}.npz'),
                                nodes_per_batch=nodes_per_batch,
                                **{f'{u}/{s}/{k}': c[k] for (u, s), c in counts.items() for k in c})
        log(f"[{conf['data']}] batch_size={bs}: {len(nodes_per_batch)} batches ({time.time() - t0:.1f}s)")
        for unit in UNITS:
            a = results[bs][f'{unit}:all']
            log(f"  [{unit}] all: total_future={a['total_future']}  per_batch={a['mean_future_per_batch']:.1f}"
                f"  per_node={a['mean_future_per_node']:.2f}  rate={a['future_rate']:.3f}"
                f"  global_rate={a['mean_global_future_rate']:.3f}")
            for name in orders:
                d = results[bs][f'{unit}:{name}/block_diag']
                o = results[bs][f'{unit}:{name}/block_offdiag']
                log(f"    {name:9s} dense diag: total_future={d['total_future']} rate={d['future_rate']:.3f}"
                    f" | sparse offdiag: total_future={o['total_future']} rate={o['future_rate']:.3f}")

    summary = {
        'config': {k: v for k, v in conf.items()},
        'dataset': {'num_nodes': snap.num_nodes, 'active_nodes': int((snap.degree > 0).sum()),
                    'train_events': snap.num_events, 'static_edges': snap.num_static_edges},
        'reorder': {'degree_split': degree_split, 'n_subgraph': part['n_subgraph'],
                    'block_sizes': part['block_sizes'].tolist(),
                    'block_group': part['block_group'].tolist(),
                    'block_bucket': part['block_bucket'].tolist(), **{'orderings': reorder_info}},
        'results': {str(bs): r for bs, r in results.items()},
    }
    with open(os.path.join(conf['output_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    write_csv(os.path.join(conf['output_dir'], 'summary.csv'), rows)
    return summary


def write_csv(path, rows):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='+', required=True)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--batch_size', type=int, nargs='+', default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    if args.output_dir is not None and len(args.config) > 1:
        parser.error('--output_dir only makes sense with a single --config')
    for path in args.config:
        conf = load_config(path, {'data_root': args.data_root, 'batch_size': args.batch_size,
                                  'output_dir': args.output_dir})
        run(conf)
