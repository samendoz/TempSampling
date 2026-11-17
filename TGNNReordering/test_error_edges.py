"""Brute-force check of the error-edge counts on a small synthetic dataset.

    python TGNNReordering/test_error_edges.py                  # real METIS (dgl / pymetis)
    python TGNNReordering/test_error_edges.py --stub_partition # random parts, no METIS needed

Runs the full pipeline (run_analysis.run) on a generated edges.csv with many
repeated pairs and self loops, then recomputes every per-batch count
(known / inbatch / future / global_future, both units, every ordering and
dense/sparse scope) with plain Python loops and asserts they match exactly.
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_analysis as ra
from gcod_reorder import labels_from_order


def make_dataset(root, rng, num_nodes=80, num_events=3000, num_train=2400):
    os.makedirs(os.path.join(root, 'SYN'))
    src = rng.integers(0, 60, num_events)    # few active nodes -> many repeated pairs
    dst = rng.integers(0, 60, num_events)
    src[::40] = dst[::40]                     # self-loop events
    src[num_train:] = rng.integers(0, num_nodes, num_events - num_train)  # val/test-only nodes
    roll = np.full(num_events, 2)
    roll[:num_train] = 0
    roll[num_train:(num_train + num_events) // 2] = 1
    pd.DataFrame({'src': src, 'dst': dst, 'time': np.arange(num_events), 'ext_roll': roll}) \
        .to_csv(os.path.join(root, 'SYN', 'edges.csv'))
    return src[:num_train], dst[:num_train]


def reference(trs, trd, bs, unit, inscope, first):
    nb = (len(trs) + bs - 1) // bs
    ref = {k: np.zeros(nb, dtype=np.int64) for k in ('known', 'inbatch', 'future', 'global_future')}
    for b in range(nb):
        nodes = set(trs[b * bs:(b + 1) * bs]) | set(trd[b * bs:(b + 1) * bs])
        for j, (x, y) in enumerate(zip(trs, trd)):
            if (unit == 'edges' and j not in first) or not inscope(x, y):
                continue
            fb = j // bs
            if fb > b:
                ref['global_future'][b] += 1
            kind = 'known' if fb < b else 'inbatch' if fb == b else 'future'
            for a in {x, y}:
                if a in nodes:
                    ref[kind][b] += 1
    return ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stub_partition', action='store_true', help='random partition instead of METIS')
    args = parser.parse_args()

    rng = np.random.default_rng(2)
    tmp = tempfile.mkdtemp()
    try:
        trs, trd = make_dataset(tmp, rng)
        partition_fn = (lambda n, u, v, k, balance_edges: rng.integers(0, k, n)) if args.stub_partition else None
        # tiny chunk_events forces many chunks, exercising the cross-chunk running counters
        conf = dict(ra.DEFAULTS, data='SYN', data_root=tmp, output_dir=os.path.join(tmp, 'out'),
                    batch_size=[7, 64], chunk_events=100)
        ra.run(conf, partition_fn=partition_fn)

        first, seen = set(), set()
        for j, (a, b) in enumerate(zip(trs, trd)):
            pair = (min(a, b), max(a, b))
            if a != b and pair not in seen:
                seen.add(pair)
                first.add(j)

        r = np.load(os.path.join(tmp, 'out', 'reorder.npz'))
        num_nodes = len(r['gcod_order'])
        assert sorted(r['gcod_order'].tolist()) == list(range(num_nodes)), 'GCoD order is not a permutation'
        for bs in conf['batch_size']:
            pb = np.load(os.path.join(tmp, 'out', f'per_batch_bs{bs}.npz'))
            nb = (len(trs) + bs - 1) // bs
            npb = [len(set(trs[b * bs:(b + 1) * bs]) | set(trd[b * bs:(b + 1) * bs])) for b in range(nb)]
            assert (pb['nodes_per_batch'] == npb).all(), 'nodes_per_batch mismatch'
            for ordering in ['gcod'] + conf['baselines']:
                blk, grp = labels_from_order(r[f'{ordering}_order'], r['block_sizes'], r['block_group'], num_nodes)
                scopes = {'all': lambda x, y: True,
                          f'{ordering}/block_diag': lambda x, y: blk[x] == blk[y],
                          f'{ordering}/block_offdiag': lambda x, y: blk[x] != blk[y],
                          f'{ordering}/group_diag': lambda x, y: grp[x] == grp[y],
                          f'{ordering}/group_offdiag': lambda x, y: grp[x] != grp[y]}
                for unit in ('events', 'edges'):
                    for scope, inscope in scopes.items():
                        ref = reference(trs, trd, bs, unit, inscope, first)
                        for kind, expected in ref.items():
                            got = pb[f'{unit}/{scope}/{kind}']
                            assert (got == expected).all(), f'mismatch: bs={bs} {unit} {scope} {kind}'
                    print(f'ok  batch_size={bs:<3d} ordering={ordering:<9s} unit={unit}')
        print('ALL CHECKS PASSED')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
