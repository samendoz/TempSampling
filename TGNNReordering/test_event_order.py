"""Brute-force checks for the temporal-violation experiment (no GPU / torch / METIS needed).

    python TGNNReordering/test_event_order.py

Checks, on a small synthetic dataset:
  1. make_event_order: every ordering is a permutation, windows stay closed,
     block orderings are sorted by block and keep time order inside a block,
     'end' puts off-diagonal events after the diagonal ones.
  2. event_reorder.sequence_df / position_of_row: rows land where perm says, val/test untouched.
  3. event_reorder.color_graph_arrays: per-node lists hold the same events,
     relabeled to processing positions and sorted by them.
  4. ViolationTracker (without mailbox): time inversions and neighbor
     known/inbatch/unprocessed counts equal a plain-loop recount.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from make_event_order import build_event_order, ORDERINGS
from event_reorder import sequence_df, position_of_row, color_graph_arrays, ViolationTracker


def stub_partition(num_nodes, u, v, k, balance_edges=True):
    return np.arange(num_nodes) % k


def synthetic(seed=0, num_nodes=60, num_events=3000, val=400, test=400):
    rng = np.random.default_rng(seed)
    n = num_events + val + test
    # two communities + some cross edges + repeats, time-sorted with ties
    comm = rng.integers(0, 2, n)
    src = rng.integers(0, num_nodes // 2, n) + comm * (num_nodes // 2)
    dst = rng.integers(0, num_nodes // 2, n) + comm * (num_nodes // 2)
    cross = rng.random(n) < 0.15
    dst[cross] = rng.integers(0, num_nodes, cross.sum())
    ts = np.sort(rng.integers(0, n // 3, n)).astype(np.float64)
    roll = np.r_[np.zeros(num_events), np.ones(val), 2 * np.ones(test)].astype(int)
    df = pd.DataFrame({'Unnamed: 0': np.arange(n), 'src': src, 'dst': dst, 'time': ts, 'ext_roll': roll})
    return df, num_nodes, num_events


def tcsr(df, num_nodes):
    """Same layout as gen_graph.py ext_full: per src, time-sorted, eid = row index."""
    order = np.lexsort((df['time'].to_numpy(), df['src'].to_numpy()))
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(df['src'].to_numpy(), minlength=num_nodes), out=indptr[1:])
    return {'indptr': indptr, 'indices': df['dst'].to_numpy()[order],
            'eid': np.arange(len(df))[order], 'ts': df['time'].to_numpy()[order]}


class FakeBlock:
    def __init__(self, eid, roots):
        self._eid, self._roots = eid, roots

    def eid(self):
        return self._eid

    def dim_out(self):
        return self._roots


def check_orders(df, num_nodes, T):
    src = df['src'].to_numpy()[:T]
    dst = df['dst'].to_numpy()[:T]
    ts = df['time'].to_numpy()[:T]
    orders = {}
    for ordering in ORDERINGS:
        for window in (-1, 500, 1234):
            for offdiag in (('src', 'end') if ordering != 'shuffle' else ('src',)):
                perm, meta = build_event_order(src, dst, ts, num_nodes, ordering, window, offdiag,
                                               partition_fn=stub_partition, log=lambda *a: None)
                assert np.array_equal(np.sort(perm), np.arange(T))
                W = T if window <= 0 else window
                for s in range(0, T, W):
                    seg = perm[s:s + W]
                    assert seg.min() >= s and seg.max() < s + W, 'event left its window'
                    if ordering != 'shuffle':
                        key = meta['_event_key'][seg]
                        assert np.all(np.diff(key) >= 0), 'not sorted by block'
                        same = key[1:] == key[:-1]
                        assert np.all(np.diff(seg)[same] > 0), 'time order broken inside a block'
                if ordering != 'shuffle' and offdiag == 'end':
                    key = meta['_event_key']
                    nb = meta['num_blocks']
                    # off-diagonal events got the max key -> after all diagonal ones in their window
                    assert key.max() == nb
                if ordering == 'gcod' and window == -1:
                    assert meta['disorder']['adjacent_inversion_rate'] > 0
                orders[(ordering, window, offdiag)] = perm
    return orders


def check_sequence_and_color_graph(df, num_nodes, T, perm):
    seq = sequence_df(df, perm, T)
    assert len(seq) == len(df)
    assert np.array_equal(seq['Unnamed: 0'].to_numpy()[:T], perm)
    assert np.array_equal(seq['Unnamed: 0'].to_numpy()[T:], np.arange(T, len(df)))
    assert list(seq.index) == list(range(len(df)))
    pos = position_of_row(perm, len(df))
    assert np.array_equal(pos[perm], np.arange(T))
    assert np.array_equal(pos[T:], np.arange(T, len(df)))

    g = tcsr(df, num_nodes)
    indptr, indices, eid = color_graph_arrays(g, pos)
    assert np.array_equal(indptr, g['indptr'])
    for n in range(num_nodes):
        a, b = indptr[n], indptr[n + 1]
        assert np.all(np.diff(eid[a:b]) > 0), 'per-node list not sorted by position'
        got = sorted(zip(eid[a:b].tolist(), indices[a:b].tolist()))
        exp = sorted((int(pos[e]), int(d)) for e, d in zip(g['eid'][a:b], g['indices'][a:b]))
        assert got == exp, f'node {n}: color graph lost/changed events'
        # each event really belongs to src n: the row at that position has src n and that dst
        for p, d in got:
            r = seq.iloc[p]
            assert r['src'] == n and r['dst'] == d
    return seq, pos


def check_tracker(df, T, perm, seq, pos, seed=1):
    rng = np.random.default_rng(seed)
    tr = ViolationTracker(pos, mailbox=None)
    tr.start_epoch()
    bounds = []
    p = 0
    while p < T:
        q = min(T, p + int(rng.integers(50, 400)))
        bounds.append((p, q))
        p = q
    exp_rows = []
    done_max = -np.inf
    for p, q in bounds:
        rows = seq.iloc[p:q]
        # fake sampled neighbors: random past-by-time events of any training row
        nbr = rng.integers(0, T, size=3 * (q - p))
        tr.record(p, q, rows, [FakeBlock(nbr, 3 * (q - p))])
        e = {'inv': 0, 'known': 0, 'inb': 0, 'unp': 0}
        for t in rows['time']:
            e['inv'] += t < done_max
        for x in nbr:
            px = int(np.nonzero(perm == x)[0][0])
            e['known' if px < p else 'inb' if px < q else 'unp'] += 1
        done_max = max(done_max, rows['time'].max())
        exp_rows.append(e)
    tr.end_epoch(0)
    for r, e in zip(tr.rows, exp_rows):
        assert r['time_inversions'] == e['inv']
        assert (r['nbr_known'], r['nbr_inbatch'], r['nbr_unprocessed']) == (e['known'], e['inb'], e['unp'])
    s = ViolationTracker.summary(tr.rows)
    assert abs(s['nbr_known_rate'] + s['nbr_inbatch_rate'] + s['nbr_unprocessed_rate'] - 1) < 1e-9
    return s


def main():
    df, num_nodes, T = synthetic()
    orders = check_orders(df, num_nodes, T)
    print('1. event orders                OK')
    identity = np.arange(T)
    for name, perm in (('time', identity), ('gcod global', orders[('gcod', -1, 'src')]),
                       ('shuffle w500', orders[('shuffle', 500, 'src')])):
        seq, pos = check_sequence_and_color_graph(df, num_nodes, T, perm)
        s = check_tracker(df, T, perm, seq, pos)
        print(f'2-4. {name:13s} sequence/color graph/tracker OK  '
              f'time_inversion_rate={s["time_inversion_rate"]:.3f}')
        if name == 'time':
            assert s['time_inversion_rate'] == 0
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
