"""Server check for ParallelSampler.set_force_search (sampler_core.cpp). Needs the rebuilt extension.

    python setup.py build_ext --inplace
    python TGNNReordering/verify_neighbor_search.py --data WIKI

Samples the first --num_batches training batches (TGN settings: 1 layer, 10 most
recent neighbors) with the cursor path (default) and with force_search, and
compares both against a plain-Python ground truth of what the TGL 'recent'
sampler is meant to return for each (root, time):

    candidates k in (max(s, e - 10), e], ts[k] < t
    s = indptr[n], e = last position with ts <= t (or s)

Expected:
  force_search  matches the ground truth for 100% of roots, in time order AND in
                shuffled order.
  cursor path   time order: matches except roots whose node repeats in the batch
                at a different time (the known two-pass ts_ptr bug, "Part 0");
                shuffled order: many roots get wrong (mostly missing) neighbors,
                which is why --neighbor_search is required for --event_order runs.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sampler_core import ParallelSampler  # noqa: E402

NEIGHS = 10


def ground_truth(g, roots, ts):
    indptr, gts, geid = g['indptr'], g['ts'], g['eid']
    out = []
    for n, t in zip(roots, ts):
        s, end = int(indptr[n]), int(indptr[n + 1])
        e = s + int(np.searchsorted(gts[s:end], t, side='right')) - 1
        e = max(e, s)
        ks = [k for k in range(e, max(s, e - NEIGHS), -1) if k < end and gts[k] < t - 1e-7]
        out.append(sorted(int(geid[k]) for k in ks))
    return out


def sampled(sampler, roots, ts):
    sampler.sample(roots, ts)
    b = sampler.get_ret()[0]
    per_root = [[] for _ in range(len(roots))]
    for r, e in zip(b.row(), b.eid()):
        per_root[int(r)].append(int(e))
    return [sorted(x) for x in per_root]


def make(g, force):
    s = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                        32, 1, 1, [NEIGHS], True, False, 1, 0.0)
    if force:
        s.set_force_search(True)
    return s


def run(g, df, T, order, batch_size, num_batches, label):
    stats = {}
    for name, force in (('cursor', False), ('force_search', True)):
        smp = make(g, force)
        ok = tot = nb_got = nb_exp = 0
        repeat_bad = 0
        for b in range(num_batches):
            idx = order[b * batch_size:(b + 1) * batch_size]
            if len(idx) == 0:
                break
            rows = df.iloc[idx]
            roots = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
            ts = np.concatenate([rows.time.values, rows.time.values]).astype(np.float32)
            got = sampled(smp, roots, ts)
            exp = ground_truth(g, roots, ts)
            # roots whose node occurs in this batch at another timestamp (Part 0 trigger)
            key = pd.Series(ts).groupby(roots).transform('nunique').to_numpy() > 1
            for i, (a, x) in enumerate(zip(got, exp)):
                tot += 1
                nb_got += len(a)
                nb_exp += len(x)
                if a == x:
                    ok += 1
                elif key[i]:
                    repeat_bad += 1
        stats[name] = (ok, tot, nb_got, nb_exp, repeat_bad)
        print(f'  [{label}] {name:12s} exact {ok}/{tot} = {ok / max(tot, 1):.4%}  '
              f'neighbors/root got {nb_got / max(tot, 1):.2f} expected {nb_exp / max(tot, 1):.2f}  '
              f'(mismatches on repeated-node roots: {repeat_bad})')
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='WIKI')
    p.add_argument('--batch_size', type=int, default=900)
    p.add_argument('--num_batches', type=int, default=20)
    args = p.parse_args()
    g = np.load(f'DATA/{args.data}/ext_full.npz')
    g = {k: g[k] for k in g.files}
    df = pd.read_csv(f'DATA/{args.data}/edges.csv')
    T = int(df[df['ext_roll'].gt(0)].index[0])
    n = min(T, args.batch_size * args.num_batches)

    time_order = np.arange(n)
    shuffled = np.random.default_rng(0).permutation(n)
    s_time = run(g, df, T, time_order, args.batch_size, args.num_batches, 'time order')
    s_shuf = run(g, df, T, shuffled, args.batch_size, args.num_batches, 'shuffled')

    ok = True
    for label, s in (('time order', s_time), ('shuffled', s_shuf)):
        a, t = s['force_search'][:2]
        if a != t:
            ok = False
            print(f'FAIL: force_search differs from ground truth ({label})')
    a, t, _, _, rep = s_time['cursor']
    if t - a != rep:
        print(f'NOTE: cursor path mismatches in time order not explained by repeated nodes: {t - a - rep}')
    print('PASS' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
