"""Summarize a temporal-violation sweep (one row per run).

    python TGNNReordering/summarize_violation.py results/violation/WIKI/TGN [more dirs...]

For each <run>.csv written by train_simple.py --exp_csv:
  test_ap / test_auc     final row (best-by-val model, memory replayed in time order)
  best_val_ap, best_epoch
  epoch_time             mean training time per epoch, epoch 0 excluded (warm-up)
  avg_batch_size, num_batches (mean over epochs)
  violation rates        mean over epochs
  d_test_ap / d_test_auc difference to the 'time' run of the same data/model
Writes <dir>/summary.csv and prints a markdown table.
"""
import glob
import os
import sys

import pandas as pd

RATES = ['time_inversion_rate', 'mem_future_rate', 'nbr_known_rate', 'nbr_inbatch_rate',
         'nbr_unprocessed_rate', 'nbr_per_root']


def summarize_run(path):
    df = pd.read_csv(path)
    ep = df[df['epoch'].astype(str) != 'test'].copy()
    ep['epoch'] = ep['epoch'].astype(int)
    test = df[df['epoch'].astype(str) == 'test']
    row = {'run': os.path.splitext(os.path.basename(path))[0],
           'data': df['data'].iloc[0], 'event_order': df['event_order'].iloc[0],
           'neighbor_search': df['neighbor_search'].iloc[0], 'epochs': len(ep)}
    if len(test):
        t = test.iloc[-1]
        row.update({'test_ap': t['test_ap'], 'test_auc': t['test_auc'],
                    'best_val_ap': t['val_ap'], 'best_epoch': t['best_epoch']})
    else:
        row.update({'test_ap': None, 'test_auc': None,
                    'best_val_ap': ep['val_ap'].max() if len(ep) else None, 'best_epoch': None})
    later = ep[ep['epoch'] > 0] if (ep['epoch'] > 0).any() else ep
    row['epoch_time'] = later['time_tot'].mean()
    row['avg_batch_size'] = ep['avg_batch_size'].mean()
    row['num_batches'] = ep['num_batches'].mean()
    row['last_train_loss'] = ep['train_loss'].iloc[-1] if len(ep) else None
    for k in RATES:
        row[k] = pd.to_numeric(ep[k], errors='coerce').mean() if k in ep else None
    return row


def main():
    dirs = sys.argv[1:] or ['results/violation']
    for d in dirs:
        files = sorted(f for f in glob.glob(os.path.join(d, '*.csv')) if not f.endswith('summary.csv'))
        if not files:
            print(f'no run CSVs in {d}')
            continue
        rows = []
        for f in files:
            try:
                rows.append(summarize_run(f))
            except Exception as e:  # a crashed run leaves a partial CSV
                print(f'skipping {f}: {e}')
        s = pd.DataFrame(rows)
        base = s[s['run'] == 'time']
        if len(base):
            for k in ('test_ap', 'test_auc', 'best_val_ap', 'epoch_time'):
                s['d_' + k] = s[k] - base.iloc[0][k]
        s.to_csv(os.path.join(d, 'summary.csv'), index=False)
        cols = [c for c in ['run', 'test_ap', 'd_test_ap', 'test_auc', 'd_test_auc', 'best_val_ap',
                            'epoch_time', 'd_epoch_time', 'avg_batch_size', 'num_batches',
                            'time_inversion_rate', 'mem_future_rate', 'nbr_unprocessed_rate',
                            'nbr_inbatch_rate', 'nbr_per_root'] if c in s]
        print(f'\n## {d}\n')
        try:
            print(s[cols].to_markdown(index=False, floatfmt='.4f'))
        except ImportError:  # tabulate not installed
            print(s[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
