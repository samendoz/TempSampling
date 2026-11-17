"""Support for the temporal-violation experiment in train_simple.py.

Run A trains Cascade on training events in time order (the original). Run B
trains the same loop on a non-temporal order of the training events built by
TGNNReordering/make_event_order.py (e.g. GCoD blocks). Validation/test always
stay in time order.

Pieces:
  load_event_order()   permutation of the training rows + its metadata
  sequence_df()        the training dataframe in processing order
  color_graph_arrays() T-CSR for Cascade's color sampler with event ids replaced
                       by processing positions (the color sampler treats a larger
                       event id as "later"), each node's list sorted by position.
                       The neighbor sampler keeps the original time-sorted graph.
  ViolationTracker     per-batch counts of what the model actually reads out of
                       time order (see record()).
  ExperimentLog        per-epoch CSV + per-batch npz.
"""
import csv
import json
import os

import numpy as np
import pandas as pd


def load_event_order(path, train_edge_end):
    z = np.load(path, allow_pickle=False)
    perm = z['perm'].astype(np.int64)
    meta = json.loads(str(z['meta'])) if 'meta' in z.files else {}
    if len(perm) != train_edge_end:
        raise ValueError(f'event order {path} has {len(perm)} events, training set has {train_edge_end}')
    if not np.array_equal(np.sort(perm), np.arange(train_edge_end)):
        raise ValueError(f'event order {path} is not a permutation of the training rows')
    return perm, meta


def sequence_df(df, perm, train_edge_end):
    """Training rows in processing order, then val/test untouched; index = position.

    'Unnamed: 0' (original event id) travels with each row, so edge features
    stay aligned.
    """
    return pd.concat([df.iloc[perm], df.iloc[train_edge_end:]]).reset_index(drop=True)


def position_of_row(perm, num_rows):
    """pos[original_row] = processing position (identity outside the training set)."""
    pos = np.arange(num_rows, dtype=np.int64)
    pos[perm] = np.arange(len(perm), dtype=np.int64)
    return pos


def color_graph_arrays(g, pos):
    indptr = np.asarray(g['indptr'])
    new_eid = pos[np.asarray(g['eid']).astype(np.int64)]
    owner = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    order = np.lexsort((new_eid, owner))
    indices = np.asarray(g['indices'])[order]
    return indptr, indices, new_eid[order].astype(np.asarray(g['eid']).dtype)


class ViolationTracker:
    """Per-batch temporal-violation counts for one epoch.

    For the positive events of a batch, processed at positions [ptr_start, ptr_end):
      time_inversions   events older than an event already processed this epoch
      mem_future        endpoint memories (src and dst) already updated by an event
                        with a LATER timestamp: future information reaching the model
      nbr_known / nbr_inbatch / nbr_unprocessed
                        sampled neighbor events (all roots, incl. negatives; always
                        in the past by timestamp) that were processed in an earlier
                        batch / are in this batch (Cascade's native staleness) /
                        have not been processed yet this epoch (their memory and
                        mail updates are missing)
      nbr_per_root      sampled neighbors per root (sanity check for the sampler)

    mem_future is accumulated on the mailbox's device and read back once per
    epoch. train_simple.py excludes the tracker's own time from batch timings.
    """

    def __init__(self, pos_of_row, mailbox=None):
        self.pos = pos_of_row
        self.mailbox = mailbox
        self.epoch_rows = []

    def start_epoch(self):
        self.max_ts_done = -np.inf
        self.rows = []
        self.mem_future = []

    def record(self, ptr_start, ptr_end, rows, ret):
        ts = rows['time'].to_numpy()
        n = len(ts)
        rec = {'ptr_start': ptr_start, 'ptr_end': ptr_end, 'events': n,
               'time_inversions': int((ts < self.max_ts_done).sum()) if n else 0}
        if self.mailbox is not None and n:
            import torch
            mts = self.mailbox.node_memory_ts
            nodes = torch.from_numpy(np.concatenate([rows['src'].to_numpy(), rows['dst'].to_numpy()]).astype(np.int64))
            ev_ts = torch.from_numpy(np.concatenate([ts, ts]).astype(np.float32))
            self.mem_future.append((mts[nodes.to(mts.device)] > ev_ts.to(mts.device)).sum())
        if ret is not None:
            eid = np.asarray(ret[0].eid()).astype(np.int64)
            p = self.pos[eid]
            rec['nbr_known'] = int((p < ptr_start).sum())
            rec['nbr_inbatch'] = int(((p >= ptr_start) & (p < ptr_end)).sum())
            rec['nbr_unprocessed'] = int((p >= ptr_end).sum())
            rec['roots'] = int(ret[0].dim_out())
        if n:
            self.max_ts_done = max(self.max_ts_done, float(ts.max()))
        self.rows.append(rec)

    def end_epoch(self, epoch):
        if self.mem_future:
            import torch
            mf = torch.stack([x.reshape(()) for x in self.mem_future]).cpu().numpy()
        else:
            mf = np.zeros(len(self.rows), dtype=np.int64)
        for r, m in zip(self.rows, mf):
            r['mem_future'] = int(m)
            r['epoch'] = epoch
        self.epoch_rows.extend(self.rows)
        return self.summary(self.rows)

    @staticmethod
    def summary(rows):
        def tot(k):
            return float(sum(r.get(k, 0) for r in rows))
        ev = max(tot('events'), 1.0)
        nb = max(tot('nbr_known') + tot('nbr_inbatch') + tot('nbr_unprocessed'), 1.0)
        return {
            'num_batches': len(rows),
            'avg_batch_size': tot('events') / max(len(rows), 1),
            'time_inversion_rate': tot('time_inversions') / ev,
            'mem_future_rate': tot('mem_future') / (2 * ev),
            'nbr_known_rate': tot('nbr_known') / nb,
            'nbr_inbatch_rate': tot('nbr_inbatch') / nb,
            'nbr_unprocessed_rate': tot('nbr_unprocessed') / nb,
            'nbr_per_root': nb / max(tot('roots'), 1.0) if tot('roots') else 0.0,
        }

    def save(self, path):
        if not self.epoch_rows:
            return
        keys = sorted({k for r in self.epoch_rows for k in r})
        np.savez_compressed(path, **{k: np.array([r.get(k, -1) for r in self.epoch_rows]) for k in keys})


EPOCH_FIELDS = ['epoch', 'train_loss', 'val_ap', 'val_auc', 'val_loss',
                'time_tot', 'time_sample', 'time_prep', 'time_model', 'eval_time',
                'num_batches', 'avg_batch_size', 'time_inversion_rate', 'mem_future_rate',
                'nbr_known_rate', 'nbr_inbatch_rate', 'nbr_unprocessed_rate', 'nbr_per_root',
                'best_epoch', 'test_ap', 'test_auc', 'total_train_time']


class ExperimentLog:
    """One CSV row per epoch, plus a final row with epoch='test'.

    Every row = static fields (run identity) + EPOCH_FIELDS; missing values are blank.
    """

    def __init__(self, path, static_fields):
        self.path = path
        self.static = dict(static_fields)
        self.fields = list(self.static) + EPOCH_FIELDS
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def write(self, row):
        row = {**self.static, **row}
        unknown = [k for k in row if k not in self.fields]
        if unknown:
            raise KeyError(f'columns {unknown} not in CSV header')
        with open(self.path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self.fields, restval='').writerow(row)
