"""Static snapshot of a TGL dataset's training partition.

Every training event (ext_roll == 0) is folded into one undirected, deduplicated,
self-loop-free graph -- the "final snapshot" a static reordering would see. For
each static edge we keep the index of the first training event that created it,
which is what the error-edge metric compares against batch boundaries.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrainSnapshot:
    num_nodes: int        # over the full dataset (train/val/test), as gen_graph.py does
    src: np.ndarray       # train events in file (= time) order, int64
    dst: np.ndarray
    u: np.ndarray         # static undirected edges, u < v, int64
    v: np.ndarray
    first_event: np.ndarray  # index into src/dst of the first event on (u, v)
    degree: np.ndarray    # static undirected degree per node

    @property
    def num_events(self):
        return len(self.src)

    @property
    def num_static_edges(self):
        return len(self.u)


def load_train_events(data_root, data):
    path = os.path.join(data_root, data, 'edges.csv')
    df = pd.read_csv(path, usecols=['src', 'dst', 'ext_roll'],
                     dtype={'src': np.int64, 'dst': np.int64, 'ext_roll': np.int8})
    num_nodes = int(max(df['src'].max(), df['dst'].max())) + 1
    train = (df['ext_roll'] == 0).to_numpy()
    src = df['src'].to_numpy()[train]
    dst = df['dst'].to_numpy()[train]
    return src, dst, num_nodes


def build_snapshot(src, dst, num_nodes):
    keep = src != dst  # GCoD removes self loops before partitioning
    idx = np.nonzero(keep)[0]
    lo = np.minimum(src[keep], dst[keep])
    hi = np.maximum(src[keep], dst[keep])
    key = lo * num_nodes + hi
    # np.unique with return_index uses a stable sort, so `first` is the earliest event
    uniq, first = np.unique(key, return_index=True)
    u = uniq // num_nodes
    v = uniq % num_nodes
    degree = np.bincount(u, minlength=num_nodes) + np.bincount(v, minlength=num_nodes)
    return TrainSnapshot(num_nodes=num_nodes, src=src, dst=dst, u=u, v=v,
                         first_event=idx[first], degree=degree.astype(np.int64))


def load_snapshot(data_root, data):
    src, dst, num_nodes = load_train_events(data_root, data)
    return build_snapshot(src, dst, num_nodes)
