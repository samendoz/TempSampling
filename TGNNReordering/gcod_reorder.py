"""Replica of GCoD's graph reordering (ReorderingExperiment/GCoD-main/get_partition.py).

my_partition_graph() in GCoD:
  1. bucket nodes by degree using `degree_split` thresholds
  2. give each bucket a number of subgraphs proportional to its share of edges
     (largest-remainder rounding), where total_subgraphs is per group
  3. METIS-partition each bucket's induced subgraph into n_subgraph[i] * num_groups
     parts, dealing part j to group j % num_groups
  4. relabel nodes so groups are contiguous, and within a group blocks follow
     bucket order then part order

The resulting blocks (subgraph parts) are the dense diagonal chunks of the
reordered adjacency; groups are the coarser diagonal level.

Deviations from the original, all forced by the change of setting:
  - dgl.distributed.partition_graph(reshuffle=True) was removed from DGL; we call
    dgl.metis_partition_assignment directly (pymetis as fallback).
  - Nodes with zero training degree are excluded from the blocks and appended at
    the end (block -1). They have no static edges, so they cannot affect the metric,
    and keeping them would hand METIS millions of isolated vertices.
  - A non-empty bucket that rounds to 0 subgraphs would silently lose its nodes in
    GCoD; here it is bumped to 1 subgraph (with a warning).
  - degree_split can be derived from degree quantiles, since GCoD's hardcoded
    per-dataset thresholds only exist for Planetoid graphs.
"""
import numpy as np


def degree_split_from_quantiles(degree, quantiles):
    active = degree[degree > 0]
    thresholds = np.quantile(active, quantiles)
    split = sorted(set(int(np.ceil(t)) for t in thresholds))
    return [s for s in split if s > 1]


def metis_assignment(num_nodes, u, v, k, balance_edges=True):
    if k <= 1 or num_nodes <= 1:
        return np.zeros(num_nodes, dtype=np.int64)
    if len(u) == 0:
        # no internal edges: METIS has nothing to cut, split evenly
        return np.arange(num_nodes, dtype=np.int64) * k // num_nodes
    try:
        import dgl
        import torch
        g = dgl.graph((torch.from_numpy(np.concatenate([u, v])),
                       torch.from_numpy(np.concatenate([v, u]))), num_nodes=num_nodes)
        return dgl.metis_partition_assignment(g, k, balance_edges=balance_edges).numpy().astype(np.int64)
    except ImportError:
        pass
    import pymetis
    a = np.concatenate([u, v])
    b = np.concatenate([v, u])
    order = np.argsort(a, kind='stable')
    xadj = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(a, minlength=num_nodes), out=xadj[1:])
    vweights = np.diff(xadj).clip(min=1) if balance_edges else None
    _, membership = pymetis.part_graph(k, xadj=xadj, adjncy=b[order], vweights=vweights)
    return np.asarray(membership, dtype=np.int64)


def gcod_partition(snapshot, degree_split, total_subgraphs, num_groups, balance_edges=True,
                   partition_fn=metis_assignment, log=print):
    assert total_subgraphs % num_groups == 0, 'total_subgraphs should be divided by num_groups'
    per_group = total_subgraphs // num_groups
    degree = snapshot.degree
    u, v = snapshot.u, snapshot.v

    active = degree > 0
    bucket = np.searchsorted(np.asarray(degree_split), degree, side='right')
    n_filter = len(degree_split) + 1

    avg_edge = degree.sum() / per_group
    n_in_edges = [degree[active & (bucket == i)].sum() for i in range(n_filter)]
    n_subgraph = [int(n / avg_edge) for n in n_in_edges]
    reminder = [n_in_edges[i] - avg_edge * n_subgraph[i] for i in range(n_filter)]
    idx = sorted(range(n_filter), key=lambda i: reminder[i], reverse=True)
    for i in range(per_group - sum(n_subgraph)):
        n_subgraph[idx[i % n_filter]] += 1
    for i in range(n_filter):
        if n_subgraph[i] == 0 and np.any(active & (bucket == i)):
            log(f'[gcod] bucket {i} rounded to 0 subgraphs, bumping to 1')
            n_subgraph[i] = 1

    group_blocks = [[] for _ in range(num_groups)]
    local = np.full(snapshot.num_nodes, -1, dtype=np.int64)
    for i in range(n_filter):
        nodes = np.nonzero(active & (bucket == i))[0]
        if len(nodes) == 0:
            continue
        k = min(n_subgraph[i] * num_groups, len(nodes))
        if k == 1:
            group_blocks[0].append((i, nodes))
            continue
        local[nodes] = np.arange(len(nodes))
        inner = (local[u] >= 0) & (local[v] >= 0)
        parts = partition_fn(len(nodes), local[u[inner]], local[v[inner]], k, balance_edges)
        local[nodes] = -1
        log(f'[gcod] bucket {i}: {len(nodes)} nodes, {int(inner.sum())} inner edges, {k} parts')
        for j in range(k):
            group_blocks[j % num_groups].append((i, nodes[parts == j]))

    blocks = [(g, b, members) for g in range(num_groups) for b, members in group_blocks[g]]
    isolated = np.nonzero(~active)[0]
    order = np.concatenate([m for _, _, m in blocks] + [isolated])
    return {
        'order': order,  # new position -> original node id
        'block_sizes': np.array([len(m) for _, _, m in blocks], dtype=np.int64),
        'block_group': np.array([g for g, _, _ in blocks], dtype=np.int64),
        'block_bucket': np.array([b for _, b, _ in blocks], dtype=np.int64),
        'degree_split': list(degree_split),
        'n_subgraph': n_subgraph,
        'num_isolated': len(isolated),
    }


def labels_from_order(order, block_sizes, block_group, num_nodes):
    """Per-node block / group labels when `order` is cut into consecutive blocks."""
    node_block = np.full(num_nodes, -1, dtype=np.int64)
    node_group = np.full(num_nodes, -1, dtype=np.int64)
    n = int(block_sizes.sum())
    block_of_pos = np.repeat(np.arange(len(block_sizes)), block_sizes)
    node_block[order[:n]] = block_of_pos
    node_group[order[:n]] = block_group[block_of_pos]
    return node_block, node_group


def baseline_order(kind, snapshot, num_isolated, seed=0):
    """Same node set as the GCoD order, without GCoD's placement: identity or random."""
    active = np.nonzero(snapshot.degree > 0)[0]
    isolated = np.nonzero(snapshot.degree == 0)[0]
    assert len(isolated) == num_isolated
    if kind == 'identity':
        pass
    elif kind == 'random':
        active = np.random.default_rng(seed).permutation(active)
    else:
        raise ValueError(f'unknown baseline ordering: {kind}')
    return np.concatenate([active, isolated])
