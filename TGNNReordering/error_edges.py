"""Per-batch "error edge" (future information) counts on a reordered snapshot.

Training events are cut into consecutive batches of `batch_size` (TGL baseline
batching, positive edges only). Every training event is a temporal edge: two
events between the same nodes A and B at different times are different edges,
since each carries its own timestamp/features.

Node-level view (what a batch would read): for a node n appearing in batch b,
each training event incident to n is classified by the batch it belongs to:
  known    batch <  b   (already happened when b is processed)
  inbatch  batch == b   (happens inside b itself)
  future   batch >  b   -> error edge: the final reordered graph exposes it to b
                           before it exists
Counts are per (batch, node) incidence: an event with both endpoints in the
batch counts once for each endpoint. Self-loop events count once.

Graph-level view (G_final vs G_b): global_future[b] = number of events of the
final graph that have not happened yet at the end of batch b, over the whole
graph, not only around the batch's nodes.

Units:
  events  every temporal event counts (the metric above)
  edges   only the first event of each distinct static pair (u, v) counts, i.e.
          leakage of neighbor *existence*, ignoring repeated interactions

Scopes restrict which events are counted, by the reordered position of their
endpoints: None = all events, otherwise a per-node label array and the event is
in scope iff both endpoints carry the same label (diagonal region).

Events are streamed in chunks of whole batches; per-node running counters carry
the "known" prefix across chunks, so memory stays bounded by the chunk size.
"""
import numpy as np

KINDS = ('known', 'inbatch', 'future', 'global_future')
UNITS = ('events', 'edges')


def _chunks(n_ev, batch_size, chunk_events):
    chunk = max(1, chunk_events // batch_size) * batch_size  # chunks never split a batch
    for e0 in range(0, n_ev, chunk):
        yield e0, min(n_ev, e0 + chunk)


def _weights(snapshot, is_first, scope_labels, e0, e1):
    src, dst = snapshot.src[e0:e1], snapshot.dst[e0:e1]
    unit_mask = {'events': np.ones(e1 - e0, dtype=bool), 'edges': is_first[e0:e1]}
    w = {}
    for scope, label in scope_labels.items():
        in_scope = np.ones(e1 - e0, dtype=bool) if label is None else label[src] == label[dst]
        for unit in UNITS:
            w[(unit, scope)] = in_scope & unit_mask[unit]
    return w


def batch_error_counts(snapshot, batch_size, scope_labels, chunk_events=20_000_000):
    """Returns (nodes_per_batch, counts, totals).

    counts[(unit, scope)][kind] is a per-batch int64 array, totals[(unit, scope)]
    the number of in-scope events in the final graph.
    """
    n_ev, N = snapshot.num_events, snapshot.num_nodes
    nb = (n_ev + batch_size - 1) // batch_size
    stride = nb + 1
    is_first = np.zeros(n_ev, dtype=bool)
    is_first[snapshot.first_event] = True
    keys = [(unit, scope) for scope in scope_labels for unit in UNITS]

    # pass 1: per-node incident totals and per-batch event counts over the final graph
    node_total = {k: np.zeros(N, dtype=np.int64) for k in keys}
    batch_events = {k: np.zeros(nb, dtype=np.int64) for k in keys}
    for e0, e1 in _chunks(n_ev, batch_size, chunk_events):
        src, dst = snapshot.src[e0:e1], snapshot.dst[e0:e1]
        nonself = src != dst
        b = np.arange(e0, e1, dtype=np.int64) // batch_size
        for k, w in _weights(snapshot, is_first, scope_labels, e0, e1).items():
            node_total[k] += np.bincount(src[w], minlength=N) + np.bincount(dst[w & nonself], minlength=N)
            batch_events[k] += np.bincount(b[w], minlength=nb)

    # pass 2: per (batch, node) known / inbatch / future
    nodes_per_batch = np.zeros(nb, dtype=np.int64)
    running = {k: np.zeros(N, dtype=np.int64) for k in keys}
    acc = {k: {kind: np.zeros(nb, dtype=np.int64) for kind in KINDS[:3]} for k in keys}
    for e0, e1 in _chunks(n_ev, batch_size, chunk_events):
        src, dst = snapshot.src[e0:e1], snapshot.dst[e0:e1]
        nonself = np.nonzero(src != dst)[0]
        b = np.arange(e0, e1, dtype=np.int64) // batch_size
        ev = np.concatenate([np.arange(e1 - e0), nonself])  # one entry per (event, endpoint)
        q, inv = np.unique(np.concatenate([src * stride + b, dst[nonself] * stride + b[nonself]]),
                           return_inverse=True)
        qn, qb = q // stride, q % stride
        nodes_per_batch += np.bincount(qb, minlength=nb)
        # q is sorted by (node, batch): index of the first entry of each node's segment
        seg_first = np.maximum.accumulate(np.where(np.r_[True, qn[1:] != qn[:-1]], np.arange(len(q)), 0))
        for k, w in _weights(snapshot, is_first, scope_labels, e0, e1).items():
            cnt = np.bincount(inv, weights=w[ev], minlength=len(q)).astype(np.int64)
            before = np.cumsum(cnt) - cnt
            known = running[k][qn] + before - before[seg_first]
            future = node_total[k][qn] - known - cnt
            for kind, c in (('known', known), ('inbatch', cnt), ('future', future)):
                acc[k][kind] += np.bincount(qb, weights=c, minlength=nb).astype(np.int64)
            running[k] += np.bincount(qn, weights=cnt, minlength=N).astype(np.int64)

    totals = {k: int(batch_events[k].sum()) for k in keys}
    for k in keys:
        acc[k]['global_future'] = totals[k] - np.cumsum(batch_events[k])
    return nodes_per_batch, acc, totals


def summarize(nodes_per_batch, counts, total):
    future = counts['future']
    seen = counts['known'] + counts['inbatch'] + future
    total_nodes = int(nodes_per_batch.sum())
    total_seen = int(seen.sum())
    return {
        'num_batches': len(future),
        'total_future': int(future.sum()),
        'mean_future_per_batch': float(future.mean()),
        'max_future_per_batch': int(future.max()),
        'mean_future_per_node': float(future.sum() / max(total_nodes, 1)),
        'future_rate': float(future.sum() / max(total_seen, 1)),
        'total_inbatch': int(counts['inbatch'].sum()),
        'total_known': int(counts['known'].sum()),
        'total_incident': total_seen,
        'total_node_occurrences': total_nodes,
        'final_graph_size': total,
        'mean_global_future_per_batch': float(counts['global_future'].mean()),
        'mean_global_future_rate': float(counts['global_future'].mean() / max(total, 1)),
    }
