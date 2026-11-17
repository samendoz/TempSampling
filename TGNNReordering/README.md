# TGNNReordering

GCoD-style graph reordering (from `ReorderingExperiment/GCoD-main`) applied to the
**final snapshot of a TGNN dataset's training partition**, plus a per-batch count of
**error edges**: temporal events that the reordered static graph exposes to a batch
before they have happened (future-information leakage).

Supported datasets: `WIKI`, `REDDIT`, `WIKI_TALK`, `STACK_OVERFLOW`, `GDELT`.

---

## 1. Environment setup

This uses the **same environment as Cascade/TempSampling** (`ts_env`). It needs only
numpy, pandas, pyyaml and METIS (through DGL). No C++ extensions and no GPU.

### If `ts_env` already exists (the training server)

```bash
cd /home/cc/TempSampling            # repo root
source ts_env/bin/activate
```

### Fresh machine

```bash
cd TempSampling
bash experimental_setup.sh          # creates ts_env/ (torch 2.2 + DGL 2.1, also builds Cascade's extensions)
source ts_env/bin/activate
```

`experimental_setup.sh` also compiles Cascade's C++ samplers and expects `g++`. This
analysis doesn't use them, so if that step fails here you can ignore it.

### Check that METIS is available

```bash
python -c "import dgl, torch; g = dgl.graph(([0,1,2],[1,2,0])); print(dgl.metis_partition_assignment(g, 2))"
```

If that fails (DGL built without METIS, or the DGL import is broken), install the
fallback. The code picks it up automatically:

```bash
pip install pymetis
```

---

## 2. Datasets

The analysis reuses **Cascade's datasets** and reads only `DATA/<NAME>/edges.csv`
(columns `,src,dst,time,ext_roll`; `ext_roll == 0` is the training split).
`ext_full.npz` and the feature files are not needed.

If you already train Cascade on a machine, the data is already there. Otherwise:

```bash
bash down.sh                        # WIKI, REDDIT, GDELT edges.csv (+ Cascade files); SNAP raw files
# WIKI_TALK and STACK_OVERFLOW come as raw SNAP .txt.gz -> convert to edges.csv:
gunzip DATA/WIKI_TALK/wiki-talk-temporal.txt.gz
python dataset_transform.py --data WIKI_TALK
gunzip DATA/STACK_OVERFLOW/sx-stackoverflow.txt.gz
python dataset_transform.py --data STACK_OVERFLOW
```

Data stored somewhere else? Pass `--data_root /path/to/DATA` to any command below.

---

## 3. Verify the implementation (run this first)

```bash
python TGNNReordering/test_error_edges.py                   # uses real METIS
python TGNNReordering/test_error_edges.py --stub_partition  # no METIS needed
```

This builds a small synthetic dataset (repeated pairs, self loops, val/test-only nodes),
runs the full pipeline, and recomputes every per-batch count with plain Python loops.
It should end with `ALL CHECKS PASSED` (it takes a few seconds). The run without
`--stub_partition` also checks that the METIS call works in your environment.

---

## 4. Run the analysis

All commands are run from the **repo root**.

```bash
# one dataset (start with WIKI, the smallest)
python TGNNReordering/run_analysis.py --config TGNNReordering/configs/WIKI.yml

# override batch sizes
python TGNNReordering/run_analysis.py --config TGNNReordering/configs/REDDIT.yml --batch_size 900 4000 8000

# all five datasets, logs to TGNNReordering/results/log_<NAME>.txt
bash TGNNReordering/run_all.sh
bash TGNNReordering/run_all.sh --batch_size 900     # extra args are passed through
```

GDELT, STACK_OVERFLOW and WIKI_TALK are large. Run them inside `tmux` or with
`nohup`, e.g.

```bash
nohup python TGNNReordering/run_analysis.py --config TGNNReordering/configs/GDELT.yml \
    > TGNNReordering/results/log_GDELT.txt 2>&1 &
```

### Configuration (`TGNNReordering/configs/<NAME>.yml`)

| Key | Default | Meaning |
|-----|---------|---------|
| `data_root` | `DATA` | dataset folder, relative to the repo root |
| `batch_size` | `[900, 8000]` | 900 = TGL batch size, 8000 = Cascade `max_batch_size` |
| `degree_split` | `null` | explicit degree thresholds for GCoD's buckets, e.g. `[3, 6]` |
| `degree_quantiles` | `[0.5, 0.8]` | used when `degree_split` is null |
| `total_subgraphs` | `12` | number of dense diagonal blocks (GCoD default); divisible by `num_groups` |
| `num_groups` | `2` | coarse diagonal groups |
| `baselines` | `[identity, random]` | comparison orderings with the same block sizes |
| `output_dir` | `results/<NAME>` | relative to `TGNNReordering/` |
| `save_per_batch` | `true` | write per-batch arrays |
| `chunk_events` | `20000000` | events per vectorized chunk; lower it if memory is tight |

---

## 5. What is measured

1. **Snapshot**: all training events are merged into one undirected static graph
   (deduplicated, self loops removed). This is the graph that gets reordered.
2. **Reordering** ([gcod_reorder.py](gcod_reorder.py)): GCoD's `my_partition_graph`.
   Nodes are bucketed by degree, METIS splits each bucket, the parts are dealt round-robin
   into groups, and nodes are renumbered in that order. The METIS parts are the **dense
   diagonal blocks**; everything off the diagonal is the **sparse region**.
3. **Batches**: consecutive blocks of `batch_size` training events (TGL order,
   positive events only).
4. **Error edges** ([error_edges.py](error_edges.py)). Every event is its own temporal
   edge, so two A–B interactions at different times are two edges.
   - *Node level*: for each node in batch `b`, each incident event is `known` (earlier
     batch), `inbatch` (same batch) or **`future`** (later batch = violation). Counts
     are per (batch, node).
   - *Graph level* (end of batch `b` vs. final graph): `global_future[b]` = events of
     the final graph that have not happened yet when batch `b` ends.
   - Unit `events` counts every event (main metric). Unit `edges` counts only the first
     event per node pair, i.e. leakage of neighbor existence alone.
   - Each metric is reported for `all`, `block_diag` (dense) and `block_offdiag`
     (sparse), plus `group_diag` / `group_offdiag`, for GCoD and for each baseline.

The total over `all` events doesn't depend on the ordering. Reordering only moves events
between the dense and sparse regions, so compare `gcod` against `identity`/`random`.

---

## 6. Outputs (`TGNNReordering/results/<NAME>/`)

| File | Content |
|------|---------|
| `summary.csv` | one row per (batch_size, unit, ordering, scope): `total_future`, `mean_future_per_batch`, `mean_future_per_node`, `future_rate` (future / all incident), known/inbatch totals, `final_graph_size`, `mean_global_future_rate` |
| `summary.json` | the same plus dataset stats, degree split, block sizes, fraction of static edges on the diagonal per ordering |
| `reorder.npz` | `<ordering>_order` (new position -> node id), `block_sizes`, `block_group`, `block_bucket` |
| `per_batch_bs<B>.npz` | per-batch arrays `<unit>/<scope>/{known,inbatch,future,global_future}`, `nodes_per_batch` |

The main comparison is in `summary.csv`: filter `unit = events` and compare `future_rate`
between `gcod` and the baselines, for `block_diag` (dense) and `block_offdiag` (sparse).

Example of loading the per-batch arrays:

```python
import numpy as np
pb = np.load('TGNNReordering/results/WIKI/per_batch_bs900.npz')
dense_future = pb['events/gcod/block_diag/future']    # one value per batch
sparse_future = pb['events/gcod/block_offdiag/future']
```

`results/` is gitignored.

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: pymetis` | DGL METIS failed to import too: `pip install pymetis`, or fix the DGL import (see `experimental_setup.sh` graphbolt patch) |
| `ModuleNotFoundError: yaml` | `pip install pyyaml` (or activate `ts_env`) |
| `FileNotFoundError: .../edges.csv` | dataset not in `DATA/<NAME>/`; see section 2 or pass `--data_root` |
| killed / out of memory on GDELT | set `chunk_events: 5000000` in `configs/GDELT.yml`; reading GDELT's `edges.csv` itself needs several GB of RAM |
| `total_subgraphs should be divided by num_groups` | make `total_subgraphs` a multiple of `num_groups` |
| `bucket i rounded to 0 subgraphs` warning | harmless: that degree bucket gets 1 block; raise `total_subgraphs` or change the degree split for finer blocks |

---

## 8. Training experiment: does a non-temporal order hurt Cascade? (`temporal_violation` branch)

The counts above say how much future information a reordered static graph *exposes*,
not how much it changes *training*. This experiment trains Cascade (unchanged loop,
color sampler, memory) on the training events in a non-temporal order and compares
accuracy and speed against Cascade in time order.

| Run | Training order | Purpose |
|-----|----------------|---------|
| `time` | time order (original Cascade), `--neighbor_search` | baseline |
| `time_cursor` | time order, original cursor sampler | shows `--neighbor_search` alone doesn't change Cascade |
| `gcod_w<win>_src` | GCoD blocks of the training snapshot, sorted by block (time order inside a block) within windows | the "reorder like a static GNN" case |
| `random_block_w<win>_src` | same, random node blocks of the same sizes | GCoD structure vs. any block grouping |
| `shuffle_w<win>` | events shuffled within windows | breaking time order with no structure |

`<win>` sweeps the reordering window (`wf0.01`, `wf0.1` = 1% / 10% of the training events,
`wglobal` = whole training set), so the result is a dose-response curve:
more violation → how much accuracy lost, for how much speed.

What stays fixed in every run: the model, configs, seed, Cascade's color sampler (it
batches whatever order it is given; it sees processing positions as event ids), and the
evaluation: validation after each epoch and the final test always run in **time order**
with time-filtered neighbors. For the test number the best model (by val AP) is reloaded
and the memory is rebuilt by replaying train + val in time order, so leaked future
information cannot inflate the test score.

### Setup (server)

```bash
git checkout temporal_violation
python setup.py build_ext --inplace                 # sampler_core.cpp changed (set_force_search)
python TGNNReordering/test_event_order.py           # CPU-only, should print ALL CHECKS PASSED
python TGNNReordering/verify_neighbor_search.py --data WIKI   # should print PASS
```

`--neighbor_search` makes the neighbor sampler binary-search each query's time window
instead of using its per-node cursors. The cursors assume time never goes backwards between
batches; out of time order they stay ahead and silently return fewer neighbors, which would
confound the experiment. In time order both paths return the same neighbors, except for the
known repeated-node cursor bug, which `verify_neighbor_search.py` reports.

### Run

```bash
bash TGNNReordering/run_violation_experiment.sh WIKI TGN
EPOCHS=30 FRACS="0.01 0.1 1" bash TGNNReordering/run_violation_experiment.sh REDDIT TGN
python TGNNReordering/summarize_violation.py results/violation/WIKI/TGN
```

Event orders are cached in `DATA/<NAME>/event_orders/<tag>.npz` (build one by hand with
`python TGNNReordering/make_event_order.py --data WIKI --ordering gcod --window_frac 0.1`).
A single run by hand:

```bash
python train_simple.py --extra_config config/adapt_exp/WIKI/TGN.yml \
    --event_order DATA/WIKI/event_orders/gcod_wglobal_src.npz --neighbor_search \
    --violation_metrics --final_test --exp_csv results/violation/WIKI/TGN/gcod_wglobal_src.csv
```

Options given on the command line now override the same key in `--extra_config`
(previously the YAML always won, so `--epoch 20` was silently ignored).

### Outputs (`results/violation/<DATA>/<MODEL>/`)

| File | Content |
|------|---------|
| `<run>.csv` | one row per epoch (train loss, val AP/AUC, epoch / sample / prep / model time, batches, avg batch size, violation rates) + a final `test` row (best epoch, test AP/AUC, total train time) |
| `<run>_batches.npz` | per-batch violation counts, all epochs |
| `<run>.log` | full stdout |
| `summary.csv` | one row per run, with `d_*` = difference to the `time` run |

Violation metrics (per batch, read before the batch updates the memory):

| Metric | Meaning |
|--------|---------|
| `time_inversion_rate` | events older than an event already processed this epoch |
| `mem_future_rate` | event endpoints whose memory was already updated by a *later* event (future information reaching the model) |
| `nbr_known_rate` / `nbr_inbatch_rate` / `nbr_unprocessed_rate` | sampled neighbor events (always in the past by timestamp) that were processed in an earlier batch / are in the same batch (Cascade's own staleness) / were not processed yet this epoch |
| `nbr_per_root` | neighbors sampled per root (sanity check) |

The tracker's own runtime is excluded from the epoch timings.

### Known limits

- Only `batch_stable_freezing` and `batch_stable_freezing_large` modes; not with
  `--use_inductive` or `--use_full_edge`.
- GDELT: the reordered dataframe is a second copy of `edges.csv` in memory, and building the
  color graph sorts all events once (a few GB of RAM).
