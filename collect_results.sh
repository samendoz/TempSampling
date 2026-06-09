#!/usr/bin/env bash
# collect_results.sh — run train.py or train_simple.py and capture output to structured log + CSV
#
# Usage (train.py style):
#   bash collect_results.sh --data WIKI --config config/TGN.yml [extra train.py args]
#
# Usage (train_simple.py style):
#   bash collect_results.sh --extra_config config/adapt_exp/test_01.yml
#
# Output:
#   results/<STEM>.log   — full stdout/stderr
#   results/<STEM>.csv   — per-epoch metrics + best result row

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$REPO_DIR/results"
mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Detect invocation mode and extract DATA / MODEL for filename
# ---------------------------------------------------------------------------
DATA=""
CONFIG=""
EXTRA_CONFIG=""
ARGS=("$@")

for i in "${!ARGS[@]}"; do
    case "${ARGS[$i]}" in
        --data)         DATA="${ARGS[$((i+1))]}" ;;
        --config)       CONFIG="${ARGS[$((i+1))]}" ;;
        --extra_config) EXTRA_CONFIG="${ARGS[$((i+1))]}" ;;
    esac
done

if [ -n "$EXTRA_CONFIG" ]; then
    # train_simple.py mode — read data/config from YAML
    SCRIPT="train_simple.py"
    DATA=$(python3 -c "import yaml; d=yaml.safe_load(open('$EXTRA_CONFIG')); print(d.get('data','unknown'))" 2>/dev/null || echo "unknown")
    CONFIG=$(python3 -c "import yaml; d=yaml.safe_load(open('$EXTRA_CONFIG')); print(d.get('config','unknown'))" 2>/dev/null || echo "unknown")
    EXP=$(basename "${EXTRA_CONFIG%.*}")
    MODEL=$(basename "${CONFIG%.*}")
    STEM="${DATA}_${MODEL}_${EXP}_$(date +%Y%m%d_%H%M%S)"
else
    # train.py mode
    SCRIPT="train.py"
    MODEL=$(basename "${CONFIG%.*}")
    STEM="${DATA}_${MODEL}_$(date +%Y%m%d_%H%M%S)"
fi

LOG_FILE="$RESULTS_DIR/${STEM}.log"
CSV_FILE="$RESULTS_DIR/${STEM}.csv"

echo "Script  : $SCRIPT"
echo "Data    : $DATA"
echo "Model   : $MODEL"
echo "Logging : $LOG_FILE"
echo "CSV     : $CSV_FILE"
echo "---"

# ---------------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------------
python "$REPO_DIR/$SCRIPT" "$@" 2>&1 | tee "$LOG_FILE"

# ---------------------------------------------------------------------------
# Parse log into CSV
# ---------------------------------------------------------------------------
python3 - "$LOG_FILE" "$CSV_FILE" << 'PYEOF'
import sys, re, csv

log_path = sys.argv[1]
csv_path = sys.argv[2]
flip_dist_path = csv_path.replace('.csv', '_flip_dist.csv')

epoch_re      = re.compile(r'Epoch\s+(\d+):')
metrics_re    = re.compile(r'train loss:([\d.]+)\s+val ap:([\d.]+)\s+val auc:([\d.]+)')
timing_re     = re.compile(r'total time:([\d.]+)s\s+sample time:([\d.]+)s\s+prep time:([\d.]+)s\s+model time:([\d.]+)s')
best_re       = re.compile(r'Best epoch:(\d+)\s+Best AP:([\d.]+)\s+Best AUC:([\d.]+)')
flip_re       = re.compile(r'stable flag flip ratio.*?mean:([\d.]+).*?std:([\d.]+).*?min:([\d.]+).*?max:([\d.]+).*?batches:(\d+)')
flip_list_re  = re.compile(r'stable flag flip list: ([\d. ]+)')
# print('\tcreate_block_time: {:.6f}s cuda_copy_time: {:.6f}s total_to_dgl_blocks_time: {:.6f}s combine_first_time: {:.6f}s node_index_time: {:.6f}s edge_index_time: {:.6f}s node_cuda_time: {:.6f}s edge_cuda_time: {:.6f}s create_dgl_block_time: {:.6f}s src_id_time: {:.6f}s edge_dt_time: {:.6f}s src_ts_time: {:.6f}s id_time: {:.6f}s'.format(create_block_time, cuda_copy_time, total_time, combine_first_time, node_index_time, edge_index_time, node_cuda_time, edge_cuda_time, create_dgl_block_time, src_id_time, edge_dt_time, src_ts_time, id_time))
dgl_profile_re    = re.compile(r'create_block_time: ([\d.]+)s cuda_copy_time: ([\d.]+)s total_to_dgl_blocks_time: ([\d.]+)s combine_first_time: ([\d.]+)s node_index_time: ([\d.]+)s edge_index_time: ([\d.]+)s node_cuda_time: ([\d.]+)s edge_cuda_time: ([\d.]+)s create_dgl_block_time: ([\d.]+)s src_id_time: ([\d.]+)s edge_dt_time: ([\d.]+)s src_ts_time: ([\d.]+)s id_time: ([\d.]+)s')
# Mailbox Index Time: 0.002116s Mailbox Update Index Time: 0.007210s Mailbox Update Deduplication Time: 0.005352s Mailbox Update Write Time: 0.002401s Memory Stability Prep Time: 0.000735s Memory Stability Math Time: 0.003773s Memory Stability Write Time: 0.000503s
mailbox_profile_re = re.compile(r'Mailbox Index Time: ([\d.]+)s Mailbox Update Index Time: ([\d.]+)s Mailbox Update Deduplication Time: ([\d.]+)s Mailbox Update Write Time: ([\d.]+)s Memory Stability Prep Time: ([\d.]+)s Memory Stability Math Time: ([\d.]+)s Memory Stability Write Time: ([\d.]+)s')
sampling_re   = re.compile(r'sampling time: ([\d.]+)s, updating indptr time: ([\d.]+)s, updating stable flag time: ([\d.]+)s')
estimated_prep_times = re.compile(r'estimated initial to_dgl_blocks time: ([\d.]+)s, prepare_input time: ([\d.]+)s, mailbox prep time: ([\d.]+)s, post to_dgl_blocks time: ([\d.]+)s, mailbox update time: ([\d.]+)s, updating indptr and stable flag time: ([\d.]+)s')

#add the results fetched by sampling_re to the CSV as well, with keys sampling_time, updating_indptr_time, updating_stable_flag_time

rows = []
flip_dist_rows = []
current = None

def flush(current, rows):
    if current and 'train_loss' in current:
        rows.append(current)

with open(log_path) as f:
    for line in f:
        line = line.strip()
        m = epoch_re.search(line)
        if m:
            flush(current, rows)
            current = {'epoch': m.group(1)}
            continue
        m = metrics_re.search(line)
        if m and current is not None:
            current.update({'train_loss': m.group(1), 'val_ap': m.group(2), 'val_auc': m.group(3)})
            continue
        m = timing_re.search(line)
        if m and current is not None:
            current.update({'time_total': m.group(1), 'time_sample': m.group(2),
                            'time_prep': m.group(3), 'time_model': m.group(4)})
            continue
        m = flip_re.search(line)
        if m and current is not None:
            current.update({'flip_mean': m.group(1), 'flip_std': m.group(2),
                            'flip_min': m.group(3), 'flip_max': m.group(4),
                            'flip_batches': m.group(5)})
            continue
        m = flip_list_re.search(line)
        if m and current is not None:
            epoch = current.get('epoch', '')
            for batch_idx, val in enumerate(m.group(1).split()):
                flip_dist_rows.append({'epoch': epoch, 'batch': batch_idx, 'flip_ratio': val})
            continue
        # m = prep_re.search(line)
        # if m and current is not None:
        #     current.update({
        #         'prep_to_dgl_blocks': m.group(1),
        #         'prep_prepare_input': m.group(2),
        #         'prep_mailbox_prep': m.group(3),
        #         'prep_mailbox_update': m.group(4),
        #         'prep_batch_postprocessing': m.group(5),
        #     })
        #     continue
        m = dgl_profile_re.search(line)
        if m and current is not None:
            current.update({'create_block_time': m.group(1), 'cuda_copy_time': m.group(2),
                            'total_to_dgl_blocks_time': m.group(3), 'combine_first_time': m.group(4),
                            'node_index_time': m.group(5), 'edge_index_time': m.group(6),
                            'node_cuda_time': m.group(7), 'edge_cuda_time': m.group(8),
                            'create_dgl_block_time': m.group(9), 'src_id_time': m.group(10),
                            'edge_dt_time': m.group(11), 'src_ts_time': m.group(12), 'id_time': m.group(13)})
            continue
        m = mailbox_profile_re.search(line)
        if m and current is not None:
            current.update({'mailbox_index_time': m.group(1), 'mailbox_up_index_time': m.group(2),
                            'mailbox_up_dedup_time': m.group(3), 'mailbox_up_write_time': m.group(4),
                            'mem_stab_prep_time': m.group(5), 'mem_stab_math_time': m.group(6),
                            'mem_stab_write_time': m.group(7)})
            continue
        m = best_re.search(line)
        if m:
            rows.append({'epoch': 'best', 'val_ap': m.group(2), 'val_auc': m.group(3),
                         'best_epoch': m.group(1)})
            continue
        m = sampling_re.search(line)
        if m and current is not None:
            current.update({'sampling_time': m.group(1), 'updating_indptr_time': m.group(2),
                            'updating_stable_flag_time': m.group(3)})
            continue
        m = estimated_prep_times.search(line)
        if m and current is not None:
            current.update({
                'estimated_prep_to_dgl_blocks': m.group(1),
                'estimated_prepare_input': m.group(2),
                'estimated_mailbox_prep': m.group(3),
                'estimated_post_to_dgl_blocks': m.group(4),
                'estimated_mailbox_update': m.group(5),
                'estimated_updating_indptr_and_stable_flag': m.group(6),
            })
            continue

flush(current, rows)

if not rows:
    print("No metrics found in log — CSV not written.")
    sys.exit(0)

fieldnames = ['epoch', 'train_loss', 'val_ap', 'val_auc',
              'time_total', 'time_sample', 'time_prep', 'time_model',
              # 'prep_to_dgl_blocks', 'prep_prepare_input', 'prep_mailbox_prep', 'prep_mailbox_update', 'prep_batch_postprocessing',
              'create_block_time', 'cuda_copy_time', 'total_to_dgl_blocks_time', 'combine_first_time', 'node_index_time', 'edge_index_time', 'node_cuda_time', 'edge_cuda_time',
              'create_dgl_block_time', 'src_id_time', 'edge_dt_time', 'src_ts_time', 'id_time',
              'mailbox_index_time', 'mailbox_up_index_time', 'mailbox_up_dedup_time', 'mailbox_up_write_time',
              'mem_stab_prep_time', 'mem_stab_math_time', 'mem_stab_write_time',
              'flip_mean', 'flip_std', 'flip_min', 'flip_max', 'flip_batches',
              'best_epoch', 'sampling_time', 'updating_indptr_time', 'updating_stable_flag_time',
              'estimated_prep_to_dgl_blocks', 'estimated_prepare_input', 'estimated_mailbox_prep', 'estimated_post_to_dgl_blocks', 'estimated_mailbox_update', 'estimated_updating_indptr_and_stable_flag']
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
print(f"Metrics saved to {csv_path} ({len(rows)} rows)")

if flip_dist_rows:
    with open(flip_dist_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'batch', 'flip_ratio'])
        w.writeheader()
        w.writerows(flip_dist_rows)
    print(f"Flip distribution saved to {flip_dist_path} ({len(flip_dist_rows)} rows)")
PYEOF
