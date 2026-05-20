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
prep_re       = re.compile(r'prep time details: to_dgl_blocks:([\d.]+)s prepare_input:([\d.]+)s mailbox_prep:([\d.]+)s mailbox_update:([\d.]+)s batch_postprocessing:([\d.]+)s')
profile_re    = re.compile(r'total_tensor_build_time: ([\d.]+)s\s+cuda_copy_time: ([\d.]+)s\s+total_to_dgl_blocks_time: ([\d.]+)s')
profile_per_re = re.compile(r'per_block_tensor_build: ([\d.]+)s\s+per_block_cuda_copy: ([\d.]+)s\s+per_block_total: ([\d.]+)s')

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
        m = prep_re.search(line)
        if m and current is not None:
            current.update({
                'prep_to_dgl_blocks': m.group(1),
                'prep_prepare_input': m.group(2),
                'prep_mailbox_prep': m.group(3),
                'prep_mailbox_update': m.group(4),
                'prep_batch_postprocessing': m.group(5),
            })
            continue
        m = profile_re.search(line)
        if m and current is not None:
            current.update({'dgl_build_time': m.group(1), 'dgl_cuda_time': m.group(2),
                            'dgl_total_time': m.group(3)})
            continue
        m = profile_per_re.search(line)
        if m and current is not None:
            current.update({'dgl_avg_build': m.group(1), 'dgl_avg_cuda': m.group(2),
                            'dgl_avg_total': m.group(3)})
            continue
        m = best_re.search(line)
        if m:
            rows.append({'epoch': 'best', 'val_ap': m.group(2), 'val_auc': m.group(3),
                         'best_epoch': m.group(1)})

flush(current, rows)

if not rows:
    print("No metrics found in log — CSV not written.")
    sys.exit(0)

fieldnames = ['epoch', 'train_loss', 'val_ap', 'val_auc',
              'time_total', 'time_sample', 'time_prep', 'time_model',
              'prep_to_dgl_blocks', 'prep_prepare_input', 'prep_mailbox_prep', 'prep_mailbox_update', 'prep_batch_postprocessing',
              'dgl_build_time', 'dgl_cuda_time', 'dgl_total_time',
              'dgl_avg_build', 'dgl_avg_cuda', 'dgl_avg_total',
              'flip_mean', 'flip_std', 'flip_min', 'flip_max', 'flip_batches',
              'best_epoch']
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
