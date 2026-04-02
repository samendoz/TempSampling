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

epoch_re   = re.compile(r'Epoch\s+(\d+):')
metrics_re = re.compile(r'train loss:([\d.]+)\s+val ap:([\d.]+)\s+val auc:([\d.]+)')
timing_re  = re.compile(r'total time:([\d.]+)s\s+sample time:([\d.]+)s\s+prep time:([\d.]+)s\s+model time:([\d.]+)s')
best_re    = re.compile(r'Best epoch:(\d+)\s+Best AP:([\d.]+)\s+Best AUC:([\d.]+)')

rows = []
current = {}

with open(log_path) as f:
    for line in f:
        line = line.strip()
        m = epoch_re.search(line)
        if m:
            current = {'epoch': m.group(1)}
            continue
        m = metrics_re.search(line)
        if m and current:
            current.update({'train_loss': m.group(1), 'val_ap': m.group(2), 'val_auc': m.group(3)})
            continue
        m = timing_re.search(line)
        if m and current:
            current.update({'time_total': m.group(1), 'time_sample': m.group(2),
                            'time_prep': m.group(3), 'time_model': m.group(4)})
            rows.append(current)
            current = {}
            continue
        m = best_re.search(line)
        if m:
            rows.append({'epoch': 'best', 'val_ap': m.group(2), 'val_auc': m.group(3),
                         'best_epoch': m.group(1)})

if not rows:
    print("No metrics found in log — CSV not written.")
    sys.exit(0)

fieldnames = ['epoch', 'train_loss', 'val_ap', 'val_auc',
              'time_total', 'time_sample', 'time_prep', 'time_model', 'best_epoch']
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)

print(f"Metrics saved to {csv_path} ({len(rows)} rows)")
PYEOF
