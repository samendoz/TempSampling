#!/bin/bash
# Temporal-violation experiment: original Cascade (time order) vs. the same
# Cascade loop trained on non-temporal event orders (GCoD blocks and controls).
#
# Usage (from anywhere):
#   bash TGNNReordering/run_violation_experiment.sh <DATA> [MODEL]
#   e.g. bash TGNNReordering/run_violation_experiment.sh WIKI TGN
#
# Environment overrides:
#   EPOCHS="20"                       epochs per run (default: the dataset config's)
#   ORDERINGS="gcod random_block shuffle"
#   FRACS="0.01 0.1 1"                reordering window, fraction of training events (1 = global)
#   GPU=0
#   CURSOR_BASELINE=1                 also run time order with the original cursor sampler
#                                     (shows --neighbor_search does not change Cascade itself)
#
# Output: results/violation/<DATA>/<MODEL>/<run>.{csv,log,_batches.npz}
#         then: python TGNNReordering/summarize_violation.py results/violation/<DATA>/<MODEL>
set -u -o pipefail

DATA=${1:?usage: run_violation_experiment.sh <DATA> [MODEL]}
MODEL=${2:-TGN}
ORDERINGS=${ORDERINGS:-"gcod random_block shuffle"}
FRACS=${FRACS:-"0.01 0.1 1"}
GPU=${GPU:-0}
CURSOR_BASELINE=${CURSOR_BASELINE:-1}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

BASE=config/adapt_exp/$DATA/$MODEL.yml
[ -f "$BASE" ] || { echo "missing $BASE"; exit 1; }
OUT=results/violation/$DATA/$MODEL
mkdir -p "$OUT"

COMMON=(--extra_config "$BASE" --gpu "$GPU" --violation_metrics --final_test)
[ -n "${EPOCHS:-}" ] && COMMON+=(--epoch "$EPOCHS")

run() {
    local tag=$1; shift
    echo "=================== $DATA $MODEL run: $tag ==================="
    if ! python train_simple.py "${COMMON[@]}" --run_tag "$tag" \
            --model_name "viol_${DATA}_${MODEL}_${tag}" \
            --exp_csv "$OUT/$tag.csv" "$@" 2>&1 | tee "$OUT/$tag.log"; then
        echo "RUN FAILED: $tag" | tee -a "$OUT/failures.txt"
    fi
}

# A: original Cascade, time order
run time --neighbor_search
[ "$CURSOR_BASELINE" = 1 ] && run time_cursor

# B: non-temporal processing orders
for o in $ORDERINGS; do
    for f in $FRACS; do
        path=$(python TGNNReordering/make_event_order.py --data "$DATA" --ordering "$o" --window_frac "$f" \
               | tee /dev/stderr | sed -n 's/^\[.*\] wrote //p' | tail -1)
        if [ -z "$path" ]; then
            echo "ORDER FAILED: $o $f" | tee -a "$OUT/failures.txt"
            continue
        fi
        tag=$(basename "$path" .npz)
        run "$tag" --neighbor_search --event_order "$path"
    done
done

python TGNNReordering/summarize_violation.py "$OUT"
