#!/bin/bash
# Run the reordering / error-edge analysis on every dataset config.
# Usage: bash TGNNReordering/run_all.sh [extra run_analysis.py args, e.g. --batch_size 900]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
mkdir -p TGNNReordering/results
for d in WIKI REDDIT WIKI_TALK STACK_OVERFLOW GDELT; do
    python TGNNReordering/run_analysis.py --config "TGNNReordering/configs/$d.yml" "$@" \
        2>&1 | tee "TGNNReordering/results/log_$d.txt"
done
