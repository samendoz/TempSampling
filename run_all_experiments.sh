#!/usr/bin/env bash
# run_all_experiments.sh — run collect_results.sh over a set of experiment configs
#
# Two modes:
#
# 1. Adapt-exp configs (train_simple.py, --extra_config):
#    bash run_all_experiments.sh --exp_dir config/adapt_exp [--configs "test_01 test_02"]
#
# 2. Dataset + model (train_simple.py, looks up config/adapt_exp/<DATA>/<MODEL>.yml):
#    bash run_all_experiments.sh --data WIKI [--models "TGAT TGN APAN"]
#
# Examples:
#   bash run_all_experiments.sh --data WIKI
#   bash run_all_experiments.sh --data WIKI --models "TGAT TGN"
#   bash run_all_experiments.sh --data REDDIT --models "TGAT TGN APAN"
#   bash run_all_experiments.sh --exp_dir config/adapt_exp
#   bash run_all_experiments.sh --exp_dir config/adapt_exp --configs "test_01 test_02"

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATA=""
MODELS="TGAT TGN APAN"
EXP_DIR=""
CONFIGS=""
EXTRA_ARGS=()

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)     DATA="$2";     shift 2 ;;
        --models)   MODELS="$2";   shift 2 ;;
        --exp_dir)  EXP_DIR="$2";  shift 2 ;;
        --configs)  CONFIGS="$2";  shift 2 ;;
        --)         shift; EXTRA_ARGS=("$@"); break ;;
        *)          echo "Unknown argument: $1"; exit 1 ;;
    esac
done

FAILED=()

# ---------------------------------------------------------------------------
# Mode 1: adapt-exp configs via train_simple.py
# ---------------------------------------------------------------------------
if [ -n "$EXP_DIR" ]; then
    EXP_DIR="$REPO_DIR/$EXP_DIR"
    if [ ! -d "$EXP_DIR" ]; then
        echo "ERROR: exp_dir not found: $EXP_DIR"
        exit 1
    fi

    if [ -n "$CONFIGS" ]; then
        # explicit list
        CONFIG_FILES=()
        for name in $CONFIGS; do
            CONFIG_FILES+=("$EXP_DIR/${name}.yml")
        done
    else
        # all .yml files in exp_dir (skip readme)
        mapfile -t CONFIG_FILES < <(find "$EXP_DIR" -maxdepth 1 -name "*.yml" ! -name "readme*" | sort)
    fi

    echo "Mode    : adapt-exp (train_simple.py)"
    echo "Configs : ${CONFIG_FILES[*]}"
    echo "========================================"

    for cfg in "${CONFIG_FILES[@]}"; do
        if [ ! -f "$cfg" ]; then
            echo "WARNING: config not found: $cfg — skipping."
            continue
        fi
        echo ""
        echo ">>> Running experiment: $(basename $cfg)..."
        echo "----------------------------------------"
        if bash "$REPO_DIR/collect_results.sh" --extra_config "$cfg" "${EXTRA_ARGS[@]}"; then
            echo "<<< $(basename $cfg): DONE"
        else
            echo "<<< $(basename $cfg): FAILED (exit $?)"
            FAILED+=("$(basename $cfg)")
        fi
    done

# ---------------------------------------------------------------------------
# Mode 2: dataset + model — looks up config/adapt_exp/<DATA>/<MODEL>.yml
# ---------------------------------------------------------------------------
elif [ -n "$DATA" ]; then
    echo "Mode    : adapt-exp by dataset (train_simple.py)"
    echo "Dataset : $DATA"
    echo "Models  : $MODELS"
    echo "========================================"

    for MODEL in $MODELS; do
        CFG="$REPO_DIR/config/adapt_exp/$DATA/${MODEL}.yml"
        if [ ! -f "$CFG" ]; then
            echo "WARNING: config not found for $MODEL on $DATA ($CFG) — skipping."
            continue
        fi
        echo ""
        echo ">>> Running $MODEL on $DATA..."
        echo "----------------------------------------"
        if bash "$REPO_DIR/collect_results.sh" --extra_config "$CFG" "${EXTRA_ARGS[@]}"; then
            echo "<<< $MODEL on $DATA: DONE"
        else
            echo "<<< $MODEL on $DATA: FAILED (exit $?)"
            FAILED+=("$MODEL")
        fi
    done

else
    echo "ERROR: provide either --data or --exp_dir."
    echo ""
    echo "Usage:"
    echo "  bash run_all_experiments.sh --data WIKI [--models \"TGAT TGN APAN\"]"
    echo "  bash run_all_experiments.sh --exp_dir config/adapt_exp [--configs \"test_01 test_02\"]"
    exit 1
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All experiments completed successfully."
else
    echo "Failed: ${FAILED[*]}"
    exit 1
fi
