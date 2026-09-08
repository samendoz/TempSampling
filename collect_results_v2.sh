#!/usr/bin/env bash
# collect_results_v2.sh — run the GIL-release / GPU-color-sampler scenarios
# (train_simple.py baseline vs. train_simple_v2.py --use_nogil_sampling /
# --use_gpu_color_sampler) and capture output to structured log + CSV, same
# convention as collect_results.sh but scenario-tagged so results from
# different flag combinations on the same config don't collide.
#
# Usage:
#   bash collect_results_v2.sh --extra_config config/adapt_exp/test_01.yml --scenario baseline
#   bash collect_results_v2.sh --extra_config config/adapt_exp/test_01.yml --scenario nogil
#   bash collect_results_v2.sh --extra_config config/adapt_exp/test_01.yml --scenario gpu
#   bash collect_results_v2.sh --extra_config config/adapt_exp/test_01.yml --scenario nogil_gpu
#   bash collect_results_v2.sh --extra_config config/adapt_exp/test_01.yml --scenario all
#
# Scenarios:
#   baseline   python train_simple.py    <extra_config>                                    (control, untouched)
#   v2         python train_simple_v2.py <extra_config>                                    (regression check: no flags, should match baseline)
#   nogil      python train_simple_v2.py <extra_config> --use_nogil_sampling
#   gpu        python train_simple_v2.py <extra_config> --use_gpu_color_sampler
#   nogil_gpu  python train_simple_v2.py <extra_config> --use_nogil_sampling --use_gpu_color_sampler
#   all        runs all five in sequence, then prints a comparison summary
#
# Any extra CLI args (e.g. --color_num_workers 4, --profile_to_dgl_blocks) are
# passed through to every scenario run.
#
# Output (per scenario):
#   results/<STEM>_<scenario>.log   — full stdout/stderr
#   results/<STEM>_<scenario>.csv   — per-epoch metrics + best result row (scenario column included)

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$REPO_DIR/results"
mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
EXTRA_CONFIG=""
SCENARIO=""
PASSTHROUGH_ARGS=()

ARGS=("$@")
i=0
while [ $i -lt ${#ARGS[@]} ]; do
    case "${ARGS[$i]}" in
        --extra_config)
            EXTRA_CONFIG="${ARGS[$((i+1))]}"
            i=$((i+2))
            ;;
        --scenario)
            SCENARIO="${ARGS[$((i+1))]}"
            i=$((i+2))
            ;;
        *)
            PASSTHROUGH_ARGS+=("${ARGS[$i]}")
            i=$((i+1))
            ;;
    esac
done

if [ -z "$EXTRA_CONFIG" ]; then
    echo "ERROR: --extra_config is required."
    echo "Usage: bash collect_results_v2.sh --extra_config <yaml> --scenario {baseline|v2|nogil|gpu|nogil_gpu|all} [extra args]"
    exit 1
fi
if [ -z "$SCENARIO" ]; then
    echo "ERROR: --scenario is required."
    echo "Usage: bash collect_results_v2.sh --extra_config <yaml> --scenario {baseline|v2|nogil|gpu|nogil_gpu|all} [extra args]"
    exit 1
fi

DATA=$(python3 -c "import yaml; d=yaml.safe_load(open('$EXTRA_CONFIG')); print(d.get('data','unknown'))" 2>/dev/null || echo "unknown")
CONFIG=$(python3 -c "import yaml; d=yaml.safe_load(open('$EXTRA_CONFIG')); print(d.get('config','unknown'))" 2>/dev/null || echo "unknown")
EXP=$(basename "${EXTRA_CONFIG%.*}")
MODEL=$(basename "${CONFIG%.*}")
RUN_TAG="${DATA}_${MODEL}_${EXP}_$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------------------
# Scenario -> (script, extra flags) lookup
# ---------------------------------------------------------------------------
scenario_script() {
    case "$1" in
        baseline)  echo "train_simple.py" ;;
        v2)        echo "train_simple_v2.py" ;;
        nogil)     echo "train_simple_v2.py" ;;
        gpu)       echo "train_simple_v2.py" ;;
        nogil_gpu) echo "train_simple_v2.py" ;;
        *) echo "" ;;
    esac
}

scenario_flags() {
    case "$1" in
        baseline)  echo "" ;;
        v2)        echo "" ;;
        nogil)     echo "--use_nogil_sampling" ;;
        gpu)       echo "--use_gpu_color_sampler" ;;
        nogil_gpu) echo "--use_nogil_sampling --use_gpu_color_sampler" ;;
        *) echo "" ;;
    esac
}

ALL_SCENARIOS=(baseline v2 nogil gpu nogil_gpu)

if [ "$SCENARIO" == "all" ]; then
    SCENARIOS_TO_RUN=("${ALL_SCENARIOS[@]}")
else
    SCRIPT_CHECK="$(scenario_script "$SCENARIO")"
    if [ -z "$SCRIPT_CHECK" ]; then
        echo "ERROR: unknown scenario '$SCENARIO'. Valid: baseline v2 nogil gpu nogil_gpu all"
        exit 1
    fi
    SCENARIOS_TO_RUN=("$SCENARIO")
fi

# ---------------------------------------------------------------------------
# Run each requested scenario
# ---------------------------------------------------------------------------
declare -A RESULT_LOG
declare -A RESULT_STATUS
FAILED=()

for scen in "${SCENARIOS_TO_RUN[@]}"; do
    SCRIPT="$(scenario_script "$scen")"
    FLAGS="$(scenario_flags "$scen")"
    STEM="${RUN_TAG}_${scen}"
    LOG_FILE="$RESULTS_DIR/${STEM}.log"
    CSV_FILE="$RESULTS_DIR/${STEM}.csv"
    RESULT_LOG[$scen]="$LOG_FILE"

    echo ""
    echo "========================================"
    echo "Scenario: $scen"
    echo "Script  : $SCRIPT"
    echo "Flags   : ${FLAGS:-(none)}"
    echo "Data    : $DATA"
    echo "Model   : $MODEL"
    echo "Logging : $LOG_FILE"
    echo "CSV     : $CSV_FILE"
    echo "----------------------------------------"

    set +e
    # shellcheck disable=SC2086
    python "$REPO_DIR/$SCRIPT" --extra_config "$EXTRA_CONFIG" $FLAGS "${PASSTHROUGH_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
    RUN_STATUS=${PIPESTATUS[0]}
    set -e
    RESULT_STATUS[$scen]=$RUN_STATUS
    if [ "$RUN_STATUS" -ne 0 ]; then
        echo "<<< scenario '$scen': FAILED (exit $RUN_STATUS)"
        FAILED+=("$scen")
        continue
    fi

    # -----------------------------------------------------------------
    # Parse log into CSV (same schema as collect_results.sh, plus a
    # scenario column so multiple scenarios' CSVs can be concatenated
    # directly for comparison)
    # -----------------------------------------------------------------
    python3 - "$LOG_FILE" "$CSV_FILE" "$scen" << 'PYEOF'
import sys, re, csv

log_path = sys.argv[1]
csv_path = sys.argv[2]
scenario = sys.argv[3]
flip_dist_path = csv_path.replace('.csv', '_flip_dist.csv')
batch_node_stats_path = csv_path.replace('.csv', '_batch_node_stats.csv')

epoch_re      = re.compile(r'Epoch\s+(\d+):')
metrics_re    = re.compile(r'train loss:([\d.]+)\s+val ap:([\d.]+)\s+val auc:([\d.]+)')
timing_re     = re.compile(r'total time:([\d.]+)s\s+sample time:([\d.]+)s\s+prep time:([\d.]+)s\s+model time:([\d.]+)s')
best_re       = re.compile(r'Best epoch:(\d+)\s+Best AP:([\d.]+)\s+Best AUC:([\d.]+)')
flip_re       = re.compile(r'stable flag flip ratio.*?mean:([\d.]+).*?std:([\d.]+).*?min:([\d.]+).*?max:([\d.]+).*?batches:(\d+)')
flip_list_re  = re.compile(r'stable flag flip list: ([\d. ]+)')
batch_node_re = re.compile(r'batch node stats: batch:(\d+) batch_size:(\d+) unique_pos_nodes:(\d+)')
dgl_profile_re    = re.compile(r'create_block_time: ([\d.]+)s cuda_copy_time: ([\d.]+)s total_to_dgl_blocks_time: ([\d.]+)s combine_first_time: ([\d.]+)s node_index_time: ([\d.]+)s edge_index_time: ([\d.]+)s node_cuda_time: ([\d.]+)s edge_cuda_time: ([\d.]+)s create_dgl_block_time: ([\d.]+)s src_id_time: ([\d.]+)s edge_dt_time: ([\d.]+)s src_ts_time: ([\d.]+)s id_time: ([\d.]+)s')
mailbox_profile_re = re.compile(r'Mailbox Index Time: ([\d.]+)s Mailbox Update Index Time: ([\d.]+)s Mailbox Update Deduplication Time: ([\d.]+)s Mailbox Update Write Time: ([\d.]+)s Mailbox Update CPU Time: ([\d.]+)s Mailbox Update CUDA Time: ([\d.]+)s Memory Stability Prep Time: ([\d.]+)s Memory Stability Math Time: ([\d.]+)s Memory Stability Write Time: ([\d.]+)s')
sampling_re   = re.compile(r'sampling time: ([\d.]+)s, updating indptr time: ([\d.]+)s, updating stable flag time: ([\d.]+)s')
estimated_prep_times = re.compile(r'estimated initial to_dgl_blocks time: ([\d.]+)s, prepare_input time: ([\d.]+)s, mailbox prep time: ([\d.]+)s, post to_dgl_blocks time: ([\d.]+)s, mailbox update time: ([\d.]+)s, update_memory_and_check_stablizing time: ([\d.]+)s, updating indptr and stable flag time: ([\d.]+)s, get_stable_flag time: ([\d.]+)s, edge_feat_index time: ([\d.]+)s')
captured_prep_time_re = re.compile(r'captured prep time: ([\d.]+)s, time_prep: ([\d.]+)s, unaccounted prep time: ([\d.]+)s \(([\d.]+)%\)')
# GIL-release pipeline telemetry (train_simple_v2.py / prefetch_pipeline.py)
prefetch_producer_re = re.compile(r'\[prefetch\] producer: sampling ([\d.]+)s, to_dgl_blocks ([\d.]+)s, prepare_input ([\d.]+)s, queue_put_wait ([\d.]+)s, batches_produced (\d+)')
prefetch_consumer_re = re.compile(r'\[prefetch\] consumer: queue_wait ([\d.]+)s')

rows = []
flip_dist_rows = []
batch_node_rows = []
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
            current = {'epoch': m.group(1), 'scenario': scenario}
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
                flip_dist_rows.append({'epoch': epoch, 'scenario': scenario, 'batch': batch_idx, 'flip_ratio': val})
            continue
        m = batch_node_re.search(line)
        if m and current is not None:
            epoch = current.get('epoch', '')
            batch_node_rows.append({'epoch': epoch, 'scenario': scenario, 'batch': m.group(1),
                                     'batch_size': m.group(2),
                                     'unique_pos_nodes': m.group(3)})
            continue
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
            current.update({
                'mailbox_index_time': m.group(1),
                'mailbox_up_index_time': m.group(2),
                'mailbox_up_dedup_time': m.group(3),
                'mailbox_up_write_time': m.group(4),
                'mailbox_up_cpu_time': m.group(5),
                'mailbox_up_cuda_time': m.group(6),
                'mem_stab_prep_time': m.group(7),
                'mem_stab_math_time': m.group(8),
                'mem_stab_write_time': m.group(9)
            })
            continue
        m = best_re.search(line)
        if m:
            rows.append({'epoch': 'best', 'scenario': scenario, 'val_ap': m.group(2), 'val_auc': m.group(3),
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
                'estimated_update_memory_and_check_stablizing': m.group(6),
                'estimated_updating_indptr_and_stable_flag': m.group(7),
                'estimated_get_stable_flag': m.group(8),
                'estimated_edge_feat_index': m.group(9),
            })
            continue
        m = captured_prep_time_re.search(line)
        if m and current is not None:
            current.update({
                'captured_prep_time': m.group(1),
                'unaccounted_prep_time': m.group(3),
                'unaccounted_prep_pct': m.group(4),
            })
            continue
        m = prefetch_producer_re.search(line)
        if m and current is not None:
            current.update({
                'prefetch_producer_sampling_time': m.group(1),
                'prefetch_producer_to_dgl_blocks_time': m.group(2),
                'prefetch_producer_prepare_input_time': m.group(3),
                'prefetch_producer_queue_put_wait_time': m.group(4),
                'prefetch_producer_batches_produced': m.group(5),
            })
            continue
        m = prefetch_consumer_re.search(line)
        if m and current is not None:
            current.update({'prefetch_consumer_queue_wait_time': m.group(1)})
            continue

flush(current, rows)

if not rows:
    print("No metrics found in log — CSV not written.")
    sys.exit(0)

fieldnames = ['epoch', 'scenario', 'train_loss', 'val_ap', 'val_auc',
              'time_total', 'time_sample', 'time_prep', 'time_model',
              'create_block_time', 'cuda_copy_time', 'total_to_dgl_blocks_time', 'combine_first_time', 'node_index_time', 'edge_index_time', 'node_cuda_time', 'edge_cuda_time',
              'create_dgl_block_time', 'src_id_time', 'edge_dt_time', 'src_ts_time', 'id_time',
              'mailbox_index_time', 'mailbox_up_index_time', 'mailbox_up_dedup_time', 'mailbox_up_write_time',
              'mailbox_up_cpu_time', 'mailbox_up_cuda_time',
              'mem_stab_prep_time', 'mem_stab_math_time', 'mem_stab_write_time',
              'flip_mean', 'flip_std', 'flip_min', 'flip_max', 'flip_batches',
              'best_epoch', 'sampling_time', 'updating_indptr_time', 'updating_stable_flag_time',
              'estimated_prep_to_dgl_blocks', 'estimated_prepare_input', 'estimated_mailbox_prep', 'estimated_post_to_dgl_blocks', 'estimated_mailbox_update', 'estimated_update_memory_and_check_stablizing', 'estimated_updating_indptr_and_stable_flag', 'estimated_get_stable_flag', 'estimated_edge_feat_index',
              'captured_prep_time', 'unaccounted_prep_time', 'unaccounted_prep_pct',
              'prefetch_producer_sampling_time', 'prefetch_producer_to_dgl_blocks_time', 'prefetch_producer_prepare_input_time', 'prefetch_producer_queue_put_wait_time', 'prefetch_producer_batches_produced',
              'prefetch_consumer_queue_wait_time']
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
print(f"Metrics saved to {csv_path} ({len(rows)} rows)")

if flip_dist_rows:
    with open(flip_dist_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'scenario', 'batch', 'flip_ratio'])
        w.writeheader()
        w.writerows(flip_dist_rows)
    print(f"Flip distribution saved to {flip_dist_path} ({len(flip_dist_rows)} rows)")

if batch_node_rows:
    with open(batch_node_stats_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'scenario', 'batch', 'batch_size', 'unique_pos_nodes'])
        w.writeheader()
        w.writerows(batch_node_rows)
    print(f"Batch node stats saved to {batch_node_stats_path} ({len(batch_node_rows)} rows)")
PYEOF

done

# ---------------------------------------------------------------------------
# Summary (only meaningful with more than one scenario, e.g. --scenario all)
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Summary — $RUN_TAG"
echo "========================================"
printf "%-12s %-8s %-12s %-12s %-10s\n" "scenario" "status" "best_ap" "best_auc" "total_time_s"
for scen in "${SCENARIOS_TO_RUN[@]}"; do
    LOG_FILE="${RESULT_LOG[$scen]}"
    STATUS="${RESULT_STATUS[$scen]:-skipped}"
    if [ "$STATUS" != "0" ]; then
        printf "%-12s %-8s %-12s %-12s %-10s\n" "$scen" "FAILED" "-" "-" "-"
        continue
    fi
    BEST_LINE=$(grep -o "Best epoch:[0-9]*  Best AP:[0-9.]*  Best AUC:[0-9.]*" "$LOG_FILE" | tail -1)
    BEST_AP=$(echo "$BEST_LINE" | grep -o "Best AP:[0-9.]*" | cut -d: -f2)
    BEST_AUC=$(echo "$BEST_LINE" | grep -o "Best AUC:[0-9.]*" | cut -d: -f2)
    TOTAL_TIME=$(grep -o "Total training time:[0-9.]*s" "$LOG_FILE" | tail -1 | grep -o "[0-9.]*")
    printf "%-12s %-8s %-12s %-12s %-10s\n" "$scen" "OK" "${BEST_AP:--}" "${BEST_AUC:--}" "${TOTAL_TIME:--}"
done

if [ ${#FAILED[@]} -ne 0 ]; then
    echo ""
    echo "Failed scenarios: ${FAILED[*]}"
    exit 1
fi
