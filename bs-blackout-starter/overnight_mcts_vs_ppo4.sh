#!/usr/bin/env bash
# Run as many paired PPO-MCTS-vs-PPO4 evaluations as practical for one night.
#
# Usage:
#   ./overnight_mcts_vs_ppo4.sh
#
# Common overrides:
#   DURATION_HOURS=10 WORKERS=8 BATCH_PAIRS=8 ./overnight_mcts_vs_ppo4.sh
#   CANDIDATE_MODEL=/path/to/mcts.zip BASELINE_MODEL=/path/to/ppo4.zip \
#     ./overnight_mcts_vs_ppo4.sh
#
# The runner starts only complete batches.  A batch that starts shortly before
# the deadline is allowed to finish, so the total runtime can exceed the target
# by up to one batch.  Every completed batch is a standalone report; Ctrl-C or
# a machine failure therefore loses at most the currently running batch.

set -Eeuo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"

DURATION_HOURS="${DURATION_HOURS:-4}"
WORKERS="${WORKERS:-8}"
BATCH_PAIRS="${BATCH_PAIRS:-$WORKERS}"
MAX_TURNS="${MAX_TURNS:-2000}"
START_SEED="${START_SEED:-$(date +%s)}"
TIME_BUDGET_MS="${TIME_BUDGET_MS:-400}"
MIN_ITERATIONS="${MIN_ITERATIONS:-24}"

# These are the current defaults of ppo_mcts.py and ppo4.py.  Explicit paths
# make the comparison immune to unrelated PPO_MODEL_PATH values in the shell.
CANDIDATE_MODEL="${CANDIDATE_MODEL:-${PPO_MCTS_MODEL_PATH:-$SCRIPT_DIR/ppo_bs_lstm_cuda_v25_targeted_finish_champion.zip}}"
BASELINE_MODEL="${BASELINE_MODEL:-${PPO_MODEL_PATH:-$SCRIPT_DIR/ppo_bs_lstm_cuda_v25_targeted_finish_champion.zip}}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/eval_results}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/overnight_mcts_vs_ppo4_$timestamp}"
LOG_FILE="$RUN_DIR/run.log"
FINAL_REPORT="$RUN_DIR/combined.json"

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        printf 'Fehler: %s muss eine positive Ganzzahl sein (war: %s).\n' "$name" "$value" >&2
        exit 2
    fi
}

require_nonnegative_number() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
        printf 'Fehler: %s muss eine nichtnegative Zahl sein (war: %s).\n' "$name" "$value" >&2
        exit 2
    fi
}

require_positive_integer WORKERS "$WORKERS"
require_positive_integer BATCH_PAIRS "$BATCH_PAIRS"
require_positive_integer MAX_TURNS "$MAX_TURNS"
require_positive_integer START_SEED "$START_SEED"
require_positive_integer MIN_ITERATIONS "$MIN_ITERATIONS"
require_nonnegative_number DURATION_HOURS "$DURATION_HOURS"
require_nonnegative_number TIME_BUDGET_MS "$TIME_BUDGET_MS"

if [[ ! -x "$PYTHON" ]]; then
    printf 'Fehler: Python-Interpreter nicht gefunden oder nicht ausfuehrbar: %s\n' "$PYTHON" >&2
    exit 2
fi
for required_file in \
    "$SCRIPT_DIR/evaluate_inference.py" \
    "$SCRIPT_DIR/combine_inference_reports.py" \
    "$CANDIDATE_MODEL" \
    "$BASELINE_MODEL"; do
    if [[ ! -f "$required_file" ]]; then
        printf 'Fehler: Datei nicht gefunden: %s\n' "$required_file" >&2
        exit 2
    fi
done

duration_seconds="$("$PYTHON" -c 'import sys; print(max(1, round(float(sys.argv[1]) * 3600)))' "$DURATION_HOURS")"
require_positive_integer duration_seconds "$duration_seconds"

mkdir -p -- "$OUTPUT_ROOT"
if [[ -e "$RUN_DIR" ]]; then
    printf 'Fehler: Ausgabeordner existiert bereits: %s\n' "$RUN_DIR" >&2
    exit 2
fi
mkdir -- "$RUN_DIR"

# Each evaluator worker already restricts PyTorch to one thread.  Restrict the
# common BLAS backends as well so eight processes do not oversubscribe the CPU.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PPO_DEVICE="${PPO_DEVICE:-cpu}"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

finalize() {
    local original_status="$?"
    local reports=("$RUN_DIR"/batch_*.json)

    if (( ${#reports[@]} >= 2 )); then
        printf '\nKombiniere %d vollstaendige Batch-Reports ...\n' "${#reports[@]}" | tee -a "$LOG_FILE"
        if ! "$PYTHON" "$SCRIPT_DIR/combine_inference_reports.py" \
            "${reports[@]}" \
            --bootstrap-seed "$START_SEED" \
            --output "$FINAL_REPORT" 2>&1 | tee -a "$LOG_FILE"; then
            printf 'Warnung: Kombination fehlgeschlagen; die Batch-Reports bleiben erhalten.\n' | tee -a "$LOG_FILE" >&2
        fi
    elif (( ${#reports[@]} == 1 )); then
        printf '\nEin vollstaendiger Report liegt vor: %s\n' "${reports[0]}" | tee -a "$LOG_FILE"
    else
        printf '\nKein Batch wurde vollstaendig beendet.\n' | tee -a "$LOG_FILE"
    fi

    printf 'Ergebnisse: %s\n' "$RUN_DIR" | tee -a "$LOG_FILE"
    if [[ -f "$FINAL_REPORT" ]]; then
        printf 'Gesamtbericht: %s\n' "$FINAL_REPORT" | tee -a "$LOG_FILE"
    fi
    return "$original_status"
}
trap finalize EXIT

started_at="$(date +%s)"
deadline=$((started_at + duration_seconds))
batch=0

{
    printf 'PPO-MCTS vs PPO4 Overnight-Evaluation\n'
    printf 'Start (UTC):       %s\n' "$timestamp"
    printf 'Zieldauer:         %s Stunden\n' "$DURATION_HOURS"
    printf 'Worker:            %s\n' "$WORKERS"
    printf 'Paare pro Batch:   %s (%s Spiele)\n' "$BATCH_PAIRS" "$((2 * BATCH_PAIRS))"
    printf 'MCTS-Zeitbudget:   %s ms pro Zug\n' "$TIME_BUDGET_MS"
    printf 'MCTS-Minimum:      %s Iterationen\n' "$MIN_ITERATIONS"
    printf 'Kandidat:          %s\n' "$CANDIDATE_MODEL"
    printf 'Baseline:          %s\n' "$BASELINE_MODEL"
    printf 'Ausgabe:           %s\n\n' "$RUN_DIR"
} | tee -a "$LOG_FILE"

while (( $(date +%s) < deadline )); do
    batch_seed=$((START_SEED + batch * BATCH_PAIRS))
    batch_report="$RUN_DIR/batch_$(printf '%04d' "$batch")_seed_${batch_seed}.json"

    printf '\n=== Batch %d | Seed %d | %d Paare ===\n' \
        "$batch" "$batch_seed" "$BATCH_PAIRS" | tee -a "$LOG_FILE"

    "$PYTHON" "$SCRIPT_DIR/evaluate_inference.py" \
        --candidate ppo-mcts \
        --baseline ppo4 \
        --candidate-model "$CANDIDATE_MODEL" \
        --baseline-model "$BASELINE_MODEL" \
        --candidate-time-ms "$TIME_BUDGET_MS" \
        --candidate-min-iterations "$MIN_ITERATIONS" \
        --layout copies \
        --pairs "$BATCH_PAIRS" \
        --workers "$WORKERS" \
        --max-turns "$MAX_TURNS" \
        --seed "$batch_seed" \
        --output "$batch_report" 2>&1 | tee -a "$LOG_FILE"

    batch=$((batch + 1))
done

printf '\nZeitfenster erreicht; es wird kein weiterer Batch gestartet.\n' | tee -a "$LOG_FILE"
