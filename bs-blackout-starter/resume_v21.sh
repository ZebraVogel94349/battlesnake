#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
V21_OUTPUT_ROOT="${V21_OUTPUT_ROOT:-$SCRIPT_DIR}"
VARIANT_DIR="$V21_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v21_balanced_duels_variants"
POOL_STATE="$V21_OUTPUT_ROOT/models/selfplay_pool_cuda_v21_balanced_duels/nash_state.json"

RESUME_PATH="${V21_RESUME_PATH:-}"
if [[ -z "$RESUME_PATH" ]]; then
    shopt -s nullglob
    CHECKPOINTS=(
        "$VARIANT_DIR"/ppo_bs_lstm_cuda_v21_balanced_duels_steps_*_u*.pt
    )
    shopt -u nullglob
    if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
        echo "No resumable V21 checkpoint found in: $VARIANT_DIR" >&2
        exit 2
    fi
    RESUME_PATH="${CHECKPOINTS[-1]}"
fi

if [[ ! -f "$RESUME_PATH" || ! -f "$POOL_STATE" ]]; then
    echo "V21 checkpoint or pool state is missing." >&2
    exit 2
fi

if [[ "${1:-}" == "--check" ]]; then
    echo "[v21] checkpoint: $RESUME_PATH"
    echo "[v21] pool state: $POOL_STATE"
    V21_RESUME_PATH="$RESUME_PATH" "$SCRIPT_DIR/train_v21.sh" --check
    exit 0
fi

V21_RESUME_PATH="$RESUME_PATH" exec "$SCRIPT_DIR/train_v21.sh" "$@"
