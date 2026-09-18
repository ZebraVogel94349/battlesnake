#!/usr/bin/env bash
set -euo pipefail

# Short, PPO-only best-response continuation against the deployed PPO4 actor.
# Outputs live in an isolated namespace and never overwrite an existing model.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SOURCE="${PPO5_BR_SOURCE:-$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v25_recovery_variants/ppo_bs_lstm_cuda_v25_recovery_steps_014974976000_u00075250.pt}"
OUTPUT_ROOT="${PPO5_BR_OUTPUT_ROOT:-$SCRIPT_DIR}"
UPDATES="${PPO5_BR_UPDATES:-120}"
START_UPDATE="${PPO5_BR_START_UPDATE:-75250}"
STOP_UPDATE=$((START_UPDATE + UPDATES))
RUN_NAME="${PPO5_BR_RUN_NAME:-ppo_bs_lstm_cuda_v26_ppo4_best_response}"
POOL_DIR="$OUTPUT_ROOT/models/selfplay_pool_cuda_v26_ppo4_best_response"
VARIANT_DIR="$OUTPUT_ROOT/models/${RUN_NAME}_variants"
RUN_DIR="$OUTPUT_ROOT/runs/$RUN_NAME"
LATEST="$OUTPUT_ROOT/models/${RUN_NAME}_latest.pt"
FINAL_ZIP="$OUTPUT_ROOT/${RUN_NAME}.zip"

if [[ ! -f "$SOURCE" ]]; then
    echo "Missing PPO5 best-response source: $SOURCE" >&2
    exit 2
fi
if [[ ! "$UPDATES" =~ ^[1-9][0-9]*$ ]]; then
    echo "PPO5_BR_UPDATES must be a positive integer" >&2
    exit 2
fi

if [[ "${1:-}" == "--check" ]]; then
    "$PYTHON" - "$SOURCE" "$START_UPDATE" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
actual = int(payload.get("update", -1))
expected = int(sys.argv[2])
if actual != expected:
    raise SystemExit(f"source update is {actual}, expected {expected}")
if not isinstance(payload.get("model_state_dict"), dict):
    raise SystemExit("source checkpoint has no model state")
print(f"validated PPO4 source update={actual}")
PY
    echo "updates: $START_UPDATE -> $STOP_UPDATE"
    echo "variants: $VARIANT_DIR"
    exit 0
fi

if [[ -e "$LATEST" || -e "$FINAL_ZIP" || -d "$POOL_DIR" || -d "$VARIANT_DIR" ]]; then
    echo "Best-response output already exists; refusing to overwrite it." >&2
    echo "Set PPO5_BR_OUTPUT_ROOT or PPO5_BR_RUN_NAME for a fresh run." >&2
    exit 2
fi
mkdir -p "$POOL_DIR" "$VARIANT_DIR" "$RUN_DIR"

exec "$PYTHON" train_cuda.py \
    --num-envs 768 \
    --n-steps 128 \
    --seq-len 128 \
    --batch-size 8192 \
    --epochs 3 \
    --updates "$UPDATES" \
    --stop-after-update "$STOP_UPDATE" \
    --schedule-origin-update "$START_UPDATE" \
    --rollout-obs-dtype float16 \
    --amp-dtype float16 \
    --channels-last \
    --compile-training-cnn \
    --compile-rollout-ops \
    --compile-mode default \
    --tensorboard-log-dir "$RUN_DIR" \
    --tensorboard-update-interval 10 \
    --timing-log-interval 10 \
    --lr 1.0e-6 \
    --continuation-lr 1.0e-6 \
    --lr-schedule cosine \
    --lr-floor 3.0e-7 \
    --lr-decay-updates "$UPDATES" \
    --target-kl 0.0025 \
    --kl-stop-mode minibatch \
    --gamma 1.0 \
    --gae-lambda 0.97 \
    --clip-range 0.10 \
    --ent-coef 0.00003 \
    --ent-coef-final 0.00001 \
    --ent-decay-updates "$UPDATES" \
    --actor-reference-path "$SOURCE" \
    --actor-reference-l2-coef 0.015 \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --actor-max-grad-norm 0.28 \
    --critic-max-grad-norm 0.5 \
    --potential-length-coef 0.05 \
    --potential-health-coef 0.02 \
    --potential-mobility-coef 0.03 \
    --reward-scheme tournament \
    --agent-action-mask observable \
    --opponent-mode selfplay \
    --training-duel-probability 0.10 \
    --selfplay-update-interval 40 \
    --pool-dir "$POOL_DIR" \
    --max-pool-checkpoints 8 \
    --pool-active-checkpoints 4 \
    --pool-active-rotation-interval 40 \
    --pool-checkpoint-weight 1.0 \
    --pool-best-weight 0.0 \
    --pool-hungry-weight 0.0 \
    --pool-random-weight 0.0 \
    --pool-latest-probability 0.10 \
    --pool-anchor-probability 0.0 \
    --pool-nash-history-size 8 \
    --pool-nash-iterations 2000 \
    --pool-nash-exploration 0.05 \
    --pool-champion-min-weight 0.80 \
    --pool-min-action-disagreement 0.015 \
    --pool-diversity-probe-size 2048 \
    --pool-deterministic-probability 0.90 \
    --pool-copies-probability 0.90 \
    --no-pool-load-existing \
    --eval-interval 40 \
    --eval-games 256 \
    --eval-num-envs 256 \
    --eval-max-turns 2000 \
    --eval-seed 526001 \
    --eval-seed-stride 0 \
    --eval-layout copies \
    --eval-opponents 0 \
    --league-champion-path "$OUTPUT_ROOT/models/${RUN_NAME}_champion.pt" \
    --league-champion-zip-path "$OUTPUT_ROOT/${RUN_NAME}_champion.zip" \
    --league-initial-champion-path "$SOURCE" \
    --league-promotion-games 512 \
    --league-promotion-seed 526101 \
    --league-promotion-seed 526909 \
    --league-promotion-layout copies \
    --league-promotion-layout solo-pair \
    --league-promotion-threshold 0.525 \
    --league-min-promotion-interval 40 \
    --sb3-checkpoint-interval 40 \
    --sb3-checkpoint-dir "$VARIANT_DIR" \
    --sb3-checkpoint-prefix "$RUN_NAME" \
    --max-sb3-checkpoints 6 \
    --fused-adam \
    --allow-tf32 \
    --seed 526025 \
    --save-path "$LATEST" \
    --sb3-save-path "$FINAL_ZIP" \
    --warm-start-path "$SOURCE" \
    "$@"
