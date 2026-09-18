#!/usr/bin/env bash
set -euo pipefail

# Conservative continuation from the strongest protected V21 champion.  The
# complete update 87,500 remains a permanent retention opponent.  V24 fixes
# the training/evaluation mismatch by sampling one Nash/PFSP policy per game
# and copying it into the opponent seats for most four-player episodes.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PYTHON" - "$PROJECT_DIR" <<'PY'
import sys
from pathlib import Path

project = Path(sys.argv[1]).resolve()
try:
    import hisss
except Exception as exc:
    raise SystemExit(f"cannot import local hisss: {exc}") from exc
module = Path(hisss.__file__).resolve()
try:
    module.relative_to(project / "src")
except ValueError as exc:
    raise SystemExit(
        f"wrong hisss package: {module}; expected source below {project / 'src'}"
    ) from exc
if not getattr(hisss, "cuda_available", lambda: False)():
    error = getattr(hisss, "cuda_last_error", lambda: "CUDA backend missing")()
    raise SystemExit(f"local hisss has no usable CUDA backend: {error}")
print(f"[v24] local CUDA hisss: {module}")
PY

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    exec "$PYTHON" train_cuda.py --help
fi

GPU_NAME="$("$PYTHON" -c \
    'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"
if [[ -z "${V24_SPEED_PROFILE:-}" ]]; then
    if [[ "$GPU_NAME" == *"RTX 4090"* ]]; then
        V24_SPEED_PROFILE=rtx4090
    else
        V24_SPEED_PROFILE=standard
    fi
fi

case "$V24_SPEED_PROFILE" in
    standard)
        DEFAULT_NUM_ENVS=1024
        DEFAULT_BATCH_SIZE=16384
        COMPILE_MODE=default
        PARALLEL_INFERENCE_ARGS=()
        ;;
    rtx4090)
        DEFAULT_NUM_ENVS=2048
        DEFAULT_BATCH_SIZE=32768
        COMPILE_MODE=reduce-overhead
        PARALLEL_INFERENCE_ARGS=(--pool-parallel-inference)
        ;;
    *)
        echo "Unknown V24_SPEED_PROFILE: $V24_SPEED_PROFILE" >&2
        exit 2
        ;;
esac

V24_NUM_ENVS="${V24_NUM_ENVS:-$DEFAULT_NUM_ENVS}"
V24_BATCH_SIZE="${V24_BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
V24_EPOCHS="${V24_EPOCHS:-3}"
V24_UPDATES="${V24_UPDATES:-20000}"
V24_START_UPDATE="${V24_START_UPDATE:-51250}"
V24_RETENTION_UPDATE="${V24_RETENTION_UPDATE:-87500}"
V24_STOP_AFTER_UPDATE=$((V24_START_UPDATE + V24_UPDATES))
V24_PEAK_LR="${V24_PEAK_LR:-3e-6}"
V24_LR_FLOOR="${V24_LR_FLOOR:-1e-6}"
V24_EVAL_GAMES="${V24_EVAL_GAMES:-1024}"
V24_EVAL_NUM_ENVS="${V24_EVAL_NUM_ENVS:-256}"

V21_CHAMPION="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v21_balanced_duels_champion.pt"
V24_START_CHECKPOINT="${V24_START_CHECKPOINT:-$V21_CHAMPION}"
V24_RETENTION_CHECKPOINT="${V24_RETENTION_CHECKPOINT:-$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v21_balanced_duels_variants/ppo_bs_lstm_cuda_v21_balanced_duels_steps_019300352000_u00087500.pt}"
V18_FINAL="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v18_v16_extension_champion.pt"
V19_FINAL="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v19_real_anchors_champion.pt"
V20_FINAL="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v20_snake25_anchors_variants/ppo_bs_lstm_cuda_v20_snake25_anchors_steps_017104896000_u00078500.pt"
SNAKE25_BC="$SCRIPT_DIR/models/snake25_bc_v21_u00078500.pt"
V21_POOL="$SCRIPT_DIR/models/selfplay_pool_cuda_v21_balanced_duels"
SNAP_83901="$V21_POOL/snap_018356895744_u00083901.pt"
SNAP_84901="$V21_POOL/snap_018619039744_u00084901.pt"
SNAP_86901="$V21_POOL/snap_019143327744_u00086901.pt"
SNAP_87451="$V21_POOL/snap_019287506944_u00087451.pt"

for required in \
    "$V24_START_CHECKPOINT" \
    "$V24_RETENTION_CHECKPOINT" \
    "$V21_CHAMPION" \
    "$V18_FINAL" \
    "$V19_FINAL" \
    "$V20_FINAL" \
    "$SNAKE25_BC" \
    "$SNAP_83901" \
    "$SNAP_84901" \
    "$SNAP_86901" \
    "$SNAP_87451"; do
    if [[ ! -f "$required" ]]; then
        echo "Required V24 input not found: $required" >&2
        exit 2
    fi
done

"$PYTHON" - \
    "$V24_START_CHECKPOINT" "$V24_START_UPDATE" \
    "$V24_RETENTION_CHECKPOINT" "$V24_RETENTION_UPDATE" <<'PY'
import sys
import torch

start_path, start_expected_raw, retention_path, retention_expected_raw = sys.argv[1:]
start_expected = int(start_expected_raw)
retention_expected = int(retention_expected_raw)
start = torch.load(start_path, map_location="cpu", weights_only=False)
retention = torch.load(retention_path, map_location="cpu", weights_only=False)
if int(start.get("update", -1)) != start_expected:
    raise SystemExit(
        f"V24 start update is {start.get('update')}, expected {start_expected}"
    )
if not isinstance(start.get("model_state_dict"), dict):
    raise SystemExit("V24 start checkpoint has no model state")
if int(retention.get("update", -1)) != retention_expected:
    raise SystemExit(
        f"V24 retention update is {retention.get('update')}, "
        f"expected {retention_expected}"
    )
if not isinstance(retention.get("model_state_dict"), dict):
    raise SystemExit("V24 retention checkpoint has no model state")
optimizer = retention.get("optimizer_state_dict")
if not isinstance(optimizer, dict) or not optimizer.get("state"):
    raise SystemExit("V24 retention checkpoint has no complete optimizer state")
print(
    f"[v24] validated champion start update={start_expected} and complete "
    f"retention update={retention_expected}"
)
PY

V24_OUTPUT_ROOT="${V24_OUTPUT_ROOT:-$SCRIPT_DIR}"
V24_POOL_DIR="$V24_OUTPUT_ROOT/models/selfplay_pool_cuda_v24_general_strength"
V24_VARIANT_DIR="$V24_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v24_general_strength_variants"
V24_LEAGUE_CHAMPION="$V24_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v24_general_strength_champion.pt"
V24_CHAMPION_ZIP="$V24_OUTPUT_ROOT/ppo_bs_lstm_cuda_v24_general_strength_champion.zip"
V24_LATEST="$V24_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v24_general_strength_latest.pt"
V24_FINAL_ZIP="$V24_OUTPUT_ROOT/ppo_bs_lstm_cuda_v24_general_strength.zip"
V24_RUN_DIR="$V24_OUTPUT_ROOT/runs/ppo_bs_lstm_cuda_v24_general_strength"

RESUME_PATH="${V24_RESUME_PATH:-}"
if [[ -z "$RESUME_PATH" ]]; then
    shopt -s nullglob
    CHECKPOINTS=(
        "$V24_VARIANT_DIR"/ppo_bs_lstm_cuda_v24_general_strength_steps_*_u*.pt
    )
    shopt -u nullglob
    if [[ ${#CHECKPOINTS[@]} -gt 0 ]]; then
        RESUME_PATH="${CHECKPOINTS[-1]}"
    fi
fi

if [[ -n "$RESUME_PATH" ]]; then
    if [[ ! -f "$RESUME_PATH" || ! -d "$V24_POOL_DIR" ]]; then
        echo "V24 resume checkpoint or population directory is missing." >&2
        exit 2
    fi
    START_ARGS=(--resume-path "$RESUME_PATH")
    echo "[v24] resuming V24: $RESUME_PATH"
else
    shopt -s nullglob
    EXISTING_V24=(
        "$V24_POOL_DIR"/*
        "$V24_VARIANT_DIR"/*
        "$V24_RUN_DIR"/*
    )
    shopt -u nullglob
    if [[ ${#EXISTING_V24[@]} -gt 0 || -e "$V24_LATEST" || -e "$V24_LEAGUE_CHAMPION" ]]; then
        echo "Partial V24 state exists but no resumable checkpoint was found." >&2
        echo "Nothing was overwritten; inspect $V24_OUTPUT_ROOT before retrying." >&2
        exit 2
    fi
    mkdir -p "$V24_POOL_DIR" "$V24_VARIANT_DIR" "$V24_RUN_DIR"
    # The stronger champion has no matching full optimizer payload.  A fresh
    # optimizer is therefore necessary, but V24 uses one fifth of V23's
    # effective starting LR instead of raising it above the prior run.
    START_ARGS=(--warm-start-path "$V24_START_CHECKPOINT")
    echo "[v24] branching from the protected V21 champion; update 87,500 is retained as an anchor"
fi

echo "[v24] general-strength profile=$V24_SPEED_PROFILE gpu=${GPU_NAME:-unknown} " \
     "envs=$V24_NUM_ENVS batch=$V24_BATCH_SIZE epochs=$V24_EPOCHS " \
     "phase_updates=$V24_UPDATES copies=0.85"

if [[ "${1:-}" == "--check" ]]; then
    echo "[v24] start mode: ${START_ARGS[*]}"
    echo "[v24] phase schedule: update $V24_START_UPDATE -> $V24_STOP_AFTER_UPDATE"
    echo "[v24] outputs: $V24_OUTPUT_ROOT"
    exit 0
fi

exec "$PYTHON" train_cuda.py \
    --num-envs "$V24_NUM_ENVS" \
    --n-steps 128 \
    --seq-len 128 \
    --batch-size "$V24_BATCH_SIZE" \
    --epochs "$V24_EPOCHS" \
    --updates "$V24_UPDATES" \
    --stop-after-update "$V24_STOP_AFTER_UPDATE" \
    --schedule-origin-update "$V24_START_UPDATE" \
    --rollout-obs-dtype float16 \
    --amp-dtype float16 \
    --channels-last \
    --compile-training-cnn \
    --compile-rollout-ops \
    --compile-mode "$COMPILE_MODE" \
    --tensorboard-log-dir "$V24_RUN_DIR" \
    --tensorboard-update-interval 10 \
    --lr "$V24_PEAK_LR" \
    --continuation-lr "$V24_PEAK_LR" \
    --lr-schedule cosine \
    --lr-floor "$V24_LR_FLOOR" \
    --lr-decay-updates "$V24_UPDATES" \
    --target-kl 0.0020 \
    --kl-stop-mode minibatch \
    --gamma 1.0 \
    --gae-lambda 0.97 \
    --clip-range 0.08 \
    --ent-coef 0.0003 \
    --ent-coef-final 0.00005 \
    --ent-decay-updates "$V24_UPDATES" \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --actor-max-grad-norm 0.25 \
    --critic-max-grad-norm 0.5 \
    --potential-length-coef 0.05 \
    --potential-health-coef 0.02 \
    --potential-mobility-coef 0.03 \
    --reward-scheme tournament \
    --agent-action-mask observable \
    --opponent-mode selfplay \
    --training-duel-probability 0.10 \
    --selfplay-update-interval 100 \
    --pool-dir "$V24_POOL_DIR" \
    --pool-anchor "source_87500=$V24_RETENTION_CHECKPOINT" \
    --pool-anchor "v18_final=$V18_FINAL" \
    --pool-anchor "v19_final=$V19_FINAL" \
    --pool-anchor "v20_final=$V20_FINAL" \
    --pool-anchor "snake25_bc=$SNAKE25_BC" \
    --pool-anchor "v21_snap_83901=$SNAP_83901" \
    --pool-anchor "v21_snap_84901=$SNAP_84901" \
    --pool-anchor "v21_snap_86901=$SNAP_86901" \
    --pool-anchor "v21_snap_87451=$SNAP_87451" \
    --pool-anchor-target source_87500=0.55 \
    --pool-anchor-target v18_final=0.50 \
    --pool-anchor-target v19_final=0.50 \
    --pool-anchor-target v20_final=0.52 \
    --pool-anchor-target snake25_bc=0.62 \
    --pool-anchor-target v21_snap_83901=0.52 \
    --pool-anchor-target v21_snap_84901=0.52 \
    --pool-anchor-target v21_snap_86901=0.52 \
    --pool-anchor-target v21_snap_87451=0.52 \
    --pool-anchor-priority-exponent 1.25 \
    --pool-anchor-min-weight 0.10 \
    --pool-anchor-score-ema 0.35 \
    --max-pool-checkpoints 80 \
    --pool-active-checkpoints 11 \
    --pool-active-rotation-interval 50 \
    --pool-checkpoint-weight 0.97 \
    --pool-best-weight 0.0 \
    --pool-hungry-weight 0.0 \
    --pool-random-weight 0.0 \
    --pool-heuristic-weight hunter=0.010 \
    --pool-heuristic-weight snake25=0.010 \
    --pool-heuristic-weight snake25_duelist=0.010 \
    --pool-duel-opponent-weight snake25_duelist=0.20 \
    --pool-duel-opponent-weight snake25_bc=0.20 \
    --pool-anchor-uniform-floor 0.32 \
    --pool-latest-probability 0.10 \
    --pool-anchor-probability 0.80 \
    --pool-nash-history-size 16 \
    --pool-nash-iterations 4000 \
    --pool-nash-exploration 0.10 \
    --pool-nash-score-half-life-updates 1000 \
    --pool-champion-min-weight 0.20 \
    --pool-min-action-disagreement 0.04 \
    --pool-diversity-probe-size 1024 \
    --pool-deterministic-probability 0.75 \
    --pool-copies-probability 0.85 \
    "${PARALLEL_INFERENCE_ARGS[@]}" \
    --pool-load-existing \
    --eval-interval 250 \
    --eval-games "$V24_EVAL_GAMES" \
    --eval-num-envs "$V24_EVAL_NUM_ENVS" \
    --eval-max-turns 2000 \
    --eval-seed 241020 \
    --eval-seed-stride 0 \
    --eval-layout copies \
    --eval-opponents 4 \
    --league-champion-path "$V24_LEAGUE_CHAMPION" \
    --league-champion-zip-path "$V24_CHAMPION_ZIP" \
    --league-initial-champion-path "$V21_CHAMPION" \
    --league-promotion-games 1024 \
    --league-promotion-seed 241123 \
    --league-promotion-seed 241888 \
    --league-promotion-seed 835242 \
    --league-promotion-layout copies \
    --league-promotion-layout solo-pair \
    --league-promotion-layout true-duel \
    --league-promotion-threshold 0.510 \
    --league-general-improvement 0.005 \
    --league-max-anchor-regression 0.025 \
    --league-min-promotion-interval 500 \
    --league-guard-score source_87500=0.48 \
    --league-guard-score v18_final=0.39 \
    --league-guard-score v19_final=0.40 \
    --league-guard-score v20_final=0.46 \
    --league-guard-score snake25_bc=0.58 \
    --league-guard-score v21_snap_83901=0.46 \
    --league-guard-score v21_snap_84901=0.46 \
    --league-guard-score v21_snap_86901=0.45 \
    --league-guard-score v21_snap_87451=0.48 \
    --league-guard-score hunter=0.58 \
    --league-guard-score snake25=0.68 \
    --league-guard-score snake25_duelist=0.65 \
    --sb3-checkpoint-interval 100 \
    --sb3-checkpoint-dir "$V24_VARIANT_DIR" \
    --sb3-checkpoint-prefix ppo_bs_lstm_cuda_v24_general_strength \
    --max-sb3-checkpoints 40 \
    --fused-adam \
    --allow-tf32 \
    --seed 242121 \
    --save-path "$V24_LATEST" \
    --sb3-save-path "$V24_FINAL_ZIP" \
    "${START_ARGS[@]}" \
    "$@"
