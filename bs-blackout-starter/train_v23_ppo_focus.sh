#!/usr/bin/env bash
set -euo pipefail

# PPO-focused continuation from the complete V21 update 87,500. Native
# heuristics are retained only as a small rehearsal set; current Nash/PFSP
# results allocate almost all training mass to hard policies and snapshots.

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
print(f"[v23] local CUDA hisss: {module}")
PY

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    exec "$PYTHON" train_cuda.py --help
fi

GPU_NAME="$("$PYTHON" -c \
    'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"
if [[ -z "${V23_SPEED_PROFILE:-}" ]]; then
    if [[ "$GPU_NAME" == *"RTX 4090"* ]]; then
        V23_SPEED_PROFILE=rtx4090
    else
        V23_SPEED_PROFILE=standard
    fi
fi

case "$V23_SPEED_PROFILE" in
    standard)
        DEFAULT_NUM_ENVS=1024
        DEFAULT_BATCH_SIZE=16384
        DEFAULT_EPOCHS=4
        COMPILE_MODE=default
        PARALLEL_INFERENCE_ARGS=()
        ;;
    rtx4090)
        DEFAULT_NUM_ENVS=2048
        DEFAULT_BATCH_SIZE=32768
        DEFAULT_EPOCHS=3
        COMPILE_MODE=reduce-overhead
        PARALLEL_INFERENCE_ARGS=(--pool-parallel-inference)
        ;;
    *)
        echo "Unknown V23_SPEED_PROFILE: $V23_SPEED_PROFILE" >&2
        exit 2
        ;;
esac

V23_NUM_ENVS="${V23_NUM_ENVS:-$DEFAULT_NUM_ENVS}"
V23_BATCH_SIZE="${V23_BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
V23_EPOCHS="${V23_EPOCHS:-$DEFAULT_EPOCHS}"
V23_ACTIVE_CHECKPOINTS="${V23_ACTIVE_CHECKPOINTS:-11}"
V23_UPDATES="${V23_UPDATES:-20000}"
V23_SOURCE_UPDATE="${V23_SOURCE_UPDATE:-87500}"
V23_STOP_AFTER_UPDATE=$((V23_SOURCE_UPDATE + V23_UPDATES))
V23_PEAK_LR="${V23_PEAK_LR:-1.5e-5}"
V23_LR_FLOOR="${V23_LR_FLOOR:-3e-6}"
V23_EVAL_GAMES="${V23_EVAL_GAMES:-512}"
V23_EVAL_NUM_ENVS="${V23_EVAL_NUM_ENVS:-256}"

echo "[v23] PPO-focus profile=$V23_SPEED_PROFILE gpu=${GPU_NAME:-unknown} " \
     "envs=$V23_NUM_ENVS batch=$V23_BATCH_SIZE epochs=$V23_EPOCHS " \
     "active_models=$V23_ACTIVE_CHECKPOINTS phase_updates=$V23_UPDATES" >&2

V23_SOURCE_CHECKPOINT="${V23_SOURCE_CHECKPOINT:-$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v21_balanced_duels_variants/ppo_bs_lstm_cuda_v21_balanced_duels_steps_019300352000_u00087500.pt}"
OLD_CHAMPION="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v21_balanced_duels_champion.pt"
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
    "$V23_SOURCE_CHECKPOINT" \
    "$OLD_CHAMPION" \
    "$V18_FINAL" \
    "$V19_FINAL" \
    "$V20_FINAL" \
    "$SNAKE25_BC" \
    "$SNAP_83901" \
    "$SNAP_84901" \
    "$SNAP_86901" \
    "$SNAP_87451"; do
    if [[ ! -f "$required" ]]; then
        echo "Required V23 input not found: $required" >&2
        exit 2
    fi
done

"$PYTHON" - "$V23_SOURCE_CHECKPOINT" "$V23_SOURCE_UPDATE" <<'PY'
import sys
import torch

path, expected_raw = sys.argv[1:]
expected = int(expected_raw)
payload = torch.load(path, map_location="cpu", weights_only=False)
if int(payload.get("update", -1)) != expected:
    raise SystemExit(
        f"V23 source update is {payload.get('update')}, expected {expected}"
    )
if not isinstance(payload.get("model_state_dict"), dict):
    raise SystemExit("V23 source has no model state")
print(
    f"[v23] validated full source update={expected} "
    f"steps={int(payload.get('total_steps', 0))}"
)
PY

V23_OUTPUT_ROOT="${V23_OUTPUT_ROOT:-$SCRIPT_DIR}"
V23_POOL_DIR="$V23_OUTPUT_ROOT/models/selfplay_pool_cuda_v23_ppo_focus"
V23_VARIANT_DIR="$V23_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v23_ppo_focus_variants"
V23_LEAGUE_CHAMPION="$V23_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v23_ppo_focus_champion.pt"
V23_CHAMPION_ZIP="$V23_OUTPUT_ROOT/ppo_bs_lstm_cuda_v23_ppo_focus_champion.zip"
V23_LATEST="$V23_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v23_ppo_focus_latest.pt"
V23_FINAL_ZIP="$V23_OUTPUT_ROOT/ppo_bs_lstm_cuda_v23_ppo_focus.zip"
V23_RUN_DIR="$V23_OUTPUT_ROOT/runs/ppo_bs_lstm_cuda_v23_ppo_focus"

RESUME_PATH="${V23_RESUME_PATH:-}"
if [[ -z "$RESUME_PATH" ]]; then
    shopt -s nullglob
    CHECKPOINTS=(
        "$V23_VARIANT_DIR"/ppo_bs_lstm_cuda_v23_ppo_focus_steps_*_u*.pt
    )
    shopt -u nullglob
    if [[ ${#CHECKPOINTS[@]} -gt 0 ]]; then
        RESUME_PATH="${CHECKPOINTS[-1]}"
    fi
fi

START_ARGS=()
if [[ -n "$RESUME_PATH" ]]; then
    if [[ ! -f "$RESUME_PATH" || ! -d "$V23_POOL_DIR" ]]; then
        echo "V23 resume checkpoint or population directory is missing." >&2
        exit 2
    fi
    START_ARGS=(--resume-path "$RESUME_PATH")
    echo "[v23] resuming: $RESUME_PATH"
else
    shopt -s nullglob
    EXISTING_V23=(
        "$V23_POOL_DIR"/*
        "$V23_VARIANT_DIR"/*
        "$V23_RUN_DIR"/*
    )
    shopt -u nullglob
    if [[ ${#EXISTING_V23[@]} -gt 0 || -e "$V23_LATEST" || -e "$V23_LEAGUE_CHAMPION" ]]; then
        echo "Partial V23 state exists but no resumable checkpoint was found." >&2
        echo "Nothing was overwritten; inspect $V23_OUTPUT_ROOT before retrying." >&2
        exit 2
    fi
    mkdir -p "$V23_POOL_DIR" "$V23_VARIANT_DIR" "$V23_RUN_DIR"
    START_ARGS=(--warm-start-path "$V23_SOURCE_CHECKPOINT")
    echo "[v23] fresh optimizer/Nash state from complete update 87,500"
fi

if [[ "${1:-}" == "--check" ]]; then
    echo "[v23] start mode: ${START_ARGS[*]}"
    echo "[v23] phase schedule: update $V23_SOURCE_UPDATE -> $V23_STOP_AFTER_UPDATE"
    echo "[v23] outputs: $V23_OUTPUT_ROOT"
    exit 0
fi

exec "$PYTHON" train_cuda.py \
    --num-envs "$V23_NUM_ENVS" \
    --n-steps 128 \
    --seq-len 128 \
    --batch-size "$V23_BATCH_SIZE" \
    --epochs "$V23_EPOCHS" \
    --updates "$V23_UPDATES" \
    --stop-after-update "$V23_STOP_AFTER_UPDATE" \
    --schedule-origin-update "$V23_SOURCE_UPDATE" \
    --rollout-obs-dtype float16 \
    --amp-dtype float16 \
    --channels-last \
    --compile-training-cnn \
    --compile-rollout-ops \
    --compile-mode "$COMPILE_MODE" \
    --tensorboard-log-dir "$V23_RUN_DIR" \
    --tensorboard-update-interval 10 \
    --lr "$V23_PEAK_LR" \
    --continuation-lr "$V23_PEAK_LR" \
    --lr-schedule cosine \
    --lr-floor "$V23_LR_FLOOR" \
    --lr-decay-updates "$V23_UPDATES" \
    --target-kl 0.006 \
    --kl-stop-mode minibatch \
    --gamma 1.0 \
    --gae-lambda 0.97 \
    --clip-range 0.12 \
    --ent-coef 0.0020 \
    --ent-coef-final 0.0008 \
    --ent-decay-updates "$V23_UPDATES" \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --potential-length-coef 0.05 \
    --potential-health-coef 0.02 \
    --potential-mobility-coef 0.03 \
    --reward-scheme tournament \
    --agent-action-mask observable \
    --opponent-mode selfplay \
    --training-duel-probability 0.10 \
    --selfplay-update-interval 50 \
    --pool-dir "$V23_POOL_DIR" \
    --pool-anchor "old_champion=$OLD_CHAMPION" \
    --pool-anchor "v18_final=$V18_FINAL" \
    --pool-anchor "v19_final=$V19_FINAL" \
    --pool-anchor "v20_final=$V20_FINAL" \
    --pool-anchor "snake25_bc=$SNAKE25_BC" \
    --pool-anchor "v21_snap_83901=$SNAP_83901" \
    --pool-anchor "v21_snap_84901=$SNAP_84901" \
    --pool-anchor "v21_snap_86901=$SNAP_86901" \
    --pool-anchor "v21_snap_87451=$SNAP_87451" \
    --pool-anchor-target old_champion=0.52 \
    --pool-anchor-target v18_final=0.52 \
    --pool-anchor-target v19_final=0.52 \
    --pool-anchor-target v20_final=0.55 \
    --pool-anchor-target snake25_bc=0.65 \
    --pool-anchor-target v21_snap_83901=0.55 \
    --pool-anchor-target v21_snap_84901=0.55 \
    --pool-anchor-target v21_snap_86901=0.55 \
    --pool-anchor-target v21_snap_87451=0.55 \
    --pool-anchor-priority-exponent 1.5 \
    --pool-anchor-min-weight 0.05 \
    --pool-anchor-score-ema 0.50 \
    --max-pool-checkpoints 100 \
    --pool-active-checkpoints "$V23_ACTIVE_CHECKPOINTS" \
    --pool-active-rotation-interval 20 \
    --pool-checkpoint-weight 0.96 \
    --pool-best-weight 0.0 \
    --pool-hungry-weight 0.0 \
    --pool-random-weight 0.0 \
    --pool-heuristic-weight hunter=0.010 \
    --pool-heuristic-weight snake25=0.015 \
    --pool-heuristic-weight snake25_duelist=0.015 \
    --pool-duel-opponent-weight snake25_duelist=0.25 \
    --pool-duel-opponent-weight snake25_bc=0.25 \
    --pool-anchor-uniform-floor 0.0 \
    --pool-latest-probability 0.15 \
    --pool-anchor-probability 0.75 \
    --pool-nash-history-size 24 \
    --pool-nash-iterations 3000 \
    --pool-nash-exploration 0.05 \
    --pool-nash-score-half-life-updates 500 \
    --pool-champion-min-weight 0.15 \
    --pool-min-action-disagreement 0.05 \
    --pool-diversity-probe-size 512 \
    --pool-focus-label v18_final \
    --pool-focus-probability 0.25 \
    --pool-deterministic-probability 0.50 \
    "${PARALLEL_INFERENCE_ARGS[@]}" \
    --pool-load-existing \
    --eval-interval 250 \
    --eval-games "$V23_EVAL_GAMES" \
    --eval-num-envs "$V23_EVAL_NUM_ENVS" \
    --eval-max-turns 2000 \
    --eval-seed 230020 \
    --eval-seed-stride 0 \
    --eval-layout copies \
    --eval-opponents 8 \
    --league-champion-path "$V23_LEAGUE_CHAMPION" \
    --league-champion-zip-path "$V23_CHAMPION_ZIP" \
    --league-initial-champion-path "$V23_SOURCE_CHECKPOINT" \
    --league-promotion-games 512 \
    --league-promotion-seed 230123 \
    --league-promotion-seed 230888 \
    --league-promotion-seed 824242 \
    --league-promotion-layout copies \
    --league-promotion-layout solo-pair \
    --league-promotion-layout true-duel \
    --league-promotion-threshold 0.515 \
    --league-min-promotion-interval 250 \
    --league-guard-score old_champion=0.40 \
    --league-guard-score v18_final=0.35 \
    --league-guard-score v19_final=0.40 \
    --league-guard-score v20_final=0.47 \
    --league-guard-score snake25_bc=0.58 \
    --league-guard-score v21_snap_83901=0.50 \
    --league-guard-score v21_snap_84901=0.51 \
    --league-guard-score v21_snap_86901=0.42 \
    --league-guard-score v21_snap_87451=0.50 \
    --league-guard-score hunter=0.90 \
    --league-guard-score snake25=0.83 \
    --league-guard-score snake25_duelist=0.76 \
    --sb3-checkpoint-interval 100 \
    --sb3-checkpoint-dir "$V23_VARIANT_DIR" \
    --sb3-checkpoint-prefix ppo_bs_lstm_cuda_v23_ppo_focus \
    --max-sb3-checkpoints 32 \
    --fused-adam \
    --allow-tf32 \
    --seed 232121 \
    --save-path "$V23_LATEST" \
    --sb3-save-path "$V23_FINAL_ZIP" \
    "${START_ARGS[@]}" \
    "$@"
