#!/usr/bin/env bash
set -euo pipefail

# Recovery phase for the stalled V21 run.  This deliberately starts from the
# last league champion (the only policy that consistently beat every late V21
# learner), uses a fresh optimizer/Nash state, and writes to isolated V22 paths.

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
    raise SystemExit(
        f"cannot import local hisss ({exc}); run "
        f"{project / 'bs-blackout-starter' / 'install_local_cuda.sh'}"
    ) from exc
module = Path(hisss.__file__).resolve()
try:
    module.relative_to(project / "src")
except ValueError as exc:
    raise SystemExit(
        f"wrong hisss package: {module}\n"
        f"Expected the editable local source below {project / 'src'}."
    ) from exc
if not getattr(hisss, "cuda_available", lambda: False)():
    error = getattr(hisss, "cuda_last_error", lambda: "CUDA backend missing")()
    raise SystemExit(f"local hisss has no usable CUDA backend: {error}")
print(f"[v22] local CUDA hisss: {module}")
PY

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    exec "$PYTHON" train_cuda.py --help
fi

GPU_NAME="$("$PYTHON" -c \
    'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"
if [[ -z "${V22_SPEED_PROFILE:-}" ]]; then
    if [[ "$GPU_NAME" == *"RTX 4090"* ]]; then
        V22_SPEED_PROFILE=rtx4090
    else
        V22_SPEED_PROFILE=standard
    fi
fi

case "$V22_SPEED_PROFILE" in
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
        echo "Unknown V22_SPEED_PROFILE: $V22_SPEED_PROFILE" >&2
        echo "Expected one of: standard, rtx4090" >&2
        exit 2
        ;;
esac

V22_NUM_ENVS="${V22_NUM_ENVS:-$DEFAULT_NUM_ENVS}"
V22_BATCH_SIZE="${V22_BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
V22_EPOCHS="${V22_EPOCHS:-$DEFAULT_EPOCHS}"
V22_ACTIVE_CHECKPOINTS="${V22_ACTIVE_CHECKPOINTS:-6}"
V22_UPDATES="${V22_UPDATES:-30000}"
V22_SOURCE_UPDATE="${V22_SOURCE_UPDATE:-51250}"
V22_STOP_AFTER_UPDATE=$((V22_SOURCE_UPDATE + V22_UPDATES))
V22_PEAK_LR="${V22_PEAK_LR:-2.5e-5}"
V22_LR_FLOOR="${V22_LR_FLOOR:-5e-6}"
V22_EVAL_GAMES="${V22_EVAL_GAMES:-512}"
V22_EVAL_NUM_ENVS="${V22_EVAL_NUM_ENVS:-256}"

echo "[v22] quality-recovery profile=$V22_SPEED_PROFILE gpu=${GPU_NAME:-unknown} " \
     "envs=$V22_NUM_ENVS batch=$V22_BATCH_SIZE epochs=$V22_EPOCHS " \
     "active_models=$V22_ACTIVE_CHECKPOINTS phase_updates=$V22_UPDATES" >&2

ANCHOR_655M="$SCRIPT_DIR/ppo_bs_lstm_cuda_steps_000655360000_u00005000.zip"
ANCHOR_V7_917M="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v7_variants/ppo_bs_lstm_cuda_v7_steps_000917504000_u00007000.zip"
V9_VARIANT_DIR="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v9_variants"
V9_HOF_750="$V9_VARIANT_DIR/ppo_bs_lstm_cuda_v9_steps_000098304000_u00000750.zip"
V9_HOF_2500="$V9_VARIANT_DIR/ppo_bs_lstm_cuda_v9_steps_000327680000_u00002500.zip"
V9_HOF_5500="$V9_VARIANT_DIR/ppo_bs_lstm_cuda_v9_steps_000720896000_u00005500.zip"
V9_LEAGUE_CHAMPION="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v9_league_champion.pt"
V13_LEAGUE_CHAMPION="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v13_league_champion.pt"
V13_SELFPLAY_LAST="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v13_league_variants/ppo_bs_lstm_cuda_v13_league_steps_001441792000_u00011000.pt"
V14_REFERENCE="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v14_league_variants/ppo_bs_lstm_cuda_v14_league_steps_000589824000_u00004500.pt"
V15_FINAL="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v15_nash_latest.pt"
V18_FINAL="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v18_v16_extension_champion.pt"
V19_FINAL="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v19_real_anchors_champion.pt"
V20_FINAL="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v20_snake25_anchors_variants/ppo_bs_lstm_cuda_v20_snake25_anchors_steps_017104896000_u00078500.pt"
SNAKE25_BC="$SCRIPT_DIR/models/snake25_bc_v21_u00078500.pt"

# This checkpoint remained champion throughout the downloaded night.  Late V21
# candidates averaged only 0.465 against it across the nine promotion suites.
V22_SOURCE_CHECKPOINT="${V22_SOURCE_CHECKPOINT:-$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v21_balanced_duels_champion.pt}"

for required in \
    "$ANCHOR_655M" \
    "$ANCHOR_V7_917M" \
    "$V9_HOF_750" \
    "$V9_HOF_2500" \
    "$V9_HOF_5500" \
    "$V9_LEAGUE_CHAMPION" \
    "$V13_LEAGUE_CHAMPION" \
    "$V13_SELFPLAY_LAST" \
    "$V14_REFERENCE" \
    "$V15_FINAL" \
    "$V18_FINAL" \
    "$V19_FINAL" \
    "$V20_FINAL" \
    "$SNAKE25_BC" \
    "$V22_SOURCE_CHECKPOINT"; do
    if [[ ! -f "$required" ]]; then
        echo "Required V22 input not found: $required" >&2
        exit 2
    fi
done

"$PYTHON" - "$V22_SOURCE_CHECKPOINT" "$V22_SOURCE_UPDATE" <<'PY'
import sys
import torch

path, expected_raw = sys.argv[1:]
expected = int(expected_raw)
payload = torch.load(path, map_location="cpu", weights_only=False)
if int(payload.get("update", -1)) != expected:
    raise SystemExit(
        f"V22 source update is {payload.get('update')}, expected {expected}; "
        "set V22_SOURCE_UPDATE explicitly if the source was intentionally changed"
    )
if not isinstance(payload.get("model_state_dict"), dict):
    raise SystemExit("V22 source has no model state")
print(
    f"[v22] validated strongest champion: update={expected} "
    f"steps={int(payload.get('total_steps', 0))}"
)
PY

V22_OUTPUT_ROOT="${V22_OUTPUT_ROOT:-$SCRIPT_DIR}"
V22_POOL_DIR="$V22_OUTPUT_ROOT/models/selfplay_pool_cuda_v22_effective"
V22_VARIANT_DIR="$V22_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v22_effective_variants"
V22_LEAGUE_CHAMPION="$V22_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v22_effective_champion.pt"
V22_CHAMPION_ZIP="$V22_OUTPUT_ROOT/ppo_bs_lstm_cuda_v22_effective_champion.zip"
V22_LATEST="$V22_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v22_effective_latest.pt"
V22_FINAL_ZIP="$V22_OUTPUT_ROOT/ppo_bs_lstm_cuda_v22_effective.zip"
V22_RUN_DIR="$V22_OUTPUT_ROOT/runs/ppo_bs_lstm_cuda_v22_effective"

RESUME_PATH="${V22_RESUME_PATH:-}"
if [[ -z "$RESUME_PATH" ]]; then
    shopt -s nullglob
    CHECKPOINTS=(
        "$V22_VARIANT_DIR"/ppo_bs_lstm_cuda_v22_effective_steps_*_u*.pt
    )
    shopt -u nullglob
    if [[ ${#CHECKPOINTS[@]} -gt 0 ]]; then
        RESUME_PATH="${CHECKPOINTS[-1]}"
    fi
fi

START_ARGS=()
if [[ -n "$RESUME_PATH" ]]; then
    if [[ ! -f "$RESUME_PATH" || ! -d "$V22_POOL_DIR" ]]; then
        echo "V22 resume checkpoint or population directory is missing." >&2
        exit 2
    fi
    START_ARGS=(--resume-path "$RESUME_PATH")
    echo "[v22] resuming: $RESUME_PATH"
else
    shopt -s nullglob
    EXISTING_V22=(
        "$V22_POOL_DIR"/*
        "$V22_VARIANT_DIR"/*
        "$V22_RUN_DIR"/*
    )
    shopt -u nullglob
    if [[ ${#EXISTING_V22[@]} -gt 0 || -e "$V22_LATEST" || -e "$V22_LEAGUE_CHAMPION" ]]; then
        echo "Partial V22 state exists but no resumable checkpoint was found." >&2
        echo "Inspect $V22_OUTPUT_ROOT before retrying; nothing was overwritten." >&2
        exit 2
    fi
    mkdir -p "$V22_POOL_DIR" "$V22_VARIANT_DIR" "$V22_RUN_DIR"
    START_ARGS=(--warm-start-path "$V22_SOURCE_CHECKPOINT")
    echo "[v22] fresh optimizer and fresh Nash population from the validated champion"
fi

if [[ "${1:-}" == "--check" ]]; then
    echo "[v22] start mode: ${START_ARGS[*]}"
    echo "[v22] phase schedule: update $V22_SOURCE_UPDATE -> $V22_STOP_AFTER_UPDATE"
    echo "[v22] outputs: $V22_OUTPUT_ROOT"
    exit 0
fi

exec "$PYTHON" train_cuda.py \
    --num-envs "$V22_NUM_ENVS" \
    --n-steps 128 \
    --seq-len 128 \
    --batch-size "$V22_BATCH_SIZE" \
    --epochs "$V22_EPOCHS" \
    --updates "$V22_UPDATES" \
    --stop-after-update "$V22_STOP_AFTER_UPDATE" \
    --schedule-origin-update "$V22_SOURCE_UPDATE" \
    --rollout-obs-dtype float16 \
    --amp-dtype float16 \
    --channels-last \
    --compile-training-cnn \
    --compile-rollout-ops \
    --compile-mode "$COMPILE_MODE" \
    --tensorboard-log-dir "$V22_RUN_DIR" \
    --tensorboard-update-interval 10 \
    --lr "$V22_PEAK_LR" \
    --continuation-lr "$V22_PEAK_LR" \
    --lr-schedule cosine \
    --lr-floor "$V22_LR_FLOOR" \
    --lr-decay-updates "$V22_UPDATES" \
    --target-kl 0.008 \
    --kl-stop-mode minibatch \
    --gamma 1.0 \
    --gae-lambda 0.97 \
    --clip-range 0.15 \
    --ent-coef 0.0035 \
    --ent-coef-final 0.0015 \
    --ent-decay-updates "$V22_UPDATES" \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --potential-length-coef 0.05 \
    --potential-health-coef 0.02 \
    --potential-mobility-coef 0.03 \
    --reward-scheme tournament \
    --agent-action-mask observable \
    --opponent-mode selfplay \
    --training-duel-probability 0.25 \
    --selfplay-update-interval 50 \
    --pool-dir "$V22_POOL_DIR" \
    --pool-anchor "old_655m=$ANCHOR_655M" \
    --pool-anchor "v7_917m=$ANCHOR_V7_917M" \
    --pool-anchor "v9_hof_750=$V9_HOF_750" \
    --pool-anchor "v9_hof_2500=$V9_HOF_2500" \
    --pool-anchor "v9_hof_5500=$V9_HOF_5500" \
    --pool-anchor "v9_champion=$V9_LEAGUE_CHAMPION" \
    --pool-anchor "v13_champion=$V13_LEAGUE_CHAMPION" \
    --pool-anchor "v13_selfplay_last=$V13_SELFPLAY_LAST" \
    --pool-anchor "v14_reference=$V14_REFERENCE" \
    --pool-anchor "v15_final=$V15_FINAL" \
    --pool-anchor "v18_final=$V18_FINAL" \
    --pool-anchor "v19_final=$V19_FINAL" \
    --pool-anchor "v20_final=$V20_FINAL" \
    --pool-anchor "snake25_bc=$SNAKE25_BC" \
    --max-pool-checkpoints 100 \
    --pool-active-checkpoints "$V22_ACTIVE_CHECKPOINTS" \
    --pool-active-rotation-interval 20 \
    --pool-checkpoint-weight 0.65 \
    --pool-best-weight 0.05 \
    --pool-hungry-weight 0.0 \
    --pool-random-weight 0.01 \
    --pool-heuristic-weight hunter=0.07 \
    --pool-heuristic-weight snake25=0.10 \
    --pool-heuristic-weight snake25_duelist=0.12 \
    --pool-duel-opponent-weight snake25_duelist=0.50 \
    --pool-duel-opponent-weight snake25_bc=0.30 \
    --pool-anchor-uniform-floor 0.0 \
    --pool-latest-probability 0.10 \
    --pool-anchor-probability 0.35 \
    --pool-nash-history-size 24 \
    --pool-nash-iterations 3000 \
    --pool-nash-exploration 0.08 \
    --pool-nash-score-half-life-updates 1000 \
    --pool-champion-min-weight 0.20 \
    --pool-min-action-disagreement 0.08 \
    --pool-diversity-probe-size 512 \
    --pool-focus-label league_champion \
    --pool-focus-probability 0.35 \
    --pool-deterministic-probability 0.50 \
    "${PARALLEL_INFERENCE_ARGS[@]}" \
    --pool-load-existing \
    --eval-interval 250 \
    --eval-games "$V22_EVAL_GAMES" \
    --eval-num-envs "$V22_EVAL_NUM_ENVS" \
    --eval-max-turns 2000 \
    --eval-seed 220020 \
    --eval-seed-stride 1000003 \
    --eval-layout copies \
    --eval-opponents 6 \
    --league-champion-path "$V22_LEAGUE_CHAMPION" \
    --league-champion-zip-path "$V22_CHAMPION_ZIP" \
    --league-initial-champion-path "$V22_SOURCE_CHECKPOINT" \
    --league-promotion-games 512 \
    --league-promotion-seed 220123 \
    --league-promotion-seed 220888 \
    --league-promotion-seed 824242 \
    --league-promotion-layout copies \
    --league-promotion-layout solo-pair \
    --league-promotion-layout true-duel \
    --league-promotion-threshold 0.525 \
    --league-min-promotion-interval 250 \
    --sb3-checkpoint-interval 100 \
    --sb3-checkpoint-dir "$V22_VARIANT_DIR" \
    --sb3-checkpoint-prefix ppo_bs_lstm_cuda_v22_effective \
    --max-sb3-checkpoints 32 \
    --fused-adam \
    --allow-tf32 \
    --seed 222121 \
    --save-path "$V22_LATEST" \
    --sb3-save-path "$V22_FINAL_ZIP" \
    "${START_ARGS[@]}" \
    "$@"
