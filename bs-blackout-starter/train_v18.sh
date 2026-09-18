#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON" >&2
    echo "Set PYTHON=/path/to/python or create $SCRIPT_DIR/.venv first." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# The 5090 has enough compute and VRAM to use a wider rollout and PPO batch.
# Auto-select that profile only for the exact device class; every other GPU
# keeps the established ``fast`` behavior unless explicitly overridden.
if [[ -z "${V18_SPEED_PROFILE:-}" ]]; then
    GPU_NAME="$("$PYTHON" -c \
        'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"
    if [[ "$GPU_NAME" == *"RTX 5090"* ]]; then
        V18_SPEED_PROFILE=rtx5090
    else
        V18_SPEED_PROFILE=fast
    fi
fi
COMPILE_MODE=default
KL_STOP_MODE=minibatch
COMPILE_TRAINING_ARGS=(--compile-training-cnn)
PARALLEL_INFERENCE_ARGS=(--no-pool-parallel-inference)
TIMING_ARGS=(--timing-log-interval 10)
case "$V18_SPEED_PROFILE" in
    quality)
        TRAINING_SHAPE_ARGS=(
            --num-envs 1024
            --n-steps 128
            --seq-len 128
            --batch-size 4096
            --epochs 4
            --pool-active-checkpoints 16
        )
        EVALUATION_SPEED_ARGS=(
            --eval-interval 250
            --eval-games 512
            --eval-num-envs 128
        )
        ;;
    fast)
        TRAINING_SHAPE_ARGS=(
            --num-envs 1024
            --n-steps 128
            --seq-len 128
            --batch-size 8192
            --epochs 4
            --pool-active-checkpoints 8
        )
        EVALUATION_SPEED_ARGS=(
            --eval-interval 500
            --eval-games 512
            --eval-num-envs 256
        )
        ;;
    turbo)
        TRAINING_SHAPE_ARGS=(
            --num-envs 1024
            --n-steps 128
            --seq-len 128
            --batch-size 8192
            --epochs 3
            --pool-active-checkpoints 4
        )
        EVALUATION_SPEED_ARGS=(
            --eval-interval 500
            --eval-games 256
            --eval-num-envs 256
        )
        ;;
    rtx5090)
        TRAINING_SHAPE_ARGS=(
            --num-envs 3072
            --n-steps 128
            --seq-len 128
            --batch-size 24576
            --epochs 4
            --pool-active-checkpoints 4
        )
        EVALUATION_SPEED_ARGS=(
            --eval-interval 500
            --eval-games 768
            --eval-num-envs 768
        )
        # Large, fixed eager cuDNN batches keep Blackwell busy without lazy
        # Inductor compilation/CUDA-Graph captures stalling the host for many
        # seconds. Set V18_TORCH_COMPILE=1 to opt back in after profiling the
        # exact server software stack.
        COMPILE_TRAINING_ARGS=(--no-compile-training-cnn)
        KL_STOP_MODE=epoch
        PARALLEL_INFERENCE_ARGS=(--pool-parallel-inference)
        TIMING_ARGS=(--timing-log-interval 1)
        ;;
    *)
        echo "Unknown V18_SPEED_PROFILE: $V18_SPEED_PROFILE" >&2
        echo "Expected one of: quality, fast, turbo, rtx5090" >&2
        exit 2
        ;;
esac
if [[ "${V18_TORCH_COMPILE:-0}" == "1" ]]; then
    COMPILE_TRAINING_ARGS=(--compile-training-cnn)
    COMPILE_MODE=default
fi
echo "V18 speed profile: $V18_SPEED_PROFILE (GPU: ${GPU_NAME:-manual selection})" >&2

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
V15_LATEST="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v15_nash_latest.pt"
V16_POOL_DIR="$SCRIPT_DIR/models/selfplay_pool_cuda_v16_nash"
V16_SOURCE_CHECKPOINT="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v16_nash_variants/ppo_bs_lstm_cuda_v16_nash_steps_005734400000_u00043750.pt"
V16_LEAGUE_CHAMPION="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v16_nash_champion.pt"

V18_OUTPUT_ROOT="${V18_OUTPUT_ROOT:-$SCRIPT_DIR}"
V18_POOL_DIR="$V18_OUTPUT_ROOT/models/selfplay_pool_cuda_v18_v16_extension"
V18_VARIANT_DIR="$V18_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v18_v16_extension_variants"
V18_BOOTSTRAP_CHECKPOINT="$V18_VARIANT_DIR/ppo_bs_lstm_cuda_v18_v16_extension_steps_005734400000_u00043750.pt"
V18_LEAGUE_CHAMPION="$V18_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v18_v16_extension_champion.pt"
V18_LATEST="$V18_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v18_v16_extension_latest.pt"
V18_CHAMPION_ZIP="$V18_OUTPUT_ROOT/ppo_bs_lstm_cuda_v18_v16_extension_champion.zip"
V18_FINAL_ZIP="$V18_OUTPUT_ROOT/ppo_bs_lstm_cuda_v18_v16_extension.zip"
V18_RUN_DIR="$V18_OUTPUT_ROOT/runs/ppo_bs_lstm_cuda_v18_v16_extension"
if [[ -n "${V18_TENSORBOARD_LOG_DIR:-}" ]]; then
    V18_TENSORBOARD_DIR="$V18_TENSORBOARD_LOG_DIR"
elif [[ "$V18_SPEED_PROFILE" == "rtx5090" ]]; then
    # Cloud workspace mounts are often optimized for durability rather than
    # many tiny event-file writes. Keep the hot event stream on local storage.
    V18_TENSORBOARD_DIR="${TMPDIR:-/tmp}/hisss-v18-tensorboard"
else
    V18_TENSORBOARD_DIR="$V18_RUN_DIR"
fi
echo "V18 TensorBoard directory: $V18_TENSORBOARD_DIR" >&2

for anchor in \
    "$ANCHOR_655M" \
    "$ANCHOR_V7_917M" \
    "$V9_HOF_750" \
    "$V9_HOF_2500" \
    "$V9_HOF_5500" \
    "$V9_LEAGUE_CHAMPION" \
    "$V13_LEAGUE_CHAMPION" \
    "$V13_SELFPLAY_LAST" \
    "$V14_REFERENCE" \
    "$V15_LATEST" \
    "$V16_SOURCE_CHECKPOINT" \
    "$V16_POOL_DIR/nash_state.json" \
    "$V16_LEAGUE_CHAMPION"; do
    if [[ ! -f "$anchor" ]]; then
        echo "Required V18 reference opponent not found: $anchor" >&2
        exit 2
    fi
done

RESUME_PATH="${RESUME_PATH:-}"
START_ARGS=()
POOL_LOAD_ARGS=(--pool-load-existing)

if [[ -n "$RESUME_PATH" ]]; then
    if [[ ! -f "$RESUME_PATH" ]]; then
        echo "V18 resume checkpoint not found: $RESUME_PATH" >&2
        exit 2
    fi
    RESUME_REAL="$(readlink -f -- "$RESUME_PATH")"
    VARIANT_REAL="$(readlink -m -- "$V18_VARIANT_DIR")"
    case "$RESUME_REAL" in
        "$VARIANT_REAL"/ppo_bs_lstm_cuda_v18_v16_extension_steps_*_u*.pt) ;;
        *)
            echo "Refusing non-V18 resume checkpoint: $RESUME_PATH" >&2
            exit 2
            ;;
    esac
    if [[ ! -f "$V18_POOL_DIR/nash_state.json" ]]; then
        echo "V18 resume requires its existing pool: $V18_POOL_DIR" >&2
        exit 2
    fi
    START_ARGS=(--resume-path "$RESUME_PATH")
    POOL_LOAD_ARGS=(--pool-load-existing)
else
    # The default invocation is the one-time V16 -> V18 branch. Refuse to merge
    # it with a previous attempt; later invocations must name a V18 checkpoint.
    shopt -s nullglob
    EXISTING_POOL=("$V18_POOL_DIR"/*)
    EXISTING_VARIANTS=("$V18_VARIANT_DIR"/*)
    EXISTING_RUN=("$V18_RUN_DIR"/*)
    shopt -u nullglob
    if [[ -e "$V18_LEAGUE_CHAMPION" \
        || -e "$V18_LATEST" \
        || -e "$V18_CHAMPION_ZIP" \
        || -e "$V18_FINAL_ZIP" \
        || ${#EXISTING_POOL[@]} -gt 0 \
        || ${#EXISTING_VARIANTS[@]} -gt 0 \
        || ${#EXISTING_RUN[@]} -gt 0 ]]; then
        echo "Existing V18 state found; refusing a second V16 bootstrap." >&2
        echo "Set RESUME_PATH to a V18 resumable .pt checkpoint instead." >&2
        exit 2
    fi
    mkdir -p "$V18_POOL_DIR" "$V18_VARIANT_DIR"
    # Pool snapshots are immutable. Hardlinks isolate retirement/quarantine by
    # directory entry without consuming another ~1.7 GiB of checkpoint data.
    cp -al -- "$V16_POOL_DIR/." "$V18_POOL_DIR/"
    ln -- "$V16_SOURCE_CHECKPOINT" "$V18_BOOTSTRAP_CHECKPOINT"
    RESUME_PATH="$V18_BOOTSTRAP_CHECKPOINT"
    START_ARGS=(--resume-path "$RESUME_PATH")
    echo "[v18] cloned the V16 population and full update-43750 state"
fi

# The bootstrap checkpoint is bit-identical to the V16 league champion and
# carries its Adam moments, AMP scaler, counters, and saved constant 5e-6 LR.
# V18 therefore starts from the strongest known policy without optimizer shock.
exec "$PYTHON" train_cuda.py \
    "${TRAINING_SHAPE_ARGS[@]}" \
    --updates 25250 \
    --stop-after-update 69000 \
    --rollout-obs-dtype float16 \
    --amp-dtype float16 \
    --channels-last \
    "${COMPILE_TRAINING_ARGS[@]}" \
    --compile-mode "$COMPILE_MODE" \
    "${TIMING_ARGS[@]}" \
    --tensorboard-log-dir "$V18_TENSORBOARD_DIR" \
    --tensorboard-update-interval 10 \
    --tensorboard-max-queue 1000 \
    --tensorboard-flush-secs 120 \
    --lr 5e-6 \
    --lr-schedule constant \
    --lr-floor 1e-6 \
    --target-kl 0.012 \
    --kl-stop-mode "$KL_STOP_MODE" \
    --gamma 1.0 \
    --gae-lambda 0.97 \
    --clip-range 0.2 \
    --ent-coef 0.012 \
    --ent-coef-final 0.0025 \
    --ent-decay-updates 18000 \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --potential-length-coef 0.05 \
    --potential-health-coef 0.02 \
    --potential-mobility-coef 0.02 \
    --reward-scheme tournament \
    --agent-action-mask observable \
    --opponent-mode curriculum \
    --selfplay-start-update 250 \
    --selfplay-update-interval 50 \
    --pool-dir "$V18_POOL_DIR" \
    --pool-anchor "old_655m=$ANCHOR_655M" \
    --pool-anchor "v7_917m=$ANCHOR_V7_917M" \
    --pool-anchor "v9_hof_750=$V9_HOF_750" \
    --pool-anchor "v9_hof_2500=$V9_HOF_2500" \
    --pool-anchor "v9_hof_5500=$V9_HOF_5500" \
    --pool-anchor "v9_champion=$V9_LEAGUE_CHAMPION" \
    --pool-anchor "v13_champion=$V13_LEAGUE_CHAMPION" \
    --pool-anchor "v13_selfplay_last=$V13_SELFPLAY_LAST" \
    --pool-anchor "v14_reference=$V14_REFERENCE" \
    --pool-anchor "v15_final=$V15_LATEST" \
    --max-pool-checkpoints 100 \
    --pool-checkpoint-weight 0.70 \
    --pool-best-weight 0.15 \
    --pool-hungry-weight 0.10 \
    --pool-random-weight 0.05 \
    --pool-latest-probability 0.10 \
    --pool-anchor-probability 0.50 \
    --pool-nash-history-size 24 \
    --pool-nash-iterations 3000 \
    --pool-nash-exploration 0.10 \
    --pool-nash-score-half-life-updates 1000 \
    --pool-champion-min-weight 0.15 \
    --pool-min-action-disagreement 0.08 \
    --pool-diversity-probe-size 512 \
    --pool-focus-label league_champion \
    --pool-focus-probability 0.0 \
    --pool-deterministic-probability 0.50 \
    "${PARALLEL_INFERENCE_ARGS[@]}" \
    "${POOL_LOAD_ARGS[@]}" \
    "${EVALUATION_SPEED_ARGS[@]}" \
    --eval-max-turns 2000 \
    --eval-seed 180018 \
    --eval-seed-stride 1000003 \
    --eval-layout copies \
    --eval-opponents 4 \
    --league-champion-path "$V18_LEAGUE_CHAMPION" \
    --league-champion-zip-path "$V18_CHAMPION_ZIP" \
    --league-initial-champion-path "$V16_LEAGUE_CHAMPION" \
    --league-promotion-games 512 \
    --league-promotion-seed 180123 \
    --league-promotion-seed 180888 \
    --league-promotion-seed 694242 \
    --league-promotion-seed-stride 1000003 \
    --league-promotion-layout copies \
    --league-promotion-layout solo-pair \
    --league-promotion-threshold 0.525 \
    --league-min-promotion-interval 250 \
    --league-guard-score old_655m=0.44 \
    --league-guard-score v7_917m=0.36 \
    --league-guard-score v9_hof_750=0.70 \
    --league-guard-score v9_hof_2500=0.69 \
    --league-guard-score v9_hof_5500=0.58 \
    --league-guard-score v9_champion=0.61 \
    --league-guard-score v13_champion=0.55 \
    --league-guard-score v13_selfplay_last=0.43 \
    --league-guard-score v14_reference=0.50 \
    --league-guard-score v15_final=0.50 \
    --sb3-checkpoint-interval 250 \
    --sb3-checkpoint-dir "$V18_VARIANT_DIR" \
    --sb3-checkpoint-prefix ppo_bs_lstm_cuda_v18_v16_extension \
    --max-sb3-checkpoints 16 \
    --fused-adam \
    --allow-tf32 \
    --seed 181818 \
    --save-path "$V18_LATEST" \
    --sb3-save-path "$V18_FINAL_ZIP" \
    "${START_ARGS[@]}" \
    "$@"
