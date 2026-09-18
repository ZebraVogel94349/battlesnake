#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    exec "$PYTHON" train_cuda.py --help
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

V20_NUM_ENVS="${V20_NUM_ENVS:-1024}"
V20_BATCH_SIZE="${V20_BATCH_SIZE:-8192}"
V20_ACTIVE_CHECKPOINTS="${V20_ACTIVE_CHECKPOINTS:-8}"
V20_UPDATES="${V20_UPDATES:-30000}"
V20_EVAL_GAMES="${V20_EVAL_GAMES:-512}"
V20_EVAL_NUM_ENVS="${V20_EVAL_NUM_ENVS:-256}"

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

# Copy the finished v19 files from the server under V19_INPUT_ROOT first.
# The final ZIP and latest checkpoint are completion markers, so v20 cannot
# accidentally clone the population while v19 is still writing it.
V19_INPUT_ROOT="${V19_INPUT_ROOT:-$SCRIPT_DIR}"
V19_POOL_DIR="$V19_INPUT_ROOT/models/selfplay_pool_cuda_v19_real_anchors"
V19_VARIANT_DIR="$V19_INPUT_ROOT/models/ppo_bs_lstm_cuda_v19_real_anchors_variants"
V19_LEAGUE_CHAMPION="$V19_INPUT_ROOT/models/ppo_bs_lstm_cuda_v19_real_anchors_champion.pt"
V19_LATEST="$V19_INPUT_ROOT/models/ppo_bs_lstm_cuda_v19_real_anchors_variants/ppo_bs_lstm_cuda_v19_real_anchors_steps_010911744000_u00060750.pt"
V19_FINAL_ZIP="$V19_INPUT_ROOT/models/ppo_bs_lstm_cuda_v19_real_anchors_variants/ppo_bs_lstm_cuda_v19_real_anchors_steps_010911744000_u00060750.zip"
V19_SOURCE_CHECKPOINT="${V19_SOURCE_CHECKPOINT:-}"
if [[ -z "$V19_SOURCE_CHECKPOINT" ]]; then
    shopt -s nullglob
    V19_CANDIDATES=(
        "$V19_VARIANT_DIR"/ppo_bs_lstm_cuda_v19_real_anchors_steps_*_u*.pt
    )
    shopt -u nullglob
    if [[ ${#V19_CANDIDATES[@]} -gt 0 ]]; then
        V19_SOURCE_CHECKPOINT="${V19_CANDIDATES[-1]}"
    fi
fi

V20_OUTPUT_ROOT="${V20_OUTPUT_ROOT:-$SCRIPT_DIR}"
V20_POOL_DIR="$V20_OUTPUT_ROOT/models/selfplay_pool_cuda_v20_snake25_anchors"
V20_VARIANT_DIR="$V20_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v20_snake25_anchors_variants"
V20_LEAGUE_CHAMPION="$V20_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v20_snake25_anchors_champion.pt"
V20_CHAMPION_ZIP="$V20_OUTPUT_ROOT/ppo_bs_lstm_cuda_v20_snake25_anchors_champion.zip"
V20_LATEST="$V20_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v20_snake25_anchors_latest.pt"
V20_FINAL_ZIP="$V20_OUTPUT_ROOT/ppo_bs_lstm_cuda_v20_snake25_anchors.zip"
V20_RUN_DIR="$V20_OUTPUT_ROOT/runs/ppo_bs_lstm_cuda_v20_snake25_anchors"

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
    "$V15_LATEST" \
    "$V19_POOL_DIR/nash_state.json" \
    "$V19_LEAGUE_CHAMPION" \
    "$V19_LATEST" \
    "$V19_FINAL_ZIP"; do
    if [[ ! -f "$required" ]]; then
        echo "Required completed-v19 input not found: $required" >&2
        echo "Let v19 finish, then copy its models, pool, and final ZIP here." >&2
        exit 2
    fi
done
if [[ -z "$V19_SOURCE_CHECKPOINT" || ! -f "$V19_SOURCE_CHECKPOINT" ]]; then
    echo "No resumable v19 checkpoint found in: $V19_VARIANT_DIR" >&2
    echo "Set V19_SOURCE_CHECKPOINT=/path/to/the/final-v19-variant.pt." >&2
    exit 2
fi

V19_REQUIRED_UPDATE="${V19_REQUIRED_UPDATE:-60750}"
SOURCE_NAME="$(basename -- "$V19_SOURCE_CHECKPOINT")"
if [[ "$SOURCE_NAME" =~ _u([0-9]+)\.pt$ ]]; then
    SOURCE_UPDATE=$((10#${BASH_REMATCH[1]}))
else
    echo "Cannot read the update counter from v19 checkpoint: $SOURCE_NAME" >&2
    exit 2
fi
if (( SOURCE_UPDATE < V19_REQUIRED_UPDATE )); then
    echo "Refusing incomplete v19 checkpoint at update $SOURCE_UPDATE." >&2
    echo "Expected at least update $V19_REQUIRED_UPDATE." >&2
    exit 2
fi
V20_STOP_AFTER_UPDATE=$((SOURCE_UPDATE + V20_UPDATES))

RESUME_PATH="${V20_RESUME_PATH:-}"
if [[ -n "$RESUME_PATH" ]]; then
    if [[ ! -f "$RESUME_PATH" ]]; then
        echo "V20 resume checkpoint not found: $RESUME_PATH" >&2
        exit 2
    fi
    if [[ ! -f "$V20_POOL_DIR/nash_state.json" ]]; then
        echo "V20 resume requires its existing pool: $V20_POOL_DIR" >&2
        exit 2
    fi
else
    shopt -s nullglob
    EXISTING_V20=("$V20_POOL_DIR"/* "$V20_VARIANT_DIR"/* "$V20_RUN_DIR"/*)
    shopt -u nullglob
    if [[ ${#EXISTING_V20[@]} -gt 0 || -e "$V20_LATEST" || -e "$V20_FINAL_ZIP" ]]; then
        echo "Existing v20 state found; refusing a second v19 bootstrap." >&2
        echo "Set V20_RESUME_PATH to a v20 resumable checkpoint." >&2
        exit 2
    fi
    mkdir -p "$V20_POOL_DIR" "$V20_VARIANT_DIR"
    cp -al -- "$V19_POOL_DIR/." "$V20_POOL_DIR/"
    RESUME_PATH="$V19_SOURCE_CHECKPOINT"
    echo "[v20] cloned the completed v19 population"
fi

if [[ "${1:-}" == "--check" ]]; then
    echo "[v20] launcher inputs and resume pool are available"
    exit 0
fi

exec "$PYTHON" train_cuda.py \
    --num-envs "$V20_NUM_ENVS" \
    --n-steps 128 \
    --seq-len 128 \
    --batch-size "$V20_BATCH_SIZE" \
    --epochs 4 \
    --updates "$V20_UPDATES" \
    --stop-after-update "$V20_STOP_AFTER_UPDATE" \
    --rollout-obs-dtype float16 \
    --amp-dtype float16 \
    --channels-last \
    --compile-training-cnn \
    --compile-mode default \
    --tensorboard-log-dir "$V20_RUN_DIR" \
    --tensorboard-update-interval 10 \
    --lr 1e-5 \
    --continuation-lr 1e-5 \
    --lr-schedule constant \
    --lr-floor 1e-6 \
    --target-kl 0.012 \
    --gamma 1.0 \
    --gae-lambda 0.97 \
    --clip-range 0.2 \
    --ent-coef 0.006 \
    --ent-coef-final 0.0025 \
    --ent-decay-updates "$V20_UPDATES" \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --potential-length-coef 0.05 \
    --potential-health-coef 0.02 \
    --potential-mobility-coef 0.02 \
    --reward-scheme tournament \
    --agent-action-mask observable \
    --opponent-mode selfplay \
    --selfplay-update-interval 50 \
    --pool-dir "$V20_POOL_DIR" \
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
    --pool-anchor "v19_final=$V19_SOURCE_CHECKPOINT" \
    --max-pool-checkpoints 100 \
    --pool-active-checkpoints "$V20_ACTIVE_CHECKPOINTS" \
    --pool-checkpoint-weight 0.50 \
    --pool-best-weight 0.10 \
    --pool-hungry-weight 0.005 \
    --pool-random-weight 0.005 \
    --pool-heuristic-weight forager=0.0025 \
    --pool-heuristic-weight hunter=0.12 \
    --pool-heuristic-weight territorial=0.0025 \
    --pool-heuristic-weight edge_trapper=0.0025 \
    --pool-heuristic-weight survivor=0.0025 \
    --pool-heuristic-weight snake25=0.12 \
    --pool-heuristic-weight snake25_interceptor=0.07 \
    --pool-heuristic-weight snake25_denier=0.07 \
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
    --pool-deterministic-probability 0.50 \
    --pool-load-existing \
    --eval-interval 250 \
    --eval-games "$V20_EVAL_GAMES" \
    --eval-num-envs "$V20_EVAL_NUM_ENVS" \
    --eval-max-turns 2000 \
    --eval-seed 200020 \
    --eval-seed-stride 1000003 \
    --eval-layout copies \
    --eval-opponents 4 \
    --league-champion-path "$V20_LEAGUE_CHAMPION" \
    --league-champion-zip-path "$V20_CHAMPION_ZIP" \
    --league-initial-champion-path "$V19_LEAGUE_CHAMPION" \
    --league-promotion-games 512 \
    --league-promotion-seed 200123 \
    --league-promotion-seed 200888 \
    --league-promotion-seed 814242 \
    --league-promotion-layout copies \
    --league-promotion-layout solo-pair \
    --league-promotion-threshold 0.525 \
    --league-min-promotion-interval 250 \
    --league-guard-score old_655m=0.44 \
    --league-guard-score v7_917m=0.36 \
    --league-guard-score v19_final=0.48 \
    --sb3-checkpoint-interval 250 \
    --sb3-checkpoint-dir "$V20_VARIANT_DIR" \
    --sb3-checkpoint-prefix ppo_bs_lstm_cuda_v20_snake25_anchors \
    --max-sb3-checkpoints 16 \
    --fused-adam \
    --allow-tf32 \
    --seed 202020 \
    --save-path "$V20_LATEST" \
    --sb3-save-path "$V20_FINAL_ZIP" \
    --resume-path "$RESUME_PATH" \
    "$@"
