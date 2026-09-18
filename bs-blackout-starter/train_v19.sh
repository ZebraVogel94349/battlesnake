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
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    exec "$PYTHON" train_cuda.py --help
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Local defaults are suitable for an 8-12 GiB GPU and can be overridden
# without editing this file. Extra train_cuda.py flags may also be appended.
V19_NUM_ENVS="${V19_NUM_ENVS:-1024}"
V19_BATCH_SIZE="${V19_BATCH_SIZE:-8192}"
V19_ACTIVE_CHECKPOINTS="${V19_ACTIVE_CHECKPOINTS:-8}"
V19_UPDATES="${V19_UPDATES:-15000}"
V19_EVAL_GAMES="${V19_EVAL_GAMES:-512}"
V19_EVAL_NUM_ENVS="${V19_EVAL_NUM_ENVS:-256}"

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

# Copy the completed server run into V18_INPUT_ROOT (or the repository) before
# starting v19. V19_SOURCE_CHECKPOINT can point at an explicit resumable .pt;
# otherwise the lexically latest padded v18 variant checkpoint is selected.
V18_INPUT_ROOT="${V18_INPUT_ROOT:-$SCRIPT_DIR}"
V18_POOL_DIR="$V18_INPUT_ROOT/models/selfplay_pool_cuda_v18_v16_extension"
V18_VARIANT_DIR="$V18_INPUT_ROOT/models/ppo_bs_lstm_cuda_v18_v16_extension_variants"
V18_LEAGUE_CHAMPION="$V18_INPUT_ROOT/models/ppo_bs_lstm_cuda_v18_v16_extension_champion.pt"
V19_SOURCE_CHECKPOINT="${V19_SOURCE_CHECKPOINT:-}"
if [[ -z "$V19_SOURCE_CHECKPOINT" ]]; then
    shopt -s nullglob
    V18_CANDIDATES=(
        "$V18_VARIANT_DIR"/ppo_bs_lstm_cuda_v18_v16_extension_steps_*_u*.pt
    )
    shopt -u nullglob
    if [[ ${#V18_CANDIDATES[@]} -gt 0 ]]; then
        V19_SOURCE_CHECKPOINT="${V18_CANDIDATES[-1]}"
    fi
fi

V19_OUTPUT_ROOT="${V19_OUTPUT_ROOT:-$SCRIPT_DIR}"
V19_POOL_DIR="$V19_OUTPUT_ROOT/models/selfplay_pool_cuda_v19_real_anchors"
V19_VARIANT_DIR="$V19_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v19_real_anchors_variants"
V19_LEAGUE_CHAMPION="$V19_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v19_real_anchors_champion.pt"
V19_CHAMPION_ZIP="$V19_OUTPUT_ROOT/ppo_bs_lstm_cuda_v19_real_anchors_champion.zip"
V19_LATEST="$V19_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v19_real_anchors_latest.pt"
V19_FINAL_ZIP="$V19_OUTPUT_ROOT/ppo_bs_lstm_cuda_v19_real_anchors.zip"
V19_RUN_DIR="$V19_OUTPUT_ROOT/runs/ppo_bs_lstm_cuda_v19_real_anchors"

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
    "$V18_POOL_DIR/nash_state.json" \
    "$V18_LEAGUE_CHAMPION"; do
    if [[ ! -f "$required" ]]; then
        echo "Required v19 input not found: $required" >&2
        echo "Copy the completed v18 models and self-play pool from the server first." >&2
        exit 2
    fi
done
if [[ -z "$V19_SOURCE_CHECKPOINT" || ! -f "$V19_SOURCE_CHECKPOINT" ]]; then
    echo "No resumable v18 training checkpoint found in: $V18_VARIANT_DIR" >&2
    echo "Set V19_SOURCE_CHECKPOINT=/path/to/the/final-v18-variant.pt." >&2
    exit 2
fi
V18_REQUIRED_UPDATE="${V18_REQUIRED_UPDATE:-54000}"
SOURCE_NAME="$(basename -- "$V19_SOURCE_CHECKPOINT")"
if [[ "$SOURCE_NAME" =~ _u([0-9]+)\.pt$ ]]; then
    SOURCE_UPDATE=$((10#${BASH_REMATCH[1]}))
else
    echo "Cannot read the update counter from v18 checkpoint: $SOURCE_NAME" >&2
    exit 2
fi
if (( SOURCE_UPDATE < V18_REQUIRED_UPDATE )); then
    echo "Refusing incomplete v18 checkpoint at update $SOURCE_UPDATE." >&2
    echo "Expected at least update $V18_REQUIRED_UPDATE; set V18_REQUIRED_UPDATE only if the server run intentionally ended earlier." >&2
    exit 2
fi

RESUME_PATH="${V19_RESUME_PATH:-}"
if [[ -n "$RESUME_PATH" ]]; then
    if [[ ! -f "$RESUME_PATH" ]]; then
        echo "V19 resume checkpoint not found: $RESUME_PATH" >&2
        exit 2
    fi
    if [[ ! -f "$V19_POOL_DIR/nash_state.json" ]]; then
        echo "V19 resume requires its existing pool: $V19_POOL_DIR" >&2
        exit 2
    fi
else
    # A fresh v19 branch inherits v18's immutable population and Nash history.
    # Hardlinks avoid another multi-gigabyte copy while keeping retirement safe.
    shopt -s nullglob
    EXISTING_V19=("$V19_POOL_DIR"/* "$V19_VARIANT_DIR"/* "$V19_RUN_DIR"/*)
    shopt -u nullglob
    if [[ ${#EXISTING_V19[@]} -gt 0 || -e "$V19_LATEST" || -e "$V19_FINAL_ZIP" ]]; then
        echo "Existing v19 state found; refusing a second v18 bootstrap." >&2
        echo "Set V19_RESUME_PATH to a v19 resumable checkpoint." >&2
        exit 2
    fi
    mkdir -p "$V19_POOL_DIR" "$V19_VARIANT_DIR"
    cp -al -- "$V18_POOL_DIR/." "$V19_POOL_DIR/"
    RESUME_PATH="$V19_SOURCE_CHECKPOINT"
    echo "[v19] cloned the completed v18 population"
fi

exec "$PYTHON" train_cuda.py \
    --num-envs "$V19_NUM_ENVS" \
    --n-steps 128 \
    --seq-len 128 \
    --batch-size "$V19_BATCH_SIZE" \
    --epochs 4 \
    --updates "$V19_UPDATES" \
    --rollout-obs-dtype float16 \
    --amp-dtype float16 \
    --channels-last \
    --compile-training-cnn \
    --compile-mode default \
    --tensorboard-log-dir "$V19_RUN_DIR" \
    --tensorboard-update-interval 10 \
    --lr 5e-6 \
    --lr-schedule constant \
    --lr-floor 1e-6 \
    --target-kl 0.012 \
    --gamma 1.0 \
    --gae-lambda 0.97 \
    --clip-range 0.2 \
    --ent-coef 0.006 \
    --ent-coef-final 0.0025 \
    --ent-decay-updates "$V19_UPDATES" \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --potential-length-coef 0.05 \
    --potential-health-coef 0.02 \
    --potential-mobility-coef 0.02 \
    --reward-scheme tournament \
    --agent-action-mask observable \
    --opponent-mode selfplay \
    --selfplay-update-interval 50 \
    --pool-dir "$V19_POOL_DIR" \
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
    --pool-anchor "v18_final=$V19_SOURCE_CHECKPOINT" \
    --max-pool-checkpoints 100 \
    --pool-active-checkpoints "$V19_ACTIVE_CHECKPOINTS" \
    --pool-checkpoint-weight 0.60 \
    --pool-best-weight 0.07 \
    --pool-hungry-weight 0.04 \
    --pool-random-weight 0.04 \
    --pool-heuristic-weight forager=0.05 \
    --pool-heuristic-weight hunter=0.05 \
    --pool-heuristic-weight territorial=0.05 \
    --pool-heuristic-weight edge_trapper=0.05 \
    --pool-heuristic-weight survivor=0.05 \
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
    --eval-interval 500 \
    --eval-games "$V19_EVAL_GAMES" \
    --eval-num-envs "$V19_EVAL_NUM_ENVS" \
    --eval-max-turns 2000 \
    --eval-seed 190019 \
    --eval-seed-stride 1000003 \
    --eval-layout copies \
    --eval-opponents 4 \
    --league-champion-path "$V19_LEAGUE_CHAMPION" \
    --league-champion-zip-path "$V19_CHAMPION_ZIP" \
    --league-initial-champion-path "$V18_LEAGUE_CHAMPION" \
    --league-promotion-games 512 \
    --league-promotion-seed 190123 \
    --league-promotion-seed 190888 \
    --league-promotion-seed 794242 \
    --league-promotion-layout copies \
    --league-promotion-layout solo-pair \
    --league-promotion-threshold 0.525 \
    --league-min-promotion-interval 250 \
    --league-guard-score old_655m=0.44 \
    --league-guard-score v7_917m=0.36 \
    --league-guard-score v18_final=0.48 \
    --sb3-checkpoint-interval 250 \
    --sb3-checkpoint-dir "$V19_VARIANT_DIR" \
    --sb3-checkpoint-prefix ppo_bs_lstm_cuda_v19_real_anchors \
    --max-sb3-checkpoints 16 \
    --fused-adam \
    --allow-tf32 \
    --seed 191919 \
    --save-path "$V19_LATEST" \
    --sb3-save-path "$V19_FINAL_ZIP" \
    --resume-path "$RESUME_PATH" \
    "$@"
