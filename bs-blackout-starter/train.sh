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

# Helps the 8 GiB RTX 3070 avoid allocator fragmentation as the checkpoint pool
# grows. CUDA_VISIBLE_DEVICES and PYTHON can still be overridden by the caller.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
V15_POOL_DIR="$SCRIPT_DIR/models/selfplay_pool_cuda_v15_nash"
V15_LEAGUE_CHAMPION="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v15_nash_champion.pt"
V15_LATEST="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v15_nash_latest.pt"
V16_POOL_DIR="$SCRIPT_DIR/models/selfplay_pool_cuda_v16_nash"
V16_VARIANT_DIR="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v16_nash_variants"
V16_BOOTSTRAP_CHECKPOINT="$V16_VARIANT_DIR/ppo_bs_lstm_cuda_v16_nash_steps_003014656000_u00023000.pt"
V16_RESUME_33000="$V16_VARIANT_DIR/ppo_bs_lstm_cuda_v16_nash_steps_004325376000_u00033000.pt"
V16_LEAGUE_CHAMPION="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v16_nash_champion.pt"
V16_LATEST="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v16_nash_latest.pt"
V16_CHAMPION_ZIP="$SCRIPT_DIR/ppo_bs_lstm_cuda_v16_nash_champion.zip"
V16_FINAL_ZIP="$SCRIPT_DIR/ppo_bs_lstm_cuda_v16_nash.zip"
V16_RUN_DIR="$SCRIPT_DIR/runs/ppo_bs_lstm_cuda_v16_nash"

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
    "$V15_LEAGUE_CHAMPION" \
    "$V15_POOL_DIR/nash_state.json"; do
    if [[ ! -f "$anchor" ]]; then
        echo "Permanent anchor not found: $anchor" >&2
        exit 2
    fi
done

RESUME_PATH="${RESUME_PATH:-$V16_RESUME_33000}"
if [[ ! -f "$RESUME_PATH" ]]; then
    echo "Resume checkpoint not found: $RESUME_PATH" >&2
    exit 2
fi

BOOTSTRAP_V16=0
if [[ "$(readlink -f -- "$RESUME_PATH")" == "$(readlink -f -- "$V15_LATEST")" ]]; then
    BOOTSTRAP_V16=1
fi

if (( BOOTSTRAP_V16 )); then
    # A default invocation is the one-time V15 -> V16 transition. Refuse to
    # mix it with a partial V16 attempt; that must use an explicit V16 resume.
    shopt -s nullglob
    EXISTING_V16_POOL=("$V16_POOL_DIR"/*)
    EXISTING_V16_VARIANTS=("$V16_VARIANT_DIR"/*)
    EXISTING_V16_RUN=("$V16_RUN_DIR"/*)
    shopt -u nullglob
    if [[ -e "$V16_LEAGUE_CHAMPION" \
        || -e "$V16_LATEST" \
        || -e "$V16_CHAMPION_ZIP" \
        || -e "$V16_FINAL_ZIP" \
        || ${#EXISTING_V16_POOL[@]} -gt 0 \
        || ${#EXISTING_V16_VARIANTS[@]} -gt 0 \
        || ${#EXISTING_V16_RUN[@]} -gt 0 ]]; then
        echo "Existing v16 state found. Set RESUME_PATH to the latest v16 .pt checkpoint." >&2
        exit 2
    fi
    if [[ -e "$V16_POOL_DIR" && ! -d "$V16_POOL_DIR" ]]; then
        echo "V16 pool path is not a directory: $V16_POOL_DIR" >&2
        exit 2
    fi
    mkdir -p "$V16_POOL_DIR"
    # Pool checkpoints are immutable and retired via unlink. Hardlinking them
    # preserves V15 byte-for-byte while avoiding another 1.3 GiB allocation.
    cp -al -- "$V15_POOL_DIR/." "$V16_POOL_DIR/"
    mkdir -p "$V16_VARIANT_DIR"
    ln -- "$V15_LATEST" "$V16_BOOTSTRAP_CHECKPOINT"
    RESUME_PATH="$V16_BOOTSTRAP_CHECKPOINT"
    echo "[v16] cloned V15 population and Nash state into $V16_POOL_DIR"
else
    if [[ ! -f "$V16_POOL_DIR/nash_state.json" ]]; then
        echo "V16 resume requires its existing population: $V16_POOL_DIR" >&2
        exit 2
    fi
fi

mkdir -p "$V16_VARIANT_DIR"
START_ARGS=(--resume-path "$RESUME_PATH")
POOL_LOAD_ARGS=(--pool-load-existing)

# V16 now resumes exactly at update 33,000. It restores learner, optimizer,
# AMP scaler, counters, population, Nash matrix, and the saved 5e-6 learning
# rate. Pool snapshots newer than the selected checkpoint are moved into a
# recoverable quarantine directory so they can never leak into the new branch.
#
# Resume V16 at update 33,000:
#   ./train.sh
# Resume a different V16 checkpoint explicitly:
#   RESUME_PATH=models/ppo_bs_lstm_cuda_v16_nash_variants/<checkpoint>.pt ./train.sh
# Monitor with: .venv/bin/tensorboard --logdir runs --port 6006
exec "$PYTHON" train_cuda.py \
    --num-envs 1024 \
    --n-steps 128 \
    --seq-len 128 \
    --updates 23000 \
    --stop-after-update 46000 \
    --batch-size 4096 \
    --epochs 4 \
    --rollout-obs-dtype float16 \
    --amp-dtype float16 \
    --channels-last \
    --compile-training-cnn \
    --compile-mode default \
    --tensorboard-log-dir "$V16_RUN_DIR" \
    --lr 5e-6 \
    --lr-schedule constant \
    --lr-floor 1e-6 \
    --target-kl 0.012 \
    --kl-stop-mode minibatch \
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
    --pool-dir "$V16_POOL_DIR" \
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
    --pool-active-checkpoints 16 \
    --pool-checkpoint-weight 0.70 \
    --pool-best-weight 0.15 \
    --pool-hungry-weight 0.10 \
    --pool-random-weight 0.05 \
    --pool-latest-probability 0.10 \
    --pool-anchor-probability 0.50 \
    --pool-nash-history-size 16 \
    --pool-nash-iterations 2000 \
    --pool-nash-exploration 0.10 \
    --pool-min-action-disagreement 0.08 \
    --pool-diversity-probe-size 512 \
    --pool-focus-label league_champion \
    --pool-focus-probability 0.0 \
    --pool-deterministic-probability 0.50 \
    "${POOL_LOAD_ARGS[@]}" \
    --eval-interval 250 \
    --eval-games 512 \
    --eval-num-envs 128 \
    --eval-max-turns 2000 \
    --eval-seed 123 \
    --eval-opponents 4 \
    --league-champion-path "$V16_LEAGUE_CHAMPION" \
    --league-champion-zip-path "$V16_CHAMPION_ZIP" \
    --league-initial-champion-path "$V15_LEAGUE_CHAMPION" \
    --league-promotion-games 768 \
    --league-promotion-seed 123 \
    --league-promotion-seed 888 \
    --league-promotion-seed 424242 \
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
    --sb3-checkpoint-dir "$V16_VARIANT_DIR" \
    --sb3-checkpoint-prefix ppo_bs_lstm_cuda_v16_nash \
    --max-sb3-checkpoints 0 \
    --fused-adam \
    --allow-tf32 \
    --seed 161616 \
    --save-path "$V16_LATEST" \
    --sb3-save-path "$V16_FINAL_ZIP" \
    "${START_ARGS[@]}" \
    "$@"
