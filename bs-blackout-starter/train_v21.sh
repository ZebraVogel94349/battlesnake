#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON="python3"

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
        f"Expected the editable local source below {project / 'src'}.\n"
        f"Run: {project / 'bs-blackout-starter' / 'install_local_cuda.sh'}"
    ) from exc
if not getattr(hisss, "cuda_available", lambda: False)():
    error = getattr(hisss, "cuda_last_error", lambda: "CUDA backend missing")()
    raise SystemExit(
        f"local hisss was built without a usable CUDA backend: {error}\n"
        f"Run: {project / 'bs-blackout-starter' / 'install_local_cuda.sh'}"
    )
print(f"[v21] local CUDA hisss: {module}")
PY

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    exec "$PYTHON" train_cuda.py --help
fi

GPU_NAME="$("$PYTHON" -c \
    'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"
if [[ -z "${V21_SPEED_PROFILE:-}" ]]; then
    if [[ "$GPU_NAME" == *"RTX 4090"* ]]; then
        V21_SPEED_PROFILE=rtx4090
    else
        V21_SPEED_PROFILE=standard
    fi
fi

case "$V21_SPEED_PROFILE" in
    standard)
        DEFAULT_NUM_ENVS=1024
        DEFAULT_BATCH_SIZE=16384
        DEFAULT_EPOCHS=4
        DEFAULT_ACTIVE_CHECKPOINTS=4
        COMPILE_MODE=default
        KL_STOP_MODE=minibatch
        PARALLEL_INFERENCE_ARGS=()
        ;;
    rtx4090)
        # Ada benefits from wider, fewer launches. The 24 GiB card can retain
        # the FP16 rollout while training 32k samples per minibatch.
        DEFAULT_NUM_ENVS=2048
        DEFAULT_BATCH_SIZE=32768
        DEFAULT_EPOCHS=3
        DEFAULT_ACTIVE_CHECKPOINTS=4
        COMPILE_MODE=reduce-overhead
        KL_STOP_MODE=epoch
        PARALLEL_INFERENCE_ARGS=(--pool-parallel-inference)
        ;;
    *)
        echo "Unknown V21_SPEED_PROFILE: $V21_SPEED_PROFILE" >&2
        echo "Expected one of: standard, rtx4090" >&2
        exit 2
        ;;
esac

V21_NUM_ENVS="${V21_NUM_ENVS:-$DEFAULT_NUM_ENVS}"
V21_BATCH_SIZE="${V21_BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
V21_EPOCHS="${V21_EPOCHS:-$DEFAULT_EPOCHS}"
V21_ACTIVE_CHECKPOINTS="${V21_ACTIVE_CHECKPOINTS:-$DEFAULT_ACTIVE_CHECKPOINTS}"
V21_ACTIVE_ROTATION_INTERVAL="${V21_ACTIVE_ROTATION_INTERVAL:-20}"
V21_UPDATES="${V21_UPDATES:-40000}"
V21_EVAL_GAMES="${V21_EVAL_GAMES:-256}"
V21_EVAL_NUM_ENVS="${V21_EVAL_NUM_ENVS:-256}"
V21_DUEL_PROBABILITY="${V21_DUEL_PROBABILITY:-0.25}"
V21_BC_EPOCHS="${V21_BC_EPOCHS:-12}"
V21_BC_BATCH_SEQUENCES="${V21_BC_BATCH_SEQUENCES:-32}"

echo "[v21] speed profile=$V21_SPEED_PROFILE gpu=${GPU_NAME:-unknown} " \
     "envs=$V21_NUM_ENVS batch=$V21_BATCH_SIZE epochs=$V21_EPOCHS " \
     "active_models=$V21_ACTIVE_CHECKPOINTS" >&2

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

# V21 is deliberately branched from the validated periodic V20 checkpoint at
# update 78,500.  It does not depend on V20's mutable "latest" or final export.
V20_INPUT_ROOT="${V20_INPUT_ROOT:-$SCRIPT_DIR}"
V20_POOL_DIR="$V20_INPUT_ROOT/models/selfplay_pool_cuda_v20_snake25_anchors"
V20_VARIANT_DIR="$V20_INPUT_ROOT/models/ppo_bs_lstm_cuda_v20_snake25_anchors_variants"
V20_LEAGUE_CHAMPION="$V20_INPUT_ROOT/models/ppo_bs_lstm_cuda_v20_snake25_anchors_champion.pt"
V20_REQUIRED_UPDATE="${V20_REQUIRED_UPDATE:-78500}"
V20_REQUIRED_UPDATE_PADDED="$(printf '%08d' "$V20_REQUIRED_UPDATE")"

V21_SOURCE_CHECKPOINT="${V21_SOURCE_CHECKPOINT:-}"
if [[ -z "$V21_SOURCE_CHECKPOINT" ]]; then
    shopt -s nullglob
    V20_CANDIDATES=(
        "$V20_VARIANT_DIR"/ppo_bs_lstm_cuda_v20_snake25_anchors_steps_*_u"$V20_REQUIRED_UPDATE_PADDED".pt
    )
    shopt -u nullglob
    if [[ ${#V20_CANDIDATES[@]} -eq 1 ]]; then
        V21_SOURCE_CHECKPOINT="${V20_CANDIDATES[0]}"
    fi
fi

shopt -s nullglob
V18_ANCHOR_CANDIDATES=("$V20_POOL_DIR"/anchor_v18_final_*.pt)
shopt -u nullglob
V18_FINAL=""
if [[ ${#V18_ANCHOR_CANDIDATES[@]} -gt 0 ]]; then
    V18_FINAL="${V18_ANCHOR_CANDIDATES[-1]}"
fi
shopt -s nullglob
V19_ANCHOR_CANDIDATES=("$V20_POOL_DIR"/anchor_v19_final_*.pt)
shopt -u nullglob
V19_FINAL=""
if [[ ${#V19_ANCHOR_CANDIDATES[@]} -gt 0 ]]; then
    V19_FINAL="${V19_ANCHOR_CANDIDATES[-1]}"
fi

V21_OUTPUT_ROOT="${V21_OUTPUT_ROOT:-$SCRIPT_DIR}"
V21_POOL_DIR="$V21_OUTPUT_ROOT/models/selfplay_pool_cuda_v21_balanced_duels"
V21_VARIANT_DIR="$V21_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v21_balanced_duels_variants"
V21_LEAGUE_CHAMPION="$V21_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v21_balanced_duels_champion.pt"
V21_CHAMPION_ZIP="$V21_OUTPUT_ROOT/ppo_bs_lstm_cuda_v21_balanced_duels_champion.zip"
V21_LATEST="$V21_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v21_balanced_duels_latest.pt"
V21_FINAL_ZIP="$V21_OUTPUT_ROOT/ppo_bs_lstm_cuda_v21_balanced_duels.zip"
V21_RUN_DIR="$V21_OUTPUT_ROOT/runs/ppo_bs_lstm_cuda_v21_balanced_duels"
V21_GAME_LOG_DIR="${V21_GAME_LOG_DIR:-$PROJECT_DIR/game_logs}"
V21_BC_CHECKPOINT="${V21_BC_CHECKPOINT:-$V21_OUTPUT_ROOT/models/snake25_bc_v21_u${V20_REQUIRED_UPDATE_PADDED}.pt}"
V21_BC_DATASET_CACHE="${V21_BC_DATASET_CACHE:-$V21_OUTPUT_ROOT/models/snake25_bc_dataset_v1.pt}"

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
    "$V18_FINAL" \
    "$V19_FINAL" \
    "$V20_POOL_DIR/nash_state.json" \
    "$V20_LEAGUE_CHAMPION"; do
    if [[ -z "$required" || ! -f "$required" ]]; then
        echo "Required V20 u${V20_REQUIRED_UPDATE_PADDED} input not found: ${required:-v18_final anchor}" >&2
        exit 2
    fi
done
if [[ -z "$V21_SOURCE_CHECKPOINT" || ! -f "$V21_SOURCE_CHECKPOINT" ]]; then
    echo "No resumable V20 checkpoint found in: $V20_VARIANT_DIR" >&2
    exit 2
fi

SOURCE_NAME="$(basename -- "$V21_SOURCE_CHECKPOINT")"
if [[ "$SOURCE_NAME" =~ _u([0-9]+)\.pt$ ]]; then
    SOURCE_UPDATE=$((10#${BASH_REMATCH[1]}))
else
    echo "Cannot read update counter from V20 checkpoint: $SOURCE_NAME" >&2
    exit 2
fi
if (( SOURCE_UPDATE != V20_REQUIRED_UPDATE )); then
    echo "Refusing V20 checkpoint at update $SOURCE_UPDATE." >&2
    echo "V21 must branch exactly at update $V20_REQUIRED_UPDATE." >&2
    exit 2
fi
V21_STOP_AFTER_UPDATE=$((SOURCE_UPDATE + V21_UPDATES))

if [[ ! -d "$V21_GAME_LOG_DIR" ]]; then
    echo "Game-log directory for Behavioral Cloning not found: $V21_GAME_LOG_DIR" >&2
    exit 2
fi

"$PYTHON" - "$V21_SOURCE_CHECKPOINT" "$V20_POOL_DIR/nash_state.json" "$V20_REQUIRED_UPDATE" <<'PY'
import json
import sys
import torch

checkpoint_path, nash_path, raw_expected = sys.argv[1:]
expected = int(raw_expected)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if int(checkpoint.get("update", -1)) != expected:
    raise SystemExit("V20 checkpoint payload has the wrong update")
if not isinstance(checkpoint.get("model_state_dict"), dict):
    raise SystemExit("V20 checkpoint has no model state")
if not isinstance(checkpoint.get("optimizer_state_dict"), dict):
    raise SystemExit("V20 checkpoint has no optimizer state")
nash = json.loads(open(nash_path, encoding="utf-8").read())
if int(nash.get("update", -1)) != expected:
    raise SystemExit("V20 pool Nash state is not from the branch update")
print(
    f"[v21] validated V20 update={expected} "
    f"steps={int(checkpoint.get('total_steps', 0))} with optimizer and Nash state"
)
PY

RESUME_PATH="${V21_RESUME_PATH:-}"
if [[ -n "$RESUME_PATH" ]]; then
    if [[ ! -f "$RESUME_PATH" || ! -f "$V21_POOL_DIR/nash_state.json" ]]; then
        echo "V21 resume checkpoint or pool state is missing." >&2
        exit 2
    fi
else
    shopt -s nullglob
    EXISTING_V21=("$V21_POOL_DIR"/* "$V21_VARIANT_DIR"/* "$V21_RUN_DIR"/*)
    shopt -u nullglob
    if [[ ${#EXISTING_V21[@]} -gt 0 || -e "$V21_LATEST" || -e "$V21_FINAL_ZIP" ]]; then
        echo "Existing V21 state found; set V21_RESUME_PATH to resume it." >&2
        exit 2
    fi
fi

if [[ "${1:-}" == "--check" ]]; then
    "$PYTHON" train_snake25_bc.py \
        --logs "$V21_GAME_LOG_DIR" \
        --source "$V21_SOURCE_CHECKPOINT" \
        --output "$V21_BC_CHECKPOINT" \
        --dataset-cache "$V21_BC_DATASET_CACHE" \
        --check
    echo "[v21] exact V20 u${V20_REQUIRED_UPDATE_PADDED} branch inputs are available"
    echo "[v21] Behavioral Clone: $V21_BC_CHECKPOINT"
    exit 0
fi

if [[ ! -f "$V21_BC_CHECKPOINT" ]]; then
    BC_EXTRA_ARGS=()
    if [[ -n "${V21_BC_MAX_FILES:-}" ]]; then
        BC_EXTRA_ARGS+=(--max-files "$V21_BC_MAX_FILES")
    fi
    echo "[v21] training the frozen Snake-25 Behavioral Clone"
    "$PYTHON" train_snake25_bc.py \
        --logs "$V21_GAME_LOG_DIR" \
        --source "$V21_SOURCE_CHECKPOINT" \
        --output "$V21_BC_CHECKPOINT" \
        --dataset-cache "$V21_BC_DATASET_CACHE" \
        --epochs "$V21_BC_EPOCHS" \
        --batch-sequences "$V21_BC_BATCH_SEQUENCES" \
        "${BC_EXTRA_ARGS[@]}"
fi

"$PYTHON" train_snake25_bc.py \
    --logs "$V21_GAME_LOG_DIR" \
    --source "$V21_SOURCE_CHECKPOINT" \
    --output "$V21_BC_CHECKPOINT" \
    --dataset-cache "$V21_BC_DATASET_CACHE" \
    --check

if [[ -z "$RESUME_PATH" ]]; then
    mkdir -p "$V21_POOL_DIR" "$V21_VARIANT_DIR"
    if [[ "$(stat -c '%d' "$V20_POOL_DIR")" == "$(stat -c '%d' "$V21_POOL_DIR")" ]]; then
        cp -al -- "$V20_POOL_DIR/." "$V21_POOL_DIR/"
        echo "[v21] hardlinked the V20 population"
    else
        cp -a --reflink=auto -- "$V20_POOL_DIR/." "$V21_POOL_DIR/"
        echo "[v21] copied the V20 population across filesystems"
    fi
    RESUME_PATH="$V21_SOURCE_CHECKPOINT"
    echo "[v21] cloned the V20 population at update $SOURCE_UPDATE"
fi

exec "$PYTHON" train_cuda.py \
    --num-envs "$V21_NUM_ENVS" \
    --n-steps 128 \
    --seq-len 128 \
    --batch-size "$V21_BATCH_SIZE" \
    --epochs "$V21_EPOCHS" \
    --updates "$V21_UPDATES" \
    --stop-after-update "$V21_STOP_AFTER_UPDATE" \
    --rollout-obs-dtype float16 \
    --amp-dtype float16 \
    --channels-last \
    --compile-training-cnn \
    --compile-rollout-ops \
    --compile-mode "$COMPILE_MODE" \
    --tensorboard-log-dir "$V21_RUN_DIR" \
    --tensorboard-update-interval 10 \
    --lr 5e-5 \
    --continuation-lr 5e-5 \
    --lr-schedule cosine \
    --lr-floor 5e-6 \
    --target-kl 0.008 \
    --kl-stop-mode "$KL_STOP_MODE" \
    --gamma 1.0 \
    --gae-lambda 0.97 \
    --clip-range 0.15 \
    --ent-coef 0.003 \
    --ent-coef-final 0.0015 \
    --ent-decay-updates "$V21_UPDATES" \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --potential-length-coef 0.05 \
    --potential-health-coef 0.02 \
    --potential-mobility-coef 0.03 \
    --reward-scheme tournament \
    --agent-action-mask observable \
    --opponent-mode selfplay \
    --training-duel-probability "$V21_DUEL_PROBABILITY" \
    --selfplay-update-interval 50 \
    --pool-dir "$V21_POOL_DIR" \
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
    --pool-anchor "v18_final=$V18_FINAL" \
    --pool-anchor "v19_final=$V19_FINAL" \
    --pool-anchor "v20_final=$V21_SOURCE_CHECKPOINT" \
    --pool-anchor "snake25_bc=$V21_BC_CHECKPOINT" \
    --max-pool-checkpoints 100 \
    --pool-active-checkpoints "$V21_ACTIVE_CHECKPOINTS" \
    --pool-active-rotation-interval "$V21_ACTIVE_ROTATION_INTERVAL" \
    --pool-checkpoint-weight 0.55 \
    --pool-best-weight 0.10 \
    --pool-hungry-weight 0.0 \
    --pool-random-weight 0.01 \
    --pool-heuristic-weight hunter=0.10 \
    --pool-heuristic-weight snake25=0.12 \
    --pool-heuristic-weight snake25_duelist=0.12 \
    --pool-duel-opponent-weight snake25_duelist=0.50 \
    --pool-duel-opponent-weight snake25_bc=0.30 \
    --pool-anchor-uniform-floor 0.30 \
    --pool-anchor-priority-exponent 2.0 \
    --pool-anchor-min-weight 0.10 \
    --pool-anchor-score-ema 0.70 \
    --pool-latest-probability 0.10 \
    --pool-anchor-probability 0.60 \
    --pool-nash-history-size 24 \
    --pool-nash-iterations 3000 \
    --pool-nash-exploration 0.10 \
    --pool-nash-score-half-life-updates 500 \
    --pool-champion-min-weight 0.10 \
    --pool-min-action-disagreement 0.08 \
    --pool-diversity-probe-size 512 \
    --pool-focus-label league_champion \
    --pool-deterministic-probability 0.50 \
    "${PARALLEL_INFERENCE_ARGS[@]}" \
    --pool-load-existing \
    --eval-interval 500 \
    --eval-games "$V21_EVAL_GAMES" \
    --eval-num-envs "$V21_EVAL_NUM_ENVS" \
    --eval-max-turns 2000 \
    --eval-seed 210020 \
    --eval-seed-stride 1000003 \
    --eval-layout copies \
    --eval-opponents 4 \
    --league-champion-path "$V21_LEAGUE_CHAMPION" \
    --league-champion-zip-path "$V21_CHAMPION_ZIP" \
    --league-initial-champion-path "$V20_LEAGUE_CHAMPION" \
    --league-promotion-games 512 \
    --league-promotion-seed 210123 \
    --league-promotion-seed 210888 \
    --league-promotion-seed 824242 \
    --league-promotion-layout copies \
    --league-promotion-layout solo-pair \
    --league-promotion-layout true-duel \
    --league-promotion-threshold 0.525 \
    --league-min-promotion-interval 250 \
    --sb3-checkpoint-interval 250 \
    --sb3-checkpoint-dir "$V21_VARIANT_DIR" \
    --sb3-checkpoint-prefix ppo_bs_lstm_cuda_v21_balanced_duels \
    --max-sb3-checkpoints 16 \
    --fused-adam \
    --allow-tf32 \
    --seed 212121 \
    --save-path "$V21_LATEST" \
    --sb3-save-path "$V21_FINAL_ZIP" \
    --resume-path "$RESUME_PATH" \
    "$@"
