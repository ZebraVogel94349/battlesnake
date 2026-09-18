#!/usr/bin/env bash
set -euo pipefail

# V25 targeted finish branches from the strongest generalist found by the
# final sprint. It preserves those gains while concentrating the remaining
# wall-clock budget on the three measured regressions and the V25 baseline.

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
print(f"[v25] local CUDA hisss: {module}")
PY

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    exec "$PYTHON" train_cuda.py --help
fi

GPU_NAME="$("$PYTHON" -c \
    'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"
if [[ -z "${V25_SPEED_PROFILE:-}" ]]; then
    if [[ "$GPU_NAME" == *"RTX 4090"* ]]; then
        V25_SPEED_PROFILE=rtx4090
    else
        V25_SPEED_PROFILE=standard
    fi
fi

case "$V25_SPEED_PROFILE" in
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
        echo "Unknown V25_SPEED_PROFILE: $V25_SPEED_PROFILE" >&2
        exit 2
        ;;
esac

V25_NUM_ENVS="${V25_NUM_ENVS:-$DEFAULT_NUM_ENVS}"
V25_BATCH_SIZE="${V25_BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
V25_EPOCHS="${V25_EPOCHS:-3}"
V25_UPDATES="${V25_UPDATES:-3250}"
V25_START_UPDATE="${V25_START_UPDATE:-76750}"
V25_STOP_AFTER_UPDATE=$((V25_START_UPDATE + V25_UPDATES))
V25_PEAK_LR="${V25_PEAK_LR:-9.0e-7}"
V25_LR_FLOOR="${V25_LR_FLOOR:-2.5e-7}"
V25_ACTOR_REFERENCE_L2="${V25_ACTOR_REFERENCE_L2:-0.025}"
V25_EVAL_GAMES="${V25_EVAL_GAMES:-512}"
V25_EVAL_NUM_ENVS="${V25_EVAL_NUM_ENVS:-256}"
V25_PROMOTION_GAMES="${V25_PROMOTION_GAMES:-512}"

V25_GENERALIST_CHECKPOINT="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v25_final_sprint_variants/ppo_bs_lstm_cuda_v25_final_sprint_steps_015368192000_u00076750.pt"
V25_BASELINE_CHECKPOINT="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v25_recovery_variants/ppo_bs_lstm_cuda_v25_recovery_steps_014974976000_u00075250.pt"
V25_SOURCE_CHECKPOINT="${V25_SOURCE_CHECKPOINT:-$V25_GENERALIST_CHECKPOINT}"
V24_FINAL_CHECKPOINT="${V25_V24_FINAL_CHECKPOINT:-$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v24_general_strength_latest.pt}"
V24_CHAMPION="${V25_V24_CHAMPION:-$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v24_general_strength_champion.pt}"
V24_CHAMPION_UPDATE="${V25_V24_CHAMPION_UPDATE:-69900}"
# Update 75,250 remains the protected direct-match baseline. Update 76,750 is
# the learner/reference because it gained +0.016 on the fixed suite and +0.022
# on heuristics; the targeted finish must repair its few regressions before it
# can replace the baseline.
V25_INITIAL_CHAMPION="${V25_INITIAL_CHAMPION:-$V25_BASELINE_CHECKPOINT}"
V25_INITIAL_CHAMPION_UPDATE="${V25_INITIAL_CHAMPION_UPDATE:-75250}"
V21_RETENTION="${V25_RETENTION_CHECKPOINT:-$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v21_balanced_duels_variants/ppo_bs_lstm_cuda_v21_balanced_duels_steps_019300352000_u00087500.pt}"
V24_POOL="$SCRIPT_DIR/models/selfplay_pool_cuda_v24_general_strength"

# Final V24 Nash scores identify u60,951 and u70,251 as the two hardest
# policies.  The remaining snapshots cover late, pre-promotion, mid-run, and
# early V24 behavior so training pressure cannot collapse onto one counter.
V24_HARD_60951="$V24_POOL/snap_011226578944_u00060951.pt"
V24_HARD_70251="$V24_POOL/snap_013664518144_u00070251.pt"
V24_RECENT_71151="$V24_POOL/snap_013900447744_u00071151.pt"
V24_PRECHAMP_69351="$V24_POOL/snap_013428588544_u00069351.pt"
V24_MID_66551="$V24_POOL/snap_012694585344_u00066551.pt"
V24_EARLY_56251="$V24_POOL/snap_009994502144_u00056251.pt"

for required in \
    "$V25_SOURCE_CHECKPOINT" \
    "$V25_INITIAL_CHAMPION" \
    "$V24_FINAL_CHECKPOINT" \
    "$V24_CHAMPION" \
    "$V21_RETENTION" \
    "$V24_HARD_60951" \
    "$V24_HARD_70251" \
    "$V24_RECENT_71151" \
    "$V24_PRECHAMP_69351" \
    "$V24_MID_66551" \
    "$V24_EARLY_56251"; do
    if [[ ! -f "$required" ]]; then
        echo "Required V25 input not found: $required" >&2
        exit 2
    fi
done

"$PYTHON" - \
    "$V25_SOURCE_CHECKPOINT" "$V25_START_UPDATE" \
    "$V25_INITIAL_CHAMPION" "$V25_INITIAL_CHAMPION_UPDATE" \
    "$V24_CHAMPION" "$V24_CHAMPION_UPDATE" \
    "$V24_FINAL_CHECKPOINT" 71250 \
    "$V21_RETENTION" 87500 \
    "$V24_HARD_60951" 60951 \
    "$V24_HARD_70251" 70251 \
    "$V24_RECENT_71151" 71151 \
    "$V24_PRECHAMP_69351" 69351 \
    "$V24_MID_66551" 66551 \
    "$V24_EARLY_56251" 56251 <<'PY'
import sys
import torch

arguments = sys.argv[1:]
source_path, source_expected_raw = arguments[:2]
source_expected = int(source_expected_raw)
source = torch.load(source_path, map_location="cpu", weights_only=False)
if int(source.get("update", -1)) != source_expected:
    raise SystemExit(
        f"V25 source update is {source.get('update')}, expected {source_expected}"
    )
if not isinstance(source.get("model_state_dict"), dict):
    raise SystemExit("V25 source checkpoint has no model state")
optimizer = source.get("optimizer_state_dict")
if not isinstance(optimizer, dict) or not optimizer.get("state"):
    raise SystemExit("V25 source checkpoint has no complete optimizer state")

validated = []
for path, expected_raw in zip(arguments[2::2], arguments[3::2]):
    expected = int(expected_raw)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("update", -1)) != expected:
        raise SystemExit(
            f"V25 anchor {path} is update {payload.get('update')}, expected {expected}"
        )
    if not isinstance(payload.get("model_state_dict"), dict):
        raise SystemExit(f"V25 anchor {path} has no model state")
    validated.append(expected)

print(
    f"[v25] validated complete source update={source_expected}, V25 league "
    f"baseline update={validated[0]}, protected V24 champion "
    f"update={validated[1]}, and {len(validated) - 2} elite anchors"
)
PY

V25_OUTPUT_ROOT="${V25_OUTPUT_ROOT:-$SCRIPT_DIR}"
# A fresh namespace prevents the completed sprint's optimizer and population
# from leaking into the targeted branch. Historical outputs stay untouched.
V25_POOL_DIR="$V25_OUTPUT_ROOT/models/selfplay_pool_cuda_v25_targeted_finish"
V25_VARIANT_DIR="$V25_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v25_targeted_finish_variants"
V25_LEAGUE_CHAMPION="$V25_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v25_targeted_finish_champion.pt"
V25_CHAMPION_ZIP="$V25_OUTPUT_ROOT/ppo_bs_lstm_cuda_v25_targeted_finish_champion.zip"
V25_LATEST="$V25_OUTPUT_ROOT/models/ppo_bs_lstm_cuda_v25_targeted_finish_latest.pt"
V25_FINAL_ZIP="$V25_OUTPUT_ROOT/ppo_bs_lstm_cuda_v25_targeted_finish.zip"
V25_RUN_DIR="$V25_OUTPUT_ROOT/runs/ppo_bs_lstm_cuda_v25_targeted_finish"

RESUME_PATH="${V25_RESUME_PATH:-}"
if [[ -z "$RESUME_PATH" ]]; then
    shopt -s nullglob
    CHECKPOINTS=(
        "$V25_VARIANT_DIR"/ppo_bs_lstm_cuda_v25_targeted_finish_steps_*_u*.pt
    )
    shopt -u nullglob
    if [[ ${#CHECKPOINTS[@]} -gt 0 ]]; then
        RESUME_PATH="${CHECKPOINTS[-1]}"
    fi
fi

if [[ -n "$RESUME_PATH" ]]; then
    if [[ ! -f "$RESUME_PATH" || ! -d "$V25_POOL_DIR" ]]; then
        echo "V25 resume checkpoint or population directory is missing." >&2
        exit 2
    fi
    START_ARGS=(--resume-path "$RESUME_PATH")
    echo "[v25] resuming V25: $RESUME_PATH"
else
    shopt -s nullglob
    EXISTING_V25=(
        "$V25_POOL_DIR"/*
        "$V25_VARIANT_DIR"/*
        "$V25_RUN_DIR"/*
    )
    shopt -u nullglob
    if [[ ${#EXISTING_V25[@]} -gt 0 || -e "$V25_LATEST" || -e "$V25_LEAGUE_CHAMPION" ]]; then
        echo "Partial V25 state exists but no resumable checkpoint was found." >&2
        echo "Nothing was overwritten; inspect $V25_OUTPUT_ROOT before retrying." >&2
        exit 2
    fi
    mkdir -p "$V25_POOL_DIR" "$V25_VARIANT_DIR" "$V25_RUN_DIR"
    START_ARGS=(--warm-start-path "$V25_SOURCE_CHECKPOINT")
    echo "[v25] branching targeted finish from update $V25_START_UPDATE with fresh Adam state"
fi

echo "[v25] targeted-finish profile=$V25_SPEED_PROFILE gpu=${GPU_NAME:-unknown} " \
     "envs=$V25_NUM_ENVS batch=$V25_BATCH_SIZE epochs=$V25_EPOCHS " \
     "phase_updates=$V25_UPDATES copies=0.60 duels=0.28 " \
     "lr=$V25_PEAK_LR->$V25_LR_FLOOR actor_reference_l2=$V25_ACTOR_REFERENCE_L2"

if [[ "${1:-}" == "--check" ]]; then
    echo "[v25] start mode: ${START_ARGS[*]}"
    echo "[v25] phase schedule: update $V25_START_UPDATE -> $V25_STOP_AFTER_UPDATE"
    echo "[v25] league baseline: $V25_INITIAL_CHAMPION"
    echo "[v25] protected V24 champion: $V24_CHAMPION"
    echo "[v25] outputs: $V25_OUTPUT_ROOT"
    exit 0
fi

exec "$PYTHON" train_cuda.py \
    --num-envs "$V25_NUM_ENVS" \
    --n-steps 128 \
    --seq-len 128 \
    --batch-size "$V25_BATCH_SIZE" \
    --epochs "$V25_EPOCHS" \
    --updates "$V25_UPDATES" \
    --stop-after-update "$V25_STOP_AFTER_UPDATE" \
    --schedule-origin-update "$V25_START_UPDATE" \
    --rollout-obs-dtype float16 \
    --amp-dtype float16 \
    --channels-last \
    --compile-training-cnn \
    --compile-rollout-ops \
    --compile-mode "$COMPILE_MODE" \
    --tensorboard-log-dir "$V25_RUN_DIR" \
    --tensorboard-update-interval 10 \
    --lr "$V25_PEAK_LR" \
    --continuation-lr "$V25_PEAK_LR" \
    --lr-schedule cosine \
    --lr-floor "$V25_LR_FLOOR" \
    --lr-decay-updates "$V25_UPDATES" \
    --target-kl 0.0025 \
    --kl-stop-mode minibatch \
    --gamma 1.0 \
    --gae-lambda 0.97 \
    --clip-range 0.10 \
    --ent-coef 0.00005 \
    --ent-coef-final 0.00001 \
    --ent-decay-updates "$V25_UPDATES" \
    --actor-reference-path "$V25_SOURCE_CHECKPOINT" \
    --actor-reference-l2-coef "$V25_ACTOR_REFERENCE_L2" \
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
    --training-duel-probability 0.28 \
    --selfplay-update-interval 100 \
    --pool-dir "$V25_POOL_DIR" \
    --pool-anchor "v24_champion_69900=$V24_CHAMPION" \
    --pool-anchor "v24_final_71250=$V24_FINAL_CHECKPOINT" \
    --pool-anchor "v24_hard_60951=$V24_HARD_60951" \
    --pool-anchor "v24_hard_70251=$V24_HARD_70251" \
    --pool-anchor "v24_recent_71151=$V24_RECENT_71151" \
    --pool-anchor "v24_prechamp_69351=$V24_PRECHAMP_69351" \
    --pool-anchor "v24_mid_66551=$V24_MID_66551" \
    --pool-anchor "v24_early_56251=$V24_EARLY_56251" \
    --pool-anchor "source_87500=$V21_RETENTION" \
    --pool-anchor-target v24_champion_69900=0.518 \
    --pool-anchor-target v24_final_71250=0.492 \
    --pool-anchor-target v24_hard_60951=0.511 \
    --pool-anchor-target v24_hard_70251=0.482 \
    --pool-anchor-target v24_recent_71151=0.517 \
    --pool-anchor-target v24_prechamp_69351=0.511 \
    --pool-anchor-target v24_mid_66551=0.481 \
    --pool-anchor-target v24_early_56251=0.568 \
    --pool-anchor-target source_87500=0.622 \
    --pool-anchor-priority-exponent 1.0 \
    --pool-anchor-min-weight 0.05 \
    --pool-anchor-score-ema 0.50 \
    --max-pool-checkpoints 32 \
    --pool-active-checkpoints 13 \
    --pool-active-rotation-interval 50 \
    --pool-checkpoint-weight 0.96 \
    --pool-best-weight 0.0 \
    --pool-hungry-weight 0.0 \
    --pool-random-weight 0.0 \
    --pool-heuristic-weight hunter=0.010 \
    --pool-heuristic-weight snake25=0.010 \
    --pool-heuristic-weight snake25_duelist=0.010 \
    --pool-duel-opponent-weight league_champion=0.45 \
    --pool-duel-opponent-weight v24_champion_69900=0.20 \
    --pool-duel-opponent-weight v24_early_56251=0.10 \
    --pool-duel-opponent-weight source_87500=0.10 \
    --pool-duel-opponent-weight snake25_duelist=0.05 \
    --pool-anchor-uniform-floor 0.30 \
    --pool-latest-probability 0.04 \
    --pool-anchor-probability 0.90 \
    --pool-nash-history-size 16 \
    --pool-nash-iterations 5000 \
    --pool-nash-exploration 0.10 \
    --pool-nash-score-half-life-updates 1000 \
    --pool-champion-min-weight 0.30 \
    --pool-deduplicate-champion \
    --pool-min-action-disagreement 0.025 \
    --pool-diversity-probe-size 2048 \
    --pool-deterministic-probability 0.80 \
    --pool-copies-probability 0.60 \
    "${PARALLEL_INFERENCE_ARGS[@]}" \
    --pool-load-existing \
    --eval-interval 250 \
    --eval-games "$V25_EVAL_GAMES" \
    --eval-num-envs "$V25_EVAL_NUM_ENVS" \
    --eval-max-turns 2000 \
    --eval-seed 351020 \
    --eval-seed-stride 0 \
    --eval-layout copies \
    --eval-opponents 6 \
    --league-champion-path "$V25_LEAGUE_CHAMPION" \
    --league-champion-zip-path "$V25_CHAMPION_ZIP" \
    --league-initial-champion-path "$V25_INITIAL_CHAMPION" \
    --league-promotion-games "$V25_PROMOTION_GAMES" \
    --league-promotion-seed 117733 \
    --league-promotion-seed 352117 \
    --league-promotion-seed 615043 \
    --league-promotion-seed 748921 \
    --league-promotion-seed 904237 \
    --league-promotion-layout copies \
    --league-promotion-layout solo-pair \
    --league-promotion-layout true-duel \
    --league-promotion-threshold 0.510 \
    --league-general-improvement 0.003 \
    --league-max-anchor-regression 0.015 \
    --league-min-promotion-interval 250 \
    --league-guard-score v24_champion_69900=0.510 \
    --league-guard-score v24_final_71250=0.45 \
    --league-guard-score v24_hard_60951=0.45 \
    --league-guard-score v24_hard_70251=0.45 \
    --league-guard-score v24_recent_71151=0.45 \
    --league-guard-score v24_prechamp_69351=0.45 \
    --league-guard-score v24_mid_66551=0.45 \
    --league-guard-score v24_early_56251=0.45 \
    --league-guard-score source_87500=0.52 \
    --league-guard-score hunter=0.64 \
    --league-guard-score snake25=0.69 \
    --league-guard-score snake25_duelist=0.68 \
    --sb3-checkpoint-interval 100 \
    --sb3-checkpoint-dir "$V25_VARIANT_DIR" \
    --sb3-checkpoint-prefix ppo_bs_lstm_cuda_v25_targeted_finish \
    --max-sb3-checkpoints 12 \
    --fused-adam \
    --allow-tf32 \
    --seed 252551 \
    --save-path "$V25_LATEST" \
    --sb3-save-path "$V25_FINAL_ZIP" \
    "${START_ARGS[@]}" \
    "$@"
