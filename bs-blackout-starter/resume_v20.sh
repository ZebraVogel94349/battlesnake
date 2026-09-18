#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"
VARIANT_DIR="$SCRIPT_DIR/models/ppo_bs_lstm_cuda_v20_snake25_anchors_variants"
POOL_STATE="$SCRIPT_DIR/models/selfplay_pool_cuda_v20_snake25_anchors/nash_state.json"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON" >&2
    exit 2
fi

RESUME_PATH="${V20_RESUME_PATH:-}"
if [[ -z "$RESUME_PATH" ]]; then
    shopt -s nullglob
    CHECKPOINTS=(
        "$VARIANT_DIR"/ppo_bs_lstm_cuda_v20_snake25_anchors_steps_*_u*.pt
    )
    shopt -u nullglob
    if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
        echo "No resumable v20 checkpoint found in: $VARIANT_DIR" >&2
        exit 2
    fi
    RESUME_PATH="${CHECKPOINTS[-1]}"
fi

if [[ ! -f "$RESUME_PATH" ]]; then
    echo "V20 resume checkpoint not found: $RESUME_PATH" >&2
    exit 2
fi
RESUME_PATH="$(realpath -- "$RESUME_PATH")"
if [[ ! -f "$POOL_STATE" ]]; then
    echo "V20 pool state not found: $POOL_STATE" >&2
    exit 2
fi

"$PYTHON" - "$RESUME_PATH" "$POOL_STATE" "$SCRIPT_DIR" <<'PY'
import json
import re
import sys
from pathlib import Path

import hisss
import torch

checkpoint_path = Path(sys.argv[1])
pool_state_path = Path(sys.argv[2])
sys.path.insert(0, sys.argv[3])
payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
pool_state = json.loads(pool_state_path.read_text())

required = {"model_state_dict", "optimizer_state_dict", "total_steps", "update"}
missing = required.difference(payload)
if missing:
    raise RuntimeError(f"checkpoint is not resumable; missing keys: {sorted(missing)}")
if not payload["optimizer_state_dict"].get("state"):
    raise RuntimeError("checkpoint optimizer state is empty")

match = re.search(r"_steps_(\d+)_u(\d+)\.pt$", checkpoint_path.name)
if not match:
    raise RuntimeError(f"cannot parse checkpoint counters: {checkpoint_path.name}")
filename_steps, filename_update = map(int, match.groups())
checkpoint_steps = int(payload["total_steps"])
checkpoint_update = int(payload["update"])
pool_update = int(pool_state["update"])
if (filename_steps, filename_update) != (checkpoint_steps, checkpoint_update):
    raise RuntimeError("checkpoint filename and payload counters do not match")
if pool_update != checkpoint_update:
    raise RuntimeError(
        f"pool/checkpoint update mismatch: pool={pool_update}, checkpoint={checkpoint_update}"
    )
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch CUDA is not available")
if not hisss.cuda_available():
    raise RuntimeError(f"HISSS CUDA is not available: {hisss.cuda_last_error()}")

from train_cuda import ActorCritic, load_training_checkpoint, validate_cuda_all_seat_reset

device = torch.device("cuda")
model = ActorCritic().to(device).enable_channels_last()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, eps=1e-5, fused=True)
grad_scaler = torch.amp.GradScaler("cuda", enabled=True)
loaded_steps, loaded_update = load_training_checkpoint(
    str(checkpoint_path), model, optimizer, device, grad_scaler
)
if (loaded_steps, loaded_update) != (checkpoint_steps, checkpoint_update):
    raise RuntimeError("loaded checkpoint counters changed unexpectedly")

env = hisss.CudaBlackoutTorchVecEnv(
    8, seed=202020, device="cuda", obs_dtype=torch.float16
)
try:
    obs_all, legal_all = env.reset_all()
    validate_cuda_all_seat_reset(
        obs_all, legal_all, context="v20 resume preflight"
    )
finally:
    env.close()

print(f"[v20] checkpoint: {checkpoint_path}")
print(f"[v20] steps={checkpoint_steps} update={checkpoint_update} pool_update={pool_update}")
print(
    f"[v20] torch={torch.__version__} runtime_cuda={torch.version.cuda} "
    f"gpu={torch.cuda.get_device_name(0)}"
)
print("[v20] model, optimizer, AMP scaler, and CUDA rollout backend loaded")
PY

if [[ "${1:-}" == "--check" ]]; then
    (
        cd "$SCRIPT_DIR"
        PYTHON="$PYTHON" V20_RESUME_PATH="$RESUME_PATH" ./train_v20.sh --check
    )
    echo "[v20] preflight passed; training was not started"
    exit 0
fi

cd "$SCRIPT_DIR"
export PYTHON
export V20_RESUME_PATH="$RESUME_PATH"
exec ./train_v20.sh "$@"
