# Training from scratch

The historical competition weights and training checkpoints are not included.
The commands below start with randomly initialized weights and need no earlier
models. They are a setup check and starting point, not a recipe that reproduces
the competition results.

## Environment

Follow the [root installation instructions](../README.md#installation), activate
that environment, and run the following commands from the repository root.
CUDA training needs an NVIDIA GPU, a compatible driver, the CUDA toolkit with
`nvcc`, and CUDA-enabled PyTorch. The CPU-only simulator build supports inference
and CPU simulation but cannot run `train_cuda.py`.

To rebuild the local simulator with CUDA required:

```bash
CMAKE_ARGS="-DHISSS_REQUIRE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native" \
  python -m pip install --force-reinstall --no-deps -e .
python -c "import torch, hisss; assert torch.cuda.is_available(), 'PyTorch CUDA unavailable'; assert hisss.cuda_available(), hisss.cuda_last_error()"
```

The `native` architecture setting requires CMake 3.24 or newer. Set `CUDACXX` to
the full path of `nvcc` if it is not on `PATH`.

## First training and export

Choose a new output directory for each independent run. From the repository root:

```bash
mkdir -p bs-blackout-starter/models/from_scratch
python bs-blackout-starter/train_cuda.py \
  --num-envs 32 --n-steps 32 --seq-len 16 --batch-size 256 \
  --epochs 1 --updates 2 --opponent-mode random --eval-interval 0 \
  --save-path bs-blackout-starter/models/from_scratch/smoke.pt \
  --sb3-save-path bs-blackout-starter/models/from_scratch/smoke.zip
```

This small smoke run checks simulation, training, checkpoint saving, and SB3
export. Its policy is essentially untrained. It does not load historical models
or use a self-play pool. To train longer, increase `--updates` and tune the rollout
and batch sizes for your GPU; `--opponent-mode curriculum` adds heuristic
opponents. `python bs-blackout-starter/train_cuda.py --help` lists the options.

The `.pt` file includes resumable training state (`--resume-path`); the `.zip`
export is the file consumed by PPO5. Run the exported policy on CPU with:

```bash
PPO5_MODEL_PATH="$PWD/bs-blackout-starter/models/from_scratch/smoke.zip" \
  PPO_DEVICE=cpu python bs-blackout-starter/ppo5.py 8080
```

Only load checkpoints you trust. Train and evaluate a policy for substantially
longer before treating it as a competitive agent.

## Historical scripts

`train.sh`, `train_v*.sh`, `resume_*.sh`, and `train_ppo5_best_response.sh` preserve
the settings of past experiments. Their anchors and continuation checkpoints
are intentionally absent. They are not entry points for a fresh checkout.
Without the historical weights, their reported results cannot be reproduced
simply by running those scripts.
