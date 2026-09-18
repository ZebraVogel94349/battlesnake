# Hisss + Battlesnake Blackout Agent

This repository contains two related Python projects:

- `src/hisss`: a high-performance Battlesnake simulator with C++/CUDA bindings.
- `bs-blackout-starter`: the Blackout competition agent, training code, evaluation tools, and HTTP server.

This project was used as our entry in the [Battlesnake Blackout competition](https://www.tnt.uni-hannover.de/bs-blackout-2026/) and placed seventh out of 66 entries.

We relied heavily on AI agents when writing the code in this repository.

The simulator is derived from [Hisss](https://github.com/ymahlau/hisss), and the agent project is based on the [Blackout starter kit](https://github.com/l-berg/battlesnake-blackout-starter). See [licensing and third-party notices](LICENSING.md), including the unresolved upstream simulator licensing status.

## Requirements

- Python 3.12 or newer
- CMake and a C++ compiler
- For CUDA training: an NVIDIA GPU, compatible driver, CUDA toolkit (`nvcc`), and a CUDA-enabled PyTorch installation
- CPU simulation and inference work without CUDA

## Installation

Create and activate a virtual environment, then install both projects in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pip install -e ./bs-blackout-starter
```

## Tests

```bash
python -m pytest test
python -m pytest bs-blackout-starter -rs
```

Model-dependent tests are marked `model` and skip with an explanation when the
optional deployment weights are absent. CUDA tests require a working GPU backend.

## Run an agent

For a first run without any model files:

```bash
python bs-blackout-starter/random_agent.py 8080
```

The repository excludes trained models, historical checkpoints, training runs,
logs, and evaluation output. To run PPO5, supply a compatible SB3 recurrent PPO
`.zip` export from the training code in this repository:

```bash
PPO5_MODEL_PATH=/absolute/path/to/your_model.zip python bs-blackout-starter/ppo5.py 8080
```

No source edit is needed. A resumable `.pt` training checkpoint is not an inference
export. The historical competition weights are not included, so this checkout
does not reproduce the submitted agent's playing strength by itself.

## Train from scratch

See [the training quickstart](bs-blackout-starter/TRAINING_FROM_SCRATCH.md) for a
checkpoint-free CUDA training command and export instructions. The versioned
`train_v*.sh`, `resume_*.sh`, `train.sh`, and best-response scripts record historical
experiments; they require checkpoints that are not included in this release.

## Local visualizer

```bash
python bs-blackout-starter/visualizer_server.py
```

Open `http://127.0.0.1:5000`. The visualizer binds to localhost with debugging
disabled. Start with the Random, Hungry, or Best agents; PPO agents need their
own compatible model files. The HTTP servers use Flask's development server;
public hosting needs a production deployment setup.

See [`bs-blackout-starter/README.md`](bs-blackout-starter/README.md) for the agent
code and historical training/inference documentation.
