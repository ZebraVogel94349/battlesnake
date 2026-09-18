# Battlesnake Blackout Competition Agent 🐍

This directory contains our competition agents, training code, and evaluation
tools, based on the [official Blackout starter kit](https://github.com/l-berg/battlesnake-blackout-starter).
It is our modified competition entry, not the official starter distribution.
For installation, use the [repository quickstart](../README.md); for new training
runs, use [Training from scratch](TRAINING_FROM_SCRATCH.md).

Historical training and inference reports below refer to experiments and model
files that are not included in this source release. The licensing status and
upstream credits are recorded in [LICENSING.md](../LICENSING.md).

In its standard form, Battlesnake is a multiplayer game of perfect information. **Battlesnake Blackout** introduces a challenging *fog of war* mechanic. Your agent will only perceive the immediate area surrounding its head. You will need to build an AI capable of opponent modeling, trap setting, long-term strategic planning, and memory management under uncertainty.

### Attribution
This competition and starter code are based on the API and design of the popular multiplayer game [Battlesnake](https://play.battlesnake.com/). While we are not officially affiliated with the play.battlesnake.com team, we rely on their excellent API architecture and this repo is based on their [Python Starter Project](https://github.com/BattlesnakeOfficial/starter-snake-python). We encourage participants to check out their platform! If you want to use another programming language, consider adapting one of their [starter repos](https://docs.battlesnake.com/starter-projects).

#### API Changes compared to standard Battlesnake
Battlesnake Blackout extends the existing Battlesnake API:
* `game.ruleset.settings.viewRadius`: New property that defines the number of tiles your snake can see from its head.
* `board.snakes.body`: Can now contain None values to represent one or more unseen body segments.
* `board.snakes.length`: No longer provides an exact count of an opponent's length if any of their body segments are outside your view radius.

---

## Getting Started

This repository contains everything you need to get a basic agent up and running.
*To participate, you will need to host your agent as a web server that responds to our game state requests. Consider using a cloud service or a port forwarding tool like [ngrok](https://ngrok.com/).*

### 1. Prerequisites

Python 3.12 or newer, CMake, and a C++ compiler. CUDA training additionally needs
an NVIDIA GPU, CUDA toolkit, and CUDA-enabled PyTorch.

### 2. Installation

From the root of this repository, follow the [installation instructions](../README.md#installation)
to install both the local simulator and this agent project in one environment.
Then enter this directory:

```bash
cd bs-blackout-starter
```

### 3. Running Your Agent
Start up the `RandomAgent` located in [`random_agent.py`](random_agent.py) by running the script and specifying the port, for example:
```bash
python random_agent.py 8080
```

### 4. Verify
To verify that the Battlesnake Blackout engine can reach your agent, navigate to `http://YOUR_IP:YOUR_PORT/`. This should show some info on your agent, for example:
```json
{"author":"Chaos Itself","color":"#32CD32"}
```
Success! Your agent is now listening for requests from the Battlesnake Blackout evaluation engine. To reduce concurrency issues, the engine will only ever run one game at a time.

### PPO-guided MCTS agent

The experimental MCTS inference entry point combines the recurrent PPO actor with a
deadline-bound information-set MCTS for Blackout:

This requires a compatible model export (not included):

```bash
PPO_MODEL_PATH=/absolute/path/to/your_model.zip python ppo_mcts.py 8080
```

The final PPO5 entry point is documented in the [root quickstart](../README.md#run-an-agent).

It uses a 400 ms request-relative budget by default, evaluates independent
16-simulation root trees, and changes PPO's move only after a strict majority
across at least three complete trees inside the soft-safe action set. PPO4
remains the anytime and safety fallback.
See [`INFERENCE_MCTS.md`](INFERENCE_MCTS.md) for the search model, deadline
controls, evaluator commands, limitations, and measured holdout results.

---

## Meet the `HungryAgent`

The [`hungry_agent.py`](hungry_agent.py) script includes a basic toolkit for handling the imperfect information mechanics of Battlesnake Blackout. 

Here is what it does out of the box:
* **Memory Management:** Because of the fog of war, you only see food when it is within your vision radius, and globally for a single tick when it spawns! The agent utilizes an `AgentState` dictionary to remember the coordinates of food it has seen until it either eats it or visually confirms it is gone.
* **Obstacle Mapping:** It creates a NumPy-based boolean grid of the board, marking visible snake body parts as impassable obstacles.
* **A\* Pathfinding:** It uses an A* search to calculate the shortest safe path to the nearest known food source.
* **Fallback Logic:** If no food is known or reachable, the agent defaults to chasing its own tail. If that fails, it picks a random safe direction to avoid immediate elimination.

---

## Local Testing & Simulation

To test your agent locally and train your algorithms, we provide **Hisss**, a high-performance C++ Battlesnake simulator with convenient Python bindings. 

You can find the simulator, documentation, and installation instructions here: 
🔗 **[Hisss Simulator Repository](https://github.com/ymahlau/hisss)**

### Historical CUDA training on 8 GiB GPUs

The following records the original experiments. These scripts depend on
unpublished checkpoints; use [Training from scratch](TRAINING_FROM_SCRATCH.md)
for a fresh checkout.

`train.sh` keeps FP16 enabled. Its default compiler mode avoids private CUDA
Graph memory pools, and completed rollout buffers are released before the next
rollout is allocated. This prevents two large buffers from overlapping and
substantially lowers intermittent VRAM peaks. `--compile-mode reduce-overhead`
is still available for GPUs with more memory.

Use `RESUME_PATH=models/<checkpoint>.pt ./train.sh` to continue a full training
checkpoint. Continuations reuse the saved learning rate and never increase it
unless `--continuation-lr` is explicitly supplied. The source policy is pinned
in the self-play pool, and TensorBoard records periodic comparisons under
`eval/*`. V15 retains up to 100 behaviorally distinct policies while at most 16
neural opponents are active in one rollout. The active set preferentially keeps
every model from the latest evaluation plus the strongest Nash-supported
opponents.

V15 was trained from randomly initialized learner weights. The strongest
completed v14 evaluation checkpoint (update 4500) was both a permanent
evaluated population member and the initial deployable league champion. A
promotion had to beat that champion over three paired seeds and retain
realistic per-reference floors derived from v14, preventing either self-play
progress or an impossible legacy floor from hiding regressions.

V15 uses Nash-weighted PFSP. Each evaluation measures the current learner
against every fixed v7/v9/v13/v14 reference, four rotating self-play policies,
and the CUDA `best`, `hungry`, and `random` heuristics. Successive evaluations
form the rows of a rolling win-rate matrix; opponents form its columns. A
zero-sum minimax solve chooses the opponent Nash distribution that is hardest
for the retained learner rows, then mixes in 10% uniform exploration. This
distribution directly controls training assignments. Matrix value, weights,
and actual assignment counts are logged under `selfplay/nash/*`,
`selfplay/nash_weight/*`, and `selfplay/*_assignments`; heuristic comparisons
are under `eval/heuristics/*`.

For the V16 continuation from update 33,000, the configured Best/Hungry/Random
weights are guaranteed floors of 15%/10%/5%. The remaining 70% is distributed
by the full Nash solution across both neural and heuristic opponents, so a hard
heuristic can receive more mass but can no longer disappear behind a slightly
harder historical checkpoint. The actual constrained mixture is logged under
`selfplay/training_weight/*`; `selfplay/nash_weight/*` remains the unconstrained
equilibrium for diagnosis.

Every 50 updates, a candidate self-play snapshot is compared with every
population checkpoint on 512 rollout observations. It enters the population
only if its greedy actions disagree with its nearest neighbor on at least 8%
of probes. This stops near-identical snapshots from crowding out strategically
different policies. The first 250 V15 updates used a scripted curriculum, and
that run used exactly 23,000 x 1,024 x 128 environment steps.

V16 extends V15 to absolute update 46,000. The current `train.sh` resumes the
periodic V16 checkpoint at update 33,000, restoring learner weights, optimizer
moments, AMP scaler, counters, and the saved constant `5e-6` learning rate.
Population snapshots newer than the selected resume checkpoint are moved into
a recoverable quarantine directory, so restarting an earlier checkpoint
cannot accidentally train against a policy from its own future.

From update 27,000 onward, the environment objective matches the tournament:
first place receives 2 points, second receives 1, and third/fourth both receive
0. Points are awarded incrementally when a top-two finish and then a win
become guaranteed, while their episode sum remains exactly 2/1/0/0. The
undiscounted `gamma=1.0` objective prevents a quick second place from becoming
more valuable than a late win. Evaluation games may run for 2,000 turns
instead of 1,000; on the update 27,000 self-match this reduced artificial
timeout draws from 25.4% to 0.4%, with no further change at 4,000 turns.
Read-only opponent and evaluation forwards use inference mode, which preserves
actions/results while reducing their overhead.

Every 250 updates, `train.sh` writes a full resumable `.pt` checkpoint and a
ppo.py-compatible ZIP to `models/ppo_bs_lstm_cuda_v16_nash_variants/`. Filenames
contain both the cumulative step and update number. Use the `.pt` with
`RESUME_PATH` to preserve the optimizer and AMP state; use the ZIP for inference
or comparisons with `compare_ppo_cuda.py`. Set `--max-sb3-checkpoints` to a
positive number to prune old checkpoint pairs; the V16 default `0` keeps every
variant. A normal `./train.sh` starts at update 33,000. To select another
checkpoint, use `RESUME_PATH=<v16-variant.pt> ./train.sh`; the absolute
`--stop-after-update 46000` prevents training beyond the intended V16 endpoint.

### V18: V16 extension

V17's from-scratch branch was abandoned because it could not reach V16's
strength within the available time. `./train_v18.sh` instead branches from the
full V16 update-43,750 checkpoint. That checkpoint is bit-identical to the
promoted V16 champion and includes its Adam moments, AMP scaler, counters, and
constant `5e-6` learning rate; V18 is therefore a true continuation rather than
a weights-only warm start.

The first invocation hardlinks V16's immutable population into a separate V18
pool and creates a V18-named hardlink to the resumable source checkpoint. V16
remains byte-for-byte untouched. V18 trains 2,250 additional updates to absolute
update 46,000: 294,912,000 new environment steps and 6,029,312,000 cumulative
steps. The V16 champion is the initial deployment champion and receives at least
15% of training assignments. Nash scores age toward 0.5 over a 1,000-update
half-life, and promotion combines the symmetric 2-vs-2 and 1-vs-3/3-vs-1
layouts.

`V18_SPEED_PROFILE=fast` is the default on ordinary GPUs. It keeps V16's
1,024 × 128 rollout and four PPO passes, raises the minibatch to 8,192, uses
the eight highest-priority frozen opponents per rollout, and evaluates every
500 updates. `quality` reproduces V16's batch size, 16 active opponents, and
250-update evaluation cadence. `turbo` uses three PPO passes, four active
opponents, and less frequent evaluations without reducing environment samples
per update.

On an RTX 5090, `train_v18.sh` automatically selects `rtx5090`: 3,072
environments, a 24,576-sample PPO minibatch, and four independent frozen
opponents on parallel CUDA streams. Four larger opponent batches avoid the
launch-bound eight-model rollout observed on wide GPUs. The profile uses eager
cuDNN by default because lazy Inductor/CUDA-Graph compilation can leave the GPU
idle for tens of seconds on some Blackwell software stacks; set
`V18_TORCH_COMPILE=1` to opt back in. Its epoch-level KL check removes a CPU/GPU
synchronization from every minibatch. This profile collects three times as many
samples per update; use
`V18_SPEED_PROFILE=fast` explicitly when update-for-update comparability with
the established continuation matters more than throughput.

The periodic console line reports `train_steps/s`, `rollout_s`, and `ppo_s`.
`train_steps/s` covers only rollout plus PPO work; the older cumulative
`steps/s` also includes evaluation, checkpoint export, and compiler warm-up
pauses and is therefore not a reliable GPU-utilization benchmark. The 5090
profile also prints a `[timing]` line after every update, separating rollout,
PPO, pool snapshot, and unaccounted host time. Evaluation and periodic export
receive their own timing lines. On this profile TensorBoard metrics are reduced
on the GPU in one batch and written every ten updates to
`/tmp/hisss-v18-tensorboard`, with a larger asynchronous queue. This avoids
blocking training on slow cloud-workspace event-file writes. Set
`V18_TENSORBOARD_LOG_DIR` to retain those diagnostics elsewhere when that
filesystem can sustain small asynchronous writes.

For native Blackwell simulator kernels, build Hisss on the 5090 host with CUDA
Toolkit 12.8 or newer. The CMake build then includes `sm_120` automatically.
After copying the repository to that server, rebuild the extension from the
repository root before starting training:

```bash
.venv/bin/python -m pip install -e .
```

```bash
# One-time V16 -> V18 branch and recommended training profile.
./train_v18.sh

# Resume an interrupted V18 checkpoint.
RESUME_PATH=models/ppo_bs_lstm_cuda_v18_v16_extension_variants/NAME.pt ./train_v18.sh

# Deadline-oriented alternative.
V18_SPEED_PROFILE=turbo RESUME_PATH=models/ppo_bs_lstm_cuda_v18_v16_extension_variants/NAME.pt ./train_v18.sh
```

The script refuses a second bootstrap once any V18 state exists. It keeps the
newest 16 resumable/export pairs. Bootstrap evaluations at local updates 1, 10,
and 20 run only for a genuinely new update-zero learner, so this mature
continuation does not repeat them after a restart.
`ppo.py` also accepts `PPO_ACTION_MASK_MODE=observable-hard` for models trained
with that optional mode; its compatibility default remains `observable`.

### V19: real-opponent heuristic anchors

`./train_v19.sh` continues the completed v18 run locally and adds five
log-derived native CUDA opponents: Forager, Hunter, Territorial, Edge-Trapper,
and Survivor. Their combined 25% floor is split evenly, while each remains a
separate Nash payoff column and evaluation series. The evidence window,
measured behavior, and profile definitions are documented in
[`OPPONENT_HEURISTICS.md`](OPPONENT_HEURISTICS.md).

Copy the completed v18 variant checkpoints, champion, and
`models/selfplay_pool_cuda_v18_v16_extension/` from the server into the same
paths here, rebuild Hisss, then run:

```bash
# Refuses the current bootstrap-only update 43,750 and expects completed v18
# update 69,000 by default.
./train_v19.sh

# Resume an interrupted local v19 run without cloning v18 again.
V19_RESUME_PATH=models/ppo_bs_lstm_cuda_v19_real_anchors_variants/NAME.pt ./train_v19.sh
```

Use `V19_SOURCE_CHECKPOINT=/path/to/final-v18.pt` if the source is elsewhere.
If the server run intentionally stops before update 69,000, set
`V18_REQUIRED_UPDATE` to that actual final update. Local rollout size, PPO
batch, active neural opponents, update count, and evaluation sizes can be
adjusted with the `V19_*` variables at the top of the launcher.

### V20: Snake-25 anchors

`./train_v20.sh` waits for the completed v19 artifacts and then continues the
final resumable v19 checkpoint for exactly 30,000 additional updates. It will
not start while v19 is still running: by default it requires update 70,000,
the final v19 ZIP, the latest training checkpoint, league champion, and Nash
pool. Set `V19_INPUT_ROOT` when the server output is copied into a separate
directory.

The main scripted opponents are Best, Hunter, the log-derived Snake-25 clone,
Snake25 Interceptor, and Snake25 Denier. Forager, Territorial, Edge-Trapper,
and Survivor remain only as 0.25% diversity floors each. Exact weights and the
Snake-25 evidence are documented in
[`OPPONENT_HEURISTICS.md`](OPPONENT_HEURISTICS.md).

```bash
# Fresh v19 -> v20 continuation after v19 has fully finished.
./train_v20.sh

# Resume v20; --stop-after-update prevents adding another 30,000 updates.
V20_RESUME_PATH=models/ppo_bs_lstm_cuda_v20_snake25_anchors_variants/NAME.pt ./train_v20.sh
```

Evaluations run every 250 updates and are written to
`runs/ppo_bs_lstm_cuda_v20_snake25_anchors`. In TensorBoard, every heuristic
has its own `eval/heuristics/<label>/...` curves alongside anchor/history
evaluation, Nash weights, and actual assignment counts.

### V21: balanced anchors and true Snake-25 duels

V20 was intentionally stopped after its complete periodic checkpoint at update
78,500 (`17,104,896,000` cumulative steps). `./train_v21.sh` branches exactly
from that resumable checkpoint, including Adam and AMP state; it rejects both
older and newer V20 inputs. It adds real two-snake CUDA resets to 25% of rollout
episodes. Of those true duels, 50% use the native log-derived
`snake25_duelist`, 30% use a recurrent Behavioral Clone learned from real
Snake-25 moves, and 20% retain the broad league mixture.

`train_snake25_bc.py` infers actions only from consecutive visible Snake-25
head positions. Because the requests were recorded from our snake's viewpoint,
it reconstructs Snake-25-centred proxy observations and keeps unobserved cells
fogged. It fine-tunes a copy of the V20 actor with game-level train/validation
splits, extra weight for duel moves, rotation augmentation, and a distillation
penalty toward V20. The result is a frozen opponent anchor; its weights are not
copied into the V21 learner. The dataset is cached after its first build.

The neural side of the population is protected against broad regression:
fixed anchors rotate through the active working set, 30% total training mass is
split evenly over the active anchors, below-target anchors receive additional
PFSP/Nash pressure, and champion promotion has an individual floor for every
anchor. A true-duel comparison is part of the candidate-versus-champion gate.

V21 keeps the hot training mix deliberately compact: Hunter, Snake-25, and the
Snake-25 Duelist are the only native profile variants sampled, Hungry's tiny
floor is removed, and at most four frozen neural policies are active in one
rollout. The removed profiles remain represented by the fixed neural anchor
population. Learner action masks and mobility shaping share one computation per
state; the launcher fuses that fixed-shape computation with `torch.compile`.

On an RTX 4090 the launcher automatically selects `rtx4090`: 2,048 environments,
32,768-sample PPO minibatches, three PPO passes, epoch-level KL stopping, parallel
frozen-policy inference, and `reduce-overhead` compilation. This is the
throughput profile; set `V21_SPEED_PROFILE=standard` for the previous 1,024-env,
four-pass training shape. `V21_NUM_ENVS`, `V21_BATCH_SIZE`, `V21_EPOCHS`, and
`V21_ACTIVE_CHECKPOINTS` can override the individual dimensions.

```bash
# Read-only validation: exact u78,500 checkpoint, anchors, logs, and BC inputs.
./train_v21.sh --check

# First run builds/trains the BC anchor, clones the u78,500 V20 pool, then PPO.
./train_v21.sh

# Resume the newest V21 checkpoint and its matching pool.
./resume_v21.sh
```

On a new GPU server, copy the repository together with `game_logs/`, recreate
the virtual environment, and rebuild Hisss so the CUDA extension targets that
server's GPU before running the checks:

```bash
# From the repository root on the GPU server.
.venv/bin/python -m pip install -e .
cd bs-blackout-starter
./train_v21.sh --check
./train_v21.sh
```

The launcher deliberately refuses any branch other than V20 update 78,500 and
refuses to overwrite an existing V21 run. `V21_DUEL_PROBABILITY`, update count,
environment count, evaluation sizes, BC epochs, and BC batch size can be
overridden through the `V21_*` environment variables. The supporting log
measurements are in
[`OPPONENT_HEURISTICS.md`](OPPONENT_HEURISTICS.md).

---

**Good luck, and happy coding! May the longest snake win.**
