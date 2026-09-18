# PPO-guided Information-Set MCTS

`ppo_mcts.py` is the inference entry point for the hybrid agent. It keeps the
recurrent PPO actor as the strategic prior and anytime fallback, then spends the
remaining request budget on an open-loop information-set PUCT search. The
production default is 400 ms, leaving roughly 100 ms of the 500 ms protocol
timeout for request parsing, logging, serialization, scheduling jitter, and the
last in-flight simulation.

## Why the PPO ZIP is not used as a value function

The SB3 ZIPs exported by `train_cuda.py` contain the trained actor, but not the
privileged critic CNN. During export the actor feature extractor deliberately
fills the value-extractor slots because PPO inference never reads values. The
resulting ZIP value is therefore meaningless and must not be backed up through
the tree.

The hybrid uses:

- PPO actor logits as the root PUCT prior;
- PPO's recurrent action as a deadline/error/undersampling fallback;
- exact terminal values and a bounded geometry/territory/health heuristic at
  non-terminal leaves;
- no ZIP critic values.

The full training `.pt` does contain a real privileged critic, but it observes
hidden state unavailable to the live agent and its recurrent history cannot be
reconstructed faithfully from one Blackout request. It is therefore not part
of the safe default.

## Search under fog of war

`mcts_simulator.py` implements immutable, hashable search states in model action
order (`up, down, left, right`). Full-information transitions were checked
against Hisss for delayed growth, food, health/hazards, tails, body collisions,
head swaps, and simultaneous head-to-head resolution.

The request is an information set, not a complete board. Each simulation draws
a deterministic, request-seeded particle:

- hidden living opponents stay alive and receive plausible off-screen states;
- opponent health and length remain uncertain and are treated conservatively;
- visible body cells remain hard blockers;
- old food hidden by Blackout is sampled outside view up to `minimumFood`;
- only visible or newly announced food influences action priors, while hidden
  particle food still affects physical collisions/eating;
- future minimum/chance food and Royale randomness use the private search RNG.

Tree nodes represent our action history rather than one guessed board, so root
edge statistics aggregate across particles. Opponent actions mix geometric
sampling with an adversarial head-to-head component. PPO4's exact own-body DFS
and hard immediate-death mask remain active before MCTS. Search normally stays
inside PPO4's soft-safe root set; a contested hard-safe action is admitted only
when there is no soft-safe move or the complete PPO4 trap search proves that the
soft route is forced death. The selected action is checked against the hard mask
again after search.

This is a deliberately inexpensive root-sampled belief approximation, not a
persistent Bayesian tracker. It avoids hidden-state leakage and is robust to
missing information, but it does not claim to reconstruct the true hidden
opponent paths.

## Deadline and production behavior

Defaults:

| Setting | Default | Meaning |
|---|---:|---|
| `PPO_MCTS_TIME_BUDGET_MS` | 400 | request-relative search deadline |
| `PPO_MCTS_TIMEOUT_MARGIN_MS` | 80 | also cap search at protocol timeout minus margin |
| `PPO_MCTS_MAX_ITERATIONS` | 50,000 | safety ceiling; wall time normally stops first |
| `PPO_MCTS_MIN_ITERATIONS` | 24 | minimum evidence before MCTS may replace PPO |
| `PPO_MCTS_WAVE_ITERATIONS` | 16 | simulations in each independent root tree |
| `PPO_MCTS_MIN_COMPLETE_WAVES` | 3 | complete evidence waves required |
| `PPO_MCTS_HORIZON` | 12 | tree plus rollout horizon |
| `PPO_MCTS_PARTICLES` | 24 | root belief particles |
| `PPO_MCTS_ROLLOUT_STEPS` | 2 | geometric rollout steps at a new leaf |
| `PPO_MCTS_C_PUCT` | 1.35 | policy/exploration weight |
| `PPO_MCTS_ADVERSARIAL_FRACTION` | 0.20 | adversarial opponent samples |

The model is warmed during construction, so lazy CUDA/CPU kernels do not consume
the first move's budget. Recurrent state and retry results are transactional and
keyed by `(game.id, you.id)`. Searches for different games use separate locks;
only the short shared model forward is serialized.

For production-sized searches, the budget is divided into independent 16-
simulation root trees with separately seeded belief particles. Each complete
tree casts one action vote. A non-PPO action replaces PPO only with a strict
majority across at least three complete trees; split votes, partial final waves,
errors, or insufficient evidence fall back to PPO. A measured-duration guard
does not start a new tree if it is unlikely to finish before the deadline. This
uses extra time as a stability check instead of allowing one increasingly deep
tree to amplify particle/model bias.

Setting `max_iterations <= 32` retains the exact fixed-budget single-tree path
used by model screens. In either mode, minimum evidence never extends the
deadline. `NaN`, infinity, negative budgets, and inconsistent iteration bounds
are rejected at construction.

The server no longer rewrites the complete growing match JSON before every
response. Moves are appended as compact JSONL records, and the conventional
full JSON is compacted asynchronously at `/end`. This removes an O(game length)
request-path cost that exceeded 500 ms by itself in long games.

## Running

From the repository root:

```bash
PYTHONPATH=bs-blackout-starter .venv/bin/python \
  bs-blackout-starter/ppo_mcts.py 8080
```

Command-line options cover time, iteration bounds, horizon, particles, rollout
steps, and seed. Environment variables with a `PPO_MCTS_` prefix expose the
complete configuration. `PPO_DEVICE=cuda` is supported, although this search is
mostly CPU-bound and should be benchmarked on the deployment host.

The no-argument model default is the bundled V24 General Strength champion,
selected during development screens. `PPO_MCTS_MODEL_PATH` overrides it
specifically for this agent; `PPO_MODEL_PATH` remains the shared fallback.

## Evaluation protocol

`evaluate_inference.py` supports `ppo-mcts` alongside PPO3/PPO4 ablations. It
uses two-game paired seeds with seat rotation, tournament placement points,
fractional top-two credit for boundary ties, bootstrap confidence intervals,
known-fatal move rates, latency percentiles, search diagnostics, model/code
hashes, and hardware/runtime metadata. The native engine RNG is re-seeded
immediately before every authoritative match step, preventing agent-side Hisss
helpers from contaminating paired food randomness.

A fixed 32-simulation screen is useful for affordable development model
selection. It is not a confirmatory strength test:

```bash
PYTHONPATH=bs-blackout-starter .venv/bin/python \
  bs-blackout-starter/evaluate_inference.py \
  --candidate ppo-mcts --baseline ppo4 \
  --candidate-model bs-blackout-starter/ppo_bs_lstm_cuda_v24_general_strength_champion.zip \
  --baseline-model bs-blackout-starter/ppo_bs_lstm_cuda_v24_general_strength_champion.zip \
  --candidate-time-ms 400 --candidate-max-iterations 32 \
  --candidate-min-iterations 24 --pairs 25 --workers 8 \
  --max-turns 2000 --seed 12345 --output /tmp/mcts-v24.json
```

Remove `--candidate-max-iterations 32` for the production 400 ms ensemble. A
single pair is a smoke test, not evidence: confidence intervals are reported as
unavailable until at least two independent pairs exist.

The production settings used for the final same-model holdout can be reproduced
explicitly (the effective values and source/model hashes are also embedded in
the JSON report):

```bash
PPO_MCTS_WAVE_ITERATIONS=16 PPO_MCTS_MIN_COMPLETE_WAVES=3 \
PYTHONPATH=bs-blackout-starter .venv/bin/python \
  bs-blackout-starter/evaluate_inference.py \
  --candidate ppo-mcts --baseline ppo4 \
  --candidate-model bs-blackout-starter/ppo_bs_lstm_cuda_v24_general_strength_champion.zip \
  --baseline-model bs-blackout-starter/ppo_bs_lstm_cuda_v24_general_strength_champion.zip \
  --candidate-time-ms 400 --candidate-min-iterations 24 \
  --layout copies --pairs 32 --workers 8 --max-turns 2000 \
  --seed 3310829 \
  --output bs-blackout-starter/eval_results/holdout_reward_aligned_wave16_min3_v24_vs_ppo4_v24_p32.json
```

## Verification

Focused tests cover API encoding, retry/LSTM transactions, strict deadlines,
root coverage, undersampling fallback, parallel game sessions, model locking,
hidden opponents/food, simulator parity, evaluator statistics, and bounded
server logging. Run them with:

```bash
.venv/bin/python -m pytest -q \
  bs-blackout-starter/test_battlesnake_server.py \
  bs-blackout-starter/test_evaluate_inference.py \
  bs-blackout-starter/test_mcts_simulator.py \
  bs-blackout-starter/test_ppo_mcts.py \
  bs-blackout-starter/test_ppo4.py
```

The complete loopback HTTP path—including JSON serialization, Flask/Pydantic,
the bounded logging enqueue, response decoding, duplicate `/start`, and a
same-turn retry—can be checked with the hard 500 ms gate:

```bash
PPO_MCTS_WAVE_ITERATIONS=16 PPO_MCTS_MIN_COMPLETE_WAVES=3 \
PYTHONPATH=bs-blackout-starter .venv/bin/python \
  bs-blackout-starter/benchmark_http_inference.py \
  --model bs-blackout-starter/ppo_bs_lstm_cuda_v24_general_strength_champion.zip \
  --moves 30 --time-budget-ms 400 --max-iterations 50000 \
  --min-iterations 24 --seed 4710829 --sla-ms 500 \
  --output bs-blackout-starter/eval_results/http_release_reward_aligned_wave16_min3_v24_m30.json
```

## Final evaluation results

All results below use final agent code `18144ed62f08`, evaluator schema 2,
Hisss 1.3.0, paired common-random-number seeds, seat rotations, and placement
awards `(2, 1, 0, 0)`. Development/tuning reports are deliberately excluded
from confirmatory claims.

The primary same-policy holdout isolates the search layer by loading the exact
same V24 model (`efc00fae8e9a`) in both agents. Across 32 fresh pairs / 64
games, PPO-MCTS scored `0.390625` versus PPO4's `0.359375`: paired delta
`+0.03125`, bootstrap 95% CI `[-0.0859375, +0.1484375]`. Wins were 34:30 and
top-two credits 66:62. The hybrid had 0 timeouts, 0 search errors, 0 moves over
500 ms, p99 384.46 ms, and changed PPO's action on 1.01% of moves. The report is
[`holdout_reward_aligned_wave16_min3_v24_vs_ppo4_v24_p32.json`](eval_results/holdout_reward_aligned_wave16_min3_v24_vs_ppo4_v24_p32.json),
SHA-256 `6b20ef7792b2d371eec45b238f5739fbe490198dc015a74d8c6db05178a2ff22`.

Two smaller cross-model robustness screens kept the production hybrid fixed at
MCTS-V24 while changing only the PPO4 opponent. Both produced a positive point
estimate, but both intervals include zero:

| Opponent | Pairs | Hybrid score | PPO4 score | Paired delta | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| PPO4-V23 | 12 | 0.4271 | 0.3229 | +0.1042 | [-0.0833, +0.3125] |
| PPO4-V25 | 12 | 0.4271 | 0.3229 | +0.1042 | [-0.0417, +0.2500] |

Artifacts:
[`robust_system_mcts_v24_vs_ppo4_v23_p12.json`](eval_results/robust_system_mcts_v24_vs_ppo4_v23_p12.json)
(SHA-256 `7e381c213183f7a9c8fba14c98917390b3dd7c00370ea20c7bbe97b6b384f7e1`)
and
[`robust_system_mcts_v24_vs_ppo4_v25_p12.json`](eval_results/robust_system_mcts_v24_vs_ppo4_v25_p12.json)
(SHA-256 `c2a38f1423249a6738ce608d90e877978e5005ad6cb68127819fccb83db20d7f`).
These comparisons measure total deployment-agent strength, not the causal MCTS
increment, because the opponent checkpoint differs.

A separate PPO-only 32-pair model check found V25 below V24 by `-0.0625`, CI
`[-0.1796875, +0.0546875]`; this supports retaining V24 but is likewise
inconclusive. Its artifact is
[`robust_ppo4_v25_vs_v24_p32.json`](eval_results/robust_ppo4_v25_vs_v24_p32.json)
(SHA-256 `6a5a77d6a371599bd0bc364e3cc1fc0c18d8fafdd8ea3b2a6d13c114cc61caef`).

The final real-HTTP benchmark passed all 33 primary/retry `/move` samples:
p99 372.56 ms, maximum 373.20 ms, 0 samples over 450 or 500 ms, complete logs,
and an identical 2.27 ms cached retry. The artifact is
[`http_release_reward_aligned_wave16_min3_v24_m30.json`](eval_results/http_release_reward_aligned_wave16_min3_v24_m30.json)
(SHA-256 `91bc8fc6a01a4f9630293e4ad2a0900dbdc72f357c9a91c3b52a143782ad2a24`).

The operational/deadline target is therefore met and the final-code results
are directionally positive across V23, V24, and V25 opponents. The primary
confidence interval still crosses zero, so these data do **not** establish that
MCTS is substantially stronger than the underlying PPO policy. A stronger
statistical claim requires a substantially larger, newly seeded frozen
holdout; it must not be manufactured by pooling development or cross-model
screens.
