# PPO4 inference agent

`ppo4.py` is the deployment replacement for `ppo3.py`. It uses the promoted
V25 Targeted Finish champion by default and keeps the learned recurrent policy
as the strategic decision maker. The added logic corrects the live observation
and vetoes only known deaths or short, fully demonstrated self-traps.

## Why a new inference path was needed

The 498 newest deduplicated production games contained 228,057 moves and ended
in 141 wins and 356 losses. Of those losses, 179 were self-collisions, 57 wall
collisions, 52 head-to-heads, 48 enemy-body collisions, and 20 starvation
deaths. In 295 loss-ending requests the submitted target was already visibly
fatal. At least 143 of those requests still had another immediately safe move.

There were also three reproducible implementation problems:

- `ppo3.py` no longer imported with the current nine-channel architecture and
  its old default checkpoint had a 14-vs-9-channel shape mismatch.
- Reconstructing a `BattleSnakeGame` discarded the spawn-turn semantics of
  globally announced food. The food plane differed from native training
  observations in 12.2% of a measured sample.
- Hisss grows a snake one physical cell after increasing its logical length.
  While growth is pending, the current tail does not vacate. `ppo3.py` always
  removed it from the blocker set, causing avoidable self-collisions.

The direct encoder in `ppo4.py` was compared with 7,800 native Hisss
perspective-observations and was bit-identical on all nine actor channels.

## Inference layers

1. Direct API-to-tensor encoding preserves newly spawned food and eliminates
   the reconstructed C++ environment from the request path.
2. The hard mask rejects walls, occupied own cells (with correct pending-growth
   tail handling), every visible enemy segment, certain starvation/hazard
   deaths, and invalid board sizes.
3. The conservative tier treats any opponent body containing a Blackout gap as
   unknown-length and avoids a possible losing head-to-head when another safe
   move exists.
4. A bounded exact own-body DFS uses the real delayed-growth rule. The default
   horizon is 10 with 6,000 nodes per root. Budget exhaustion is `UNKNOWN` and
   never proves a move bad. A policy move that survives to the horizon remains
   unchanged.
5. If every uncontested action is a proven self-trap, a non-proven contested
   move can be used as an escape. Among doomed lines, an override requires at
   least four fully demonstrated extra survival turns.
6. The selected move is checked against the hard mask again as an independent
   final invariant.

Recurrent state is keyed by `(game.id, you.id)`, repeated requests return the
cached move without advancing the LSTM twice, and game-end callbacks release
the session. Game logs record the model hash, code hash, and inference config.

Eight-way D4 test-time augmentation remains available through
`PPO_SYMMETRIES=8`, but it is not the default: the ablation lost 0.292
normalized points per paired block against identity inference on the
development seeds.

## Evaluation

`evaluate_inference.py` uses API-shaped JSON with `include_eliminated=False`,
paired C++/Python/NumPy/Torch seeds, balanced seat rotations, a 2,000-turn
limit, no swallowed exceptions, tournament placement points, bootstrap
confidence intervals, elimination causes, known-fatal moves, and latency
percentiles. Candidate and baseline always used the same checkpoint when the
inference implementation itself was compared.

Final-code holdout results with the V25 champion:

| Layout | Games | PPO4 score | PPO3 score | Paired delta | 95% CI | Wins |
|---|---:|---:|---:|---:|---:|---:|
| two PPO4 vs two PPO3 | 20 | 0.500 | 0.250 | +0.250 | +0.050 to +0.450 | 15–5 |
| paired one-vs-three / three-vs-one | 20 | 0.463 | 0.288 | +0.175 | -0.050 to +0.375 | 14–6 |

A larger preceding holdout (the same code except for the final conservative
single-soft-trap fallback) added 80 games. Across both holdouts, 120 games gave
a paired delta of +0.183 with a bootstrap 95% CI of +0.104 to +0.263. The
two-vs-two subset was +0.221 (+0.121 to +0.329). PPO4 emitted zero avoidable
known-fatal moves; PPO3 emitted 240 in the 50-game large two-vs-two run.

The inference improvement also generalized when both sides were given older
weights:

| Shared checkpoint | Games | Paired PPO4–PPO3 delta | 95% CI |
|---|---:|---:|---:|
| V24 General Strength champion | 16 | +0.500 | +0.313 to +0.688 |
| V23 PPO Focus champion | 16 | +0.406 | +0.156 to +0.625 |
| V22 Effective champion | 16 | +0.156 | +0.000 to +0.282 |

The final-code runs had a PPO4 mean inference latency of 4.5–4.6 ms and p99 of
6.7–6.9 ms under multi-process CPU contention. There were no timeouts or agent
exceptions. The complete repository suite passed: 239 tests.

## Run the agent

From the repository root:

```bash
PYTHONPATH=bs-blackout-starter .venv/bin/python bs-blackout-starter/ppo4.py 8080
```

Useful overrides:

```bash
PPO_MODEL_PATH=/absolute/model.zip
PPO_SAFETY_SEARCH=0
PPO_SEARCH_HORIZON=10
PPO_SEARCH_NODE_BUDGET=6000
PPO_SYMMETRIES=1
```

Run a paired evaluation:

```bash
PYTHONPATH=bs-blackout-starter .venv/bin/python \
  bs-blackout-starter/evaluate_inference.py \
  --candidate ppo4 --baseline ppo3 \
  --pairs 25 --workers 4 --max-turns 2000 \
  --layout copies --output /tmp/ppo4-eval.json
```

Available ablation kinds are `ppo3`, `ppo3-no-space`, `ppo4-policy`,
`ppo4-search`, `ppo4-tta`, and `ppo4`.
