# PPO5 inference agent

`ppo5.py` is a deliberately non-MCTS successor to `ppo4.py`. It uses a short
PPO4-specialized best-response continuation, the exact Blackout encoder,
recurrent actor, conservative opponent masks, and final hard-safety invariant.

The change is intentionally narrow: PPO4's exact own-body survival proof now
looks 18 turns ahead instead of 10 and has a bounded budget of 12,000 states
instead of 6,000. Budget exhaustion remains `UNKNOWN`; it never proves a move
bad and cannot veto the PPO policy. There are no opponent rollouts, sampled
worlds, tree statistics, UCB/PUCT scores, or MCTS dependencies.

## Evaluation against PPO4

Three paired, API-faithful seed blocks were run against `ppo4`, without any
comparison to `ppo_mcts`:

| Layout | Games | PPO5 score | PPO4 score | Paired delta |
|---|---:|---:|---:|---:|
| copies development | 16 | 0.406 | 0.344 | +0.063 |
| copies holdout | 24 | 0.375 | 0.375 | 0.000 |
| solo-pair holdout | 16 | 0.375 | 0.375 | 0.000 |
| combined | 56 | 0.384 | 0.366 | +0.018 |

The combined bootstrap 95% interval for the paired delta was `0.000` to
`+0.054`. PPO5 had 25 wins versus PPO4's 24 and reduced known-fatal moves from
81 to 77; neither agent emitted an avoidable known-fatal move. Mean latency was
3.83 ms versus 3.61 ms, p99 was 5.17 ms versus 4.84 ms, and no request exceeded
450 ms. A rare exhaustive proof took 355 ms, still below the 500 ms timeout.

This is a measured improvement, but not evidence of a large strength jump.
Broader one-ply heuristics, PPO ensembles, a later V25 checkpoint, and relaxed
fog head-to-head/tail masks were also tested and rejected because independent
blocks were neutral or negative.

The selected deployment checkpoint is
`ppo_bs_lstm_cuda_v26_ppo4_best_response_selected.zip` (update 75,290). In its
fresh API-shaped `copies` holdout it scored 0.391 versus PPO4's 0.359, a paired
delta of `+0.031`, with 12 versus 11 wins. Its larger internal PPO4 gates had a
mean match score of roughly 0.531. Later updates regressed and were rejected.

## Run and evaluate

The historical deployment weights and comparison checkpoints discussed here
are not included in the source release. For a new policy, follow
[Training from scratch](TRAINING_FROM_SCRATCH.md) and point PPO5 at the exported
SB3 `.zip` file:

```bash
PPO5_MODEL_PATH=/absolute/path/to/your_model.zip python bs-blackout-starter/ppo5.py 8080
```

The historical evaluation commands below require the original comparison
models or explicit replacement paths; they do not work without model files.

Eine einstündige gepaarte Auswertung von PPO5 gegen PPO4 mit dem besten
promoteten Modell auf beiden Seiten benötigt keine Argumente. Der Report wird
automatisch unter `bs-blackout-starter/eval_results/` gespeichert:

```bash
PYTHONPATH=bs-blackout-starter .venv/bin/python \
  bs-blackout-starter/evaluate_inference.py
```

Nach 3.600 Sekunden werden keine neuen Paare mehr gestartet. Bereits laufende
Sitzrotationen werden vollständig beendet, damit ausschließlich komplette
Paare in die Statistik eingehen.

```bash
PYTHONPATH=bs-blackout-starter .venv/bin/python \
  bs-blackout-starter/evaluate_inference.py \
  --candidate ppo5 --baseline ppo4 \
  --pairs 25 --workers 4 --max-turns 2000 \
  --layout copies --output /tmp/ppo5-vs-ppo4.json
```

Optional overrides:

```bash
PPO5_SEARCH_HORIZON=18
PPO5_SEARCH_NODE_BUDGET=12000
```
