# Log-derived CUDA opponents for v19-v21

## Evidence window

The profiles use the 260 newest log files present on 31 July 2026. After
choosing the richest file for each `game_id`, this is 259 distinct games from
25–30 July and 32,371 reconstructed opponent moves. A move is counted only
when the same opponent head is visible in two consecutive observations, so the
measurements do not invent behavior outside Blackout's sight radius.

The recent population is not homogeneous:

| Observed cluster | Representative opponents | Distinguishing measurements |
| --- | --- | --- |
| persistent food seekers | 34, 35, 40 | 84.5–88.5% of moves reduce visible-food distance; 51.0–55.2% continue straight |
| visible-head pressure | 30, 32, 41 | 61.7–70.3% of moves approach our visible head |
| center controllers | 38, 46 | 66.2–67.1% of observed targets are in the central band; only 1.5–3.4% touch an edge |
| edge players | 28, 43 | 10.4–10.9% of observed targets touch an edge; substantially less central than the controllers |
| evasive/weaving players | 15, 17, 21 | only 32.8–35.5% continue straight and roughly half of moves approach visible food or heads |

These are behavioral observations, not claims about an opponent's source code.
Opponent health is normally fogged in the logs, and only behavior while visible
can be measured. The CUDA implementations therefore use the profile trends as
anchors while retaining hard collision and starvation safeguards.

## The original five profiles

1. **Forager** strongly minimizes visible-food distance, amplifies that drive
   at low health, persists in its current direction, and accepts more edge and
   corridor exposure.
2. **Hunter** closes distance to visible heads, prioritizes shorter prey, and
   commits strongly to winning head-to-heads while rejecting losing contests.
3. **Territorial** maximizes flood-filled reachable space, exits, and wall
   clearance. Food becomes important only under health pressure.
4. **Edge-Trapper** prefers lanes one cell from a wall and pressures visible
   opponents already near an edge. It deliberately tolerates narrower routes.
5. **Survivor** maximizes escape routes and separation, follows its tail when
   useful, and receives a turn bonus plus deterministic jitter to reproduce the
   high-direction-change cluster.

All policies execute inside `battlesnake_cuda.cu`. Python passes only a
small device-resident profile-id tensor and receives native CUDA actions; no
board state or action is copied through the CPU.

## Snake 25

Snake 25 occurs in 90 of the 259 newest distinct games. The logged agent lost
all 90 of those games; Snake 25 was the last reported survivor in 34. This is
not enough to recover its source code, but 3,680 reconstructed visible moves
show a consistent policy:

| Measurement | Snake 25 |
| --- | ---: |
| chooses a move tied for most immediate exits | 79.3% |
| reduces distance to nearest visible food | 51.3% |
| chooses a food-distance-optimal legal move | 56.0% |
| continues straight | 39.7% |
| changes direction | 52.5% |
| enters a possible visible head contest | 3.4% |
| contest set contains a shorter opponent | 79.4% of contests |
| moves into a one-exit corridor | 6.8% |

The resulting `snake25` anchor is therefore a **space-first opportunist**, not
a pure hunter: it maximizes local exits and observed flood-fill space, keeps
moderate food pressure at high health, increases it when hungry, changes
direction slightly more often than it goes straight, and commits strongly only
to favorable head-to-heads. Hard collision, hazard, and starvation penalties
remain absolute safeguards.

Two additional profiles target that policy through different mechanisms:

1. **Snake25 Interceptor** uses full safe-space planning and predicts which
   exit-rich cell a visible opponent is likely to choose. It intercepts that
   route only with a length advantage and otherwise preserves separation. Its
   food drive is deliberately low unless health demands it.
2. **Snake25 Denier** takes the resource route: it combines broad observed
   territory with aggressive visible-food races, grows past Snake 25, then
   converts the length lead into controlled head pressure.

In a fixed 2-vs-2 CUDA probe with all six seat-pair rotations and 2,048 games
per comparison, the local clone produced these results. They validate the
counter design against the implemented anchor, not against Snake 25's unknown
server implementation:

| A vs `snake25` | A win | Snake 25 win | draw |
| --- | ---: | ---: | ---: |
| `best` | 58.1% | 41.0% | 1.0% |
| `snake25_interceptor` | 54.1% | 45.1% | 0.9% |
| `snake25_denier` | 52.4% | 43.8% | 3.7% |
| `hunter` | 26.6% | 71.9% | 1.5% |

Hunter remains important because it is empirically strong against v19; the
table only says that its aggressive style is not a direct counter to the local
Snake-25 model.

## Snake 25 in a true 1-v-1

The 126 new logs from 1-2 August contain 69 games with Snake 25. The logged
agent lost 61, won 7, and one result is incomplete. Nineteen games reached a
pure Snake 24 versus Snake 25 phase; Snake 25 survived 18 of them (94.7%). The
losses were mostly patient constrictions rather than immediate attacks: nine
ended in our self-collision, four in a head-to-head caused by Snake 25, two in
a collision with its body, two at a wall, and one by starvation.

The visible-move profile also changes once only two snakes remain:

| Measurement | 1-v-1 phase | More than two alive |
| --- | ---: | ---: |
| follows its own visible tail | 72.4% | 61.5% |
| chooses a move tied for most immediate exits | 72.1% | 80.8% |
| enters a one-exit corridor | 16.1% | 6.4% |
| reduces distance to our visible head | 53.3% | 48.5% |
| reduces visible-food distance | 59.9% | 59.6% |
| changes direction after an observed prior move | 46.4% | 41.3% |

`snake25_duelist` captures that late-game policy separately: it is central and
food-aware, deliberately follows its tail through viable narrow lanes, turns
more than the general clone, applies measured head pressure with a length lead,
and otherwise waits for the opponent to run out of safe territory. This is
paired with native two-snake resets; it is not merely placed into another
four-player game.

## Training integration

`train_cuda.py --pool-real-heuristic-weight W` enables the whole group and
splits `W` evenly. For focused runs, repeat
`--pool-heuristic-weight LABEL=WEIGHT`; these exact per-profile floors disable
unspecified native profiles and are mutually exclusive with the aggregate
option. Every enabled profile remains its own `heuristic:<name>` payoff-matrix
column, so Nash can add pressure to a hard profile without collapsing its
configured floor.

The v19 launcher enables a combined weight of `0.25` (five floors of `0.05`)
and reserves the remaining configured mass for historical neural opponents,
the previous best heuristic, Hungry, and Random.

V20 uses exact floors. The complete configured mix is 50% neural checkpoints,
10% Best, 12% Hunter, 12% Snake 25, 7% Interceptor, 7% Denier, 0.5% Hungry,
0.5% Random, and 0.25% each for Forager, Territorial, Edge-Trapper, and
Survivor. Thus the four weak old CUDA profiles together fall from 20% in v19
to 1%, while Hunter rises from 5% to 12%.

At every evaluation interval, TensorBoard receives individual
`eval/heuristics/<label>/{current_win_rate,opponent_win_rate,draw_rate}` series,
aggregate `eval/*` summaries, assignment counts under `selfplay/*_assignments`,
and the constrained mix under `selfplay/training_weight/heuristic/<label>`.

V21 starts 25% of rollout episodes with exactly two live snakes. Within those
duels, 50% force the native `snake25_duelist`, 30% force the neural
`snake25_bc`, and the ordinary league pool supplies the remaining 20%. Across
all rollout environments this yields 12.5% native Snake-25-strategy duels, 7.5%
Behavioral-Clone duels, and 5% diverse league duels; all four-player games keep
the compact league mixture. For rollout throughput, V21 samples only three
native profile variants: 10% Hunter, 12% Snake 25, and 12% Snake-25 Duelist.
The other configured mass is 55% neural checkpoints, 10% Best, and 1% Random;
the launch-bound Hungry implementation and five low-floor native variants are
not part of the training mix.

The Behavioral Clone uses the full deduplicated log directory rather than only
the latest evidence window. At implementation time it yielded 20,275 actions
from consecutive visible Snake-25 head positions, including 2,773 actions in
states with exactly two live snakes. Each input is recentered on Snake 25. The
original request does not reveal Snake 25's private view, so cells outside the
intersection of both view radii are explicitly left fogged and lower-coverage
examples receive less weight. Training starts from the V20 actor, uses a
game-level validation split and recurrent sequences, weights duel actions 2.5x,
and regularizes toward the V20 policy. The resulting policy is used only as a
frozen opponent anchor, avoiding the invalid shortcut of teaching our learner
to execute an opponent's action from our own observation.

To stop neural-anchor forgetting, V21 raises checkpoint mass to 55%, rotates
pinned anchors through a four-policy working set, and reserves 30 percentage
points equally across the active anchors. Anchor-target deficits additionally
reweight the remaining Nash mass.
This fixes the old behavior where PFSP priorities were calculated and logged
but ignored as soon as a Nash matrix existed. Candidate champions must also
pass a separate floor for every fixed neural anchor and for the true-duel
Snake-25 evaluation.
