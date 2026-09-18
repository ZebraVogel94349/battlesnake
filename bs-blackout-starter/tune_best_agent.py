from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import hisss
import numpy as np

import best_agent
from battlesnake_types import BaseAgent, Direction, GameState
from best_agent import BestAgent
from hungry_agent import HungryAgent
from old_agent import BestAgent as OldBestAgent
from random_agent import RandomAgent


DIRECTION_TO_HISSS = {
    Direction.UP: hisss.UP,
    Direction.DOWN: hisss.DOWN,
    Direction.LEFT: hisss.LEFT,
    Direction.RIGHT: hisss.RIGHT,
}

CONSTANT_NAMES = [
    "BASE_STRATEGY_WEIGHT",
    "FOOD_MISSING_WEIGHT_MULTIPLIER",
    "EAT_HEALTH_LOW_THRESHOLD",
    "EAT_HEALTH_MEDIUM_THRESHOLD",
    "EAT_CLOSE_FOOD_DISTANCE",
    "EAT_NEAR_FOOD_DISTANCE",
    "EAT_HEALTH_LOW_MULTIPLIER",
    "EAT_HEALTH_MEDIUM_MULTIPLIER",
    "EAT_CLOSE_FOOD_MULTIPLIER",
    "EAT_NEAR_FOOD_MULTIPLIER",
    "EAT_CRITICAL_HEALTH_THRESHOLD",
    "EAT_CRITICAL_HEALTH_MULTIPLIER",
    "EAT_REACHABLE_SPACE_WEIGHT",
    "EAT_HEALTH_BIAS_BASE",
    "EAT_HEALTH_BIAS_SCALE",
    "CHILL_REACHABLE_SPACE_WEIGHT",
    "CHILL_OPEN_EXITS_WEIGHT",
    "CHILL_WALL_DISTANCE_WEIGHT",
    "CHILL_OPEN_SPACE_RATIO_THRESHOLD",
    "CHILL_OPEN_SPACE_MULTIPLIER",
    "CHILL_DENSE_SPACE_RATIO_THRESHOLD",
    "CHILL_DENSE_SPACE_MULTIPLIER",
    "AVOID_STUCK_REACHABLE_SPACE_WEIGHT",
    "AVOID_STUCK_OPEN_EXITS_WEIGHT",
    "AVOID_STUCK_WALL_DISTANCE_WEIGHT",
    "AVOID_STUCK_DEAD_END_THRESHOLD",
    "AVOID_STUCK_CORRIDOR_THRESHOLD",
    "AVOID_STUCK_DEAD_END_MULTIPLIER",
    "AVOID_STUCK_CORRIDOR_MULTIPLIER",
    "AVOID_STUCK_TIGHT_SPACE_RATIO",
    "AVOID_STUCK_TIGHT_SPACE_MULTIPLIER",
    "AVOID_STUCK_MEDIUM_SPACE_RATIO",
    "AVOID_STUCK_MEDIUM_SPACE_MULTIPLIER",
    "FLEE_CLOSE_DISTANCE",
    "FLEE_NEAR_DISTANCE",
    "FLEE_MEDIUM_DISTANCE",
    "FLEE_CLOSE_DISTANCE_MULTIPLIER",
    "FLEE_NEAR_DISTANCE_MULTIPLIER",
    "FLEE_MEDIUM_DISTANCE_MULTIPLIER",
    "FLEE_LONGER_SNAKE_MULTIPLIER_STEP",
    "FLEE_REACHABLE_SPACE_WEIGHT",
    "FLEE_WALL_DISTANCE_WEIGHT",
    "LOW_SPACE_BODY_FACTOR",
    "LOW_SPACE_SELF_BODY_MULTIPLIER",
    "MID_SPACE_SELF_BODY_MULTIPLIER",
    "HIGH_SPACE_RATIO",
    "HIGH_SPACE_CHILL_MULTIPLIER",
    "LOW_HEALTH_AGGRESSION_THRESHOLD",
    "LOW_HEALTH_CHILL_MULTIPLIER",
    "SAFE_MOVE_SPACE_WEIGHT",
    "SAFE_MOVE_WALL_DISTANCE_WEIGHT",
    "FALLBACK_OPEN_SPACE_WEIGHT",
    "FALLBACK_WALL_DISTANCE_WEIGHT",
    "HEAD_TO_HEAD_SAFE_SCORE",
    "HEAD_TO_HEAD_RISK_SCORE",
    "HEAD_TO_HEAD_ATTACK_SCORE",
    "HEAD_TO_HEAD_SPACE_WEIGHT",
    "EXPLORE_STALE_CELL_WEIGHT",
    "EXPLORE_DISTANCE_DECAY",
    "EXPLORE_REACHABLE_SPACE_WEIGHT",
    "EXPLORE_HEALTH_THRESHOLD",
    "EXPLORE_UNKNOWN_RATIO_THRESHOLD",
    "EXPLORE_OLD_CELL_TURN_THRESHOLD",
    "EXPLORE_WEIGHT_MULTIPLIER",
    "EXPLORE_LOW_HEALTH_MULTIPLIER",
    "STARVE_URGENCY_HEALTH",
    "STARVE_MEDIUM_HEALTH",
    "STARVE_PATH_BLOCK_WEIGHT",
    "STARVE_FOOD_RACE_WEIGHT",
    "STARVE_CLOSE_FOOD_DISTANCE",
    "STARVE_URGENCY_MULTIPLIER",
    "STARVE_MEDIUM_MULTIPLIER",
    "STARVE_WEIGHT_MULTIPLIER",
    "STARVE_LOW_HEALTH_MULTIPLIER",
    "GROWTH_SAFE_SPACE_RATIO",
    "GROWTH_FOOD_MULTIPLIER",
    "GROWTH_OPEN_SPACE_CHILL_MULTIPLIER",
    "STRATEGY_COMMITMENT_MIN_TURNS",
    "STRATEGY_COMMITMENT_MULTIPLIER",
    "STRATEGY_SWITCH_MARGIN",
]


@dataclass(frozen=True)
class TuningResult:
    label: str
    constants: dict[str, float | int]
    games: int
    candidate_wins: int
    candidate_survivor_draws: int
    draws: int
    avg_turns: float
    score: float
    opponents: dict[str, int]
    elimination_reasons: dict[str, int]


class TunedBestAgent(BestAgent):
    """BestAgent variant whose constants are patched only for this move."""

    def __init__(self, constants: dict[str, float | int], label: str):
        super().__init__()
        self.constants = constants
        self.label = label

    def get_name(self):
        return self.label

    def _with_constants(self, func: Callable[[], Any]) -> Any:
        old_values = {name: getattr(best_agent, name) for name in self.constants}
        try:
            for name, value in self.constants.items():
                setattr(best_agent, name, value)
            return func()
        finally:
            for name, value in old_values.items():
                setattr(best_agent, name, value)

    def move(self, game_state: GameState):
        return self._with_constants(lambda: super(TunedBestAgent, self).move(game_state))


class QuietHungryAgent(HungryAgent):
    def get_name(self):
        return "HungryAgent"


class QuietOldAgent(OldBestAgent):
    def get_name(self):
        return "OldAgent"


class QuietRandomAgent(RandomAgent):
    def get_name(self):
        return "RandomAgent"


def current_constants() -> dict[str, float | int]:
    return {name: getattr(best_agent, name) for name in CONSTANT_NAMES}


def tuned_factory(constants: dict[str, float | int], label: str) -> Callable[[], BaseAgent]:
    return lambda constants=constants, label=label: TunedBestAgent(constants, label)


def agent_factory(agent_cls: type[BaseAgent]) -> Callable[[], BaseAgent]:
    return lambda agent_cls=agent_cls: agent_cls()


def build_game_config(num_players: int, width: int, height: int):
    game_cfg = hisss.restricted_standard_config()
    game_cfg.all_actions_legal = True
    game_cfg.w = width
    game_cfg.h = height
    game_cfg.num_players = num_players
    game_cfg.init_snake_len = [3] * num_players
    return game_cfg


def state_for_agent(env, idx: int, game_id: str, include_eliminated: bool = False) -> GameState:
    state_json = hisss.to_battlesnake_json(env, idx, include_eliminated=include_eliminated)
    state_dict = json.loads(state_json)
    state_dict["game"]["id"] = game_id
    return GameState.model_validate(state_dict)


def run_match(
    agent_factories: list[Callable[[], BaseAgent]],
    labels: list[str],
    width: int,
    height: int,
    game_id: str,
    max_turns: int,
) -> tuple[list[str], int, dict[str, str], Counter]:
    agents = [factory() for factory in agent_factories]
    env = hisss.BattleSnakeGame(build_game_config(len(agents), width, height))
    runtime_errors = Counter()

    for idx, agent in enumerate(agents):
        agent.start(state_for_agent(env, idx, game_id))

    turns = 0
    while not env.is_terminal() and turns < max_turns:
        moves = []
        for idx, agent in enumerate(agents):
            if not env.is_player_at_turn(idx):
                continue
            try:
                move_result = agent.move(state_for_agent(env, idx, game_id, include_eliminated=True))
                moves.append(DIRECTION_TO_HISSS[move_result.move])
            except Exception:
                runtime_errors[labels[idx]] += 1
                moves.append(hisss.UP)
        env.step(actions=tuple(moves))
        turns += 1

    for idx, agent in enumerate(agents):
        try:
            agent.end(state_for_agent(env, idx, game_id))
        except Exception:
            pass

    state = env.get_state()
    alive = [idx for idx, is_alive in enumerate(state.snakes_alive) if is_alive]
    winner_idxs = alive[:1] if len(alive) <= 1 else alive
    winners = [labels[idx] for idx in winner_idxs]

    elimination_reasons = {}
    if state.elimination_events is not None:
        for idx, event in state.elimination_events.items():
            elimination_reasons[labels[idx]] = str(event.cause)

    return winners, turns, elimination_reasons, runtime_errors


def rotated(items: list[Any], offset: int) -> list[Any]:
    offset %= len(items)
    return items[offset:] + items[:offset]


def matchup_for_game(
    game_idx: int,
    candidate_constants: dict[str, float | int],
    incumbent_constants: dict[str, float | int],
) -> tuple[list[Callable[[], BaseAgent]], list[str]]:
    matchups: list[tuple[list[Callable[[], BaseAgent]], list[str]]] = [
        (
            [
                tuned_factory(candidate_constants, "candidate"),
                tuned_factory(incumbent_constants, "incumbent"),
            ],
            ["candidate", "incumbent"],
        ),
        (
            [
                tuned_factory(candidate_constants, "candidate"),
                agent_factory(QuietHungryAgent),
            ],
            ["candidate", "hungry"],
        ),
        (
            [
                tuned_factory(candidate_constants, "candidate"),
                tuned_factory(incumbent_constants, "incumbent"),
                agent_factory(QuietHungryAgent),
            ],
            ["candidate", "incumbent", "hungry"],
        ),
        (
            [
                tuned_factory(candidate_constants, "candidate"),
                tuned_factory(incumbent_constants, "incumbent"),
                agent_factory(QuietHungryAgent),
                agent_factory(QuietOldAgent),
            ],
            ["candidate", "incumbent", "hungry", "old"],
        ),
        (
            [
                tuned_factory(candidate_constants, "candidate"),
                agent_factory(QuietHungryAgent),
                agent_factory(QuietOldAgent),
            ],
            ["candidate", "hungry", "old"],
        ),
        (
            [
                tuned_factory(candidate_constants, "candidate"),
                tuned_factory(incumbent_constants, "incumbent"),
                agent_factory(QuietHungryAgent),
                agent_factory(QuietRandomAgent),
            ],
            ["candidate", "incumbent", "hungry", "random"],
        ),
    ]

    factories, labels = matchups[game_idx % len(matchups)]
    offset = (game_idx // len(matchups)) % len(labels)
    return rotated(factories, offset), rotated(labels, offset)


def score_result(
    games: int,
    candidate_wins: int,
    candidate_survivor_draws: int,
    avg_turns: float,
    max_turns: int,
    opponents: Counter,
    elimination_reasons: Counter,
) -> float:
    losses = games - candidate_wins - candidate_survivor_draws
    bad_elimination_penalty = (
        elimination_reasons["wall-collision"] * 0.35
        + elimination_reasons["snake-self-collision"] * 0.28
        + elimination_reasons["head-collision"] * 0.18
        + elimination_reasons["snake-collision"] * 0.12
        + elimination_reasons["out-of-health"] * 0.22
        + elimination_reasons["runtime-error"] * 1.0
    )
    opponent_penalty = (
        opponents["incumbent"] * 0.25
        + opponents["hungry"] * 0.2
        + opponents["old"] * 0.18
        + opponents["random"] * 0.5
    )
    survival_bonus = min(avg_turns, max_turns) * 0.001

    return (
        candidate_wins
        + candidate_survivor_draws * 0.5
        - losses * 0.12
        - opponent_penalty
        - bad_elimination_penalty
        + survival_bonus
    ) / max(1, games)


def evaluate_candidate_task(payload: dict[str, Any]) -> dict[str, Any]:
    candidate_constants = payload["candidate_constants"]
    incumbent_constants = payload["incumbent_constants"]
    games = payload["games"]
    width = payload["width"]
    height = payload["height"]
    max_turns = payload["max_turns"]
    seed = payload["seed"]
    label = payload["label"]

    candidate_wins = 0
    candidate_survivor_draws = 0
    draws = 0
    turns_seen = []
    opponents = Counter()
    elimination_reasons = Counter()

    for game_idx in range(games):
        game_seed = seed + game_idx * 7919
        random.seed(game_seed)
        np.random.seed(game_seed % (2**32 - 1))
        factories, labels = matchup_for_game(game_idx, candidate_constants, incumbent_constants)

        winners, turns, reasons, runtime_errors = run_match(
            factories,
            labels,
            width=width,
            height=height,
            game_id=f"tune-{seed}-{game_idx}",
            max_turns=max_turns,
        )
        turns_seen.append(turns)

        if len(winners) == 1:
            winner = winners[0]
            if winner == "candidate":
                candidate_wins += 1
            else:
                opponents[winner] += 1
        else:
            draws += 1
            if "candidate" in winners:
                candidate_survivor_draws += 1

        if "candidate" in reasons:
            elimination_reasons[reasons["candidate"]] += 1
        if runtime_errors["candidate"]:
            elimination_reasons["runtime-error"] += runtime_errors["candidate"]

    avg_turns = statistics.fmean(turns_seen) if turns_seen else 0.0
    score = score_result(
        games=games,
        candidate_wins=candidate_wins,
        candidate_survivor_draws=candidate_survivor_draws,
        avg_turns=avg_turns,
        max_turns=max_turns,
        opponents=opponents,
        elimination_reasons=elimination_reasons,
    )

    return TuningResult(
        label=label,
        constants=candidate_constants,
        games=games,
        candidate_wins=candidate_wins,
        candidate_survivor_draws=candidate_survivor_draws,
        draws=draws,
        avg_turns=avg_turns,
        score=score,
        opponents=dict(opponents),
        elimination_reasons=dict(elimination_reasons),
    ).__dict__


def candidate_values(name: str, value: float | int) -> list[float | int]:
    if isinstance(value, int) and not isinstance(value, bool):
        if "DISTANCE" in name or "THRESHOLD" in name or name == "LOW_SPACE_BODY_FACTOR":
            deltas = [-2, -1, 1, 2]
            values = [max(1, value + delta) for delta in deltas]
        else:
            values = [max(1, round(value * factor)) for factor in (0.75, 0.9, 1.1, 1.25)]
    else:
        if value == 0:
            values = [0.01, 0.05, 0.1]
        elif "RATIO" in name:
            values = [value * factor for factor in (0.7, 0.85, 1.15, 1.3)]
            values = [min(0.95, max(0.02, candidate)) for candidate in values]
        elif "MULTIPLIER" in name or "WEIGHT" in name or "SCALE" in name or "STEP" in name:
            values = [value * factor for factor in (0.6, 0.8, 1.2, 1.5)]
            values = [max(0.001, candidate) for candidate in values]
        else:
            values = [value * factor for factor in (0.75, 0.9, 1.1, 1.25)]

    clean_values = []
    for candidate in values:
        if isinstance(value, int) and not isinstance(value, bool):
            candidate = int(candidate)
        else:
            candidate = round(float(candidate), 6)
        if candidate != value and candidate not in clean_values:
            clean_values.append(candidate)
    return clean_values


def write_checkpoint(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_result(prefix: str, result: dict[str, Any]) -> None:
    print(
        f"{prefix} {result['label']}: score={result['score']:.4f}, "
        f"wins={result['candidate_wins']}/{result['games']}, "
        f"draw_survive={result['candidate_survivor_draws']}, "
        f"avg_turns={result['avg_turns']:.1f}, "
        f"lost_to={result['opponents']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune BestAgent constants with parallel Hisss simulations."
    )
    parser.add_argument("--games", type=int, default=500, help="Games per candidate value")
    parser.add_argument("--rounds", type=int, default=2, help="Full passes over all constants")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument(
        "--min-score-improvement",
        type=float,
        default=0.005,
        help="Minimum paired score improvement required before accepting a constant change",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("tune_best_agent_results.json"))
    parser.add_argument(
        "--constants",
        nargs="*",
        default=CONSTANT_NAMES,
        help="Optional subset of constants to tune",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    unknown = sorted(set(args.constants) - set(CONSTANT_NAMES))
    if unknown:
        raise SystemExit(f"Unknown constants: {', '.join(unknown)}")

    incumbent = current_constants()
    history: list[dict[str, Any]] = []
    started = time.time()

    print(
        f"Tuning {len(args.constants)} constants, {args.games} games/value, "
        f"{args.rounds} rounds, {args.workers} workers"
    )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for round_idx in range(args.rounds):
            print(f"\n=== Round {round_idx + 1}/{args.rounds} ===")
            improved = 0

            for const_idx, name in enumerate(args.constants, start=1):
                base_value = incumbent[name]
                values = candidate_values(name, base_value)
                tasks = []
                eval_seed = args.seed + round_idx * 100000 + const_idx * 1000

                for value_idx, value in enumerate(values):
                    candidate = dict(incumbent)
                    candidate[name] = value
                    label = f"{name}={value}"
                    tasks.append(
                        {
                            "candidate_constants": candidate,
                            "incumbent_constants": incumbent,
                            "games": args.games,
                            "width": args.width,
                            "height": args.height,
                            "max_turns": args.max_turns,
                            "seed": eval_seed,
                            "label": label,
                        }
                    )
                tasks.append(
                    {
                        "candidate_constants": incumbent,
                        "incumbent_constants": incumbent,
                        "games": args.games,
                        "width": args.width,
                        "height": args.height,
                        "max_turns": args.max_turns,
                        "seed": eval_seed,
                        "label": f"{name}=current",
                    }
                )

                print(f"\n[{const_idx}/{len(args.constants)}] {name} current={base_value}")
                futures = [executor.submit(evaluate_candidate_task, task) for task in tasks]
                results = []
                baseline = None
                for future in as_completed(futures):
                    result = future.result()
                    if result["label"].endswith("=current"):
                        baseline = result
                        print_result(" baseline", result)
                    else:
                        results.append(result)
                        print_result(" ", result)

                best = max(results, key=lambda item: item["score"])
                if baseline is None:
                    raise RuntimeError(f"Missing baseline result for {name}")

                accepted = best["score"] > baseline["score"] + args.min_score_improvement
                if accepted:
                    incumbent = dict(best["constants"])
                    improved += 1
                    print(f" -> accepted {name}: {base_value} -> {incumbent[name]}")
                else:
                    delta = best["score"] - baseline["score"]
                    print(
                        f" -> kept {name}: {base_value} "
                        f"(best delta {delta:.4f}, required {args.min_score_improvement:.4f})"
                    )

                history.append(
                    {
                        "round": round_idx + 1,
                        "constant": name,
                        "previous": base_value,
                        "accepted": accepted,
                        "best_result": best,
                        "baseline_result": baseline,
                    }
                )
                write_checkpoint(
                    args.checkpoint,
                    {
                        "elapsed_seconds": round(time.time() - started, 3),
                        "games_per_value": args.games,
                        "min_score_improvement": args.min_score_improvement,
                        "rounds_requested": args.rounds,
                        "best_constants": incumbent,
                        "history": history,
                    },
                )

            print(f"\nRound {round_idx + 1} improvements: {improved}")
            if improved == 0:
                break

    print("\n=== Best constants ===")
    for name in CONSTANT_NAMES:
        print(f"{name} = {incumbent[name]!r}")
    print(f"\nSaved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
