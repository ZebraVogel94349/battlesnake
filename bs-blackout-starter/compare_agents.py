from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import hisss

from battlesnake_types import BaseAgent, Direction, GameState


DIRECTION_TO_HISSS = {
    Direction.UP: hisss.UP,
    Direction.DOWN: hisss.DOWN,
    Direction.LEFT: hisss.LEFT,
    Direction.RIGHT: hisss.RIGHT,
}


@dataclass
class MatchResult:
    winners: list[int]
    turns: int
    alive_order: list[int]
    elimination_reasons: dict[int, str]


def load_agent(spec: str) -> BaseAgent:
    """Load an agent from `module:ClassName` or a known top-level class name."""
    if ":" in spec:
        module_name, class_name = spec.split(":", 1)
    else:
        module_name, class_name = spec, spec

    module = importlib.import_module(module_name)
    agent_cls = getattr(module, class_name)
    return agent_cls()


def build_game_config(num_players: int, width: int, height: int):
    game_cfg = hisss.restricted_standard_config()
    game_cfg.all_actions_legal = True

    # Keep the standard starter-friendly shape and avoid unnecessary visual overhead.
    game_cfg.w = width
    game_cfg.h = height
    game_cfg.num_players = num_players
    game_cfg.init_snake_len = [3] * num_players
    return game_cfg


def run_single_match(agent_specs: list[str], width: int, height: int) -> MatchResult:
    agents = [load_agent(spec) for spec in agent_specs]
    game_cfg = build_game_config(len(agents), width, height)
    env = hisss.BattleSnakeGame(game_cfg)

    stable_game_id = "compare-agents-session"

    for idx, agent in enumerate(agents):
        state_json = hisss.to_battlesnake_json(env, idx)
        state_dict = json.loads(state_json)
        state_dict["game"]["id"] = stable_game_id
        agent.start(GameState.model_validate(state_dict))

    turn_count = 0
    max_turns = 1000

    while not env.is_terminal() and turn_count < max_turns:
        moves: list[int] = []
        for idx, agent in enumerate(agents):
            if not env.is_player_at_turn(idx):
                continue

            state_json = hisss.to_battlesnake_json(env, idx, include_eliminated=True)
            state_dict = json.loads(state_json)
            state_dict["game"]["id"] = stable_game_id
            game_state = GameState.model_validate(state_dict)

            try:
                move_result = agent.move(game_state)
                moves.append(DIRECTION_TO_HISSS[move_result.move])
            except Exception:
                # Illegal move or runtime error: fall back to up so the match can continue.
                moves.append(hisss.UP)

        env.step(actions=tuple(moves))
        turn_count += 1

    for idx, agent in enumerate(agents):
        state_json = hisss.to_battlesnake_json(env, idx)
        state_dict = json.loads(state_json)
        state_dict["game"]["id"] = stable_game_id
        agent.end(GameState.model_validate(state_dict))

    state = env.get_state()
    alive = [idx for idx, is_alive in enumerate(state.snakes_alive) if is_alive]

    winners = alive[:1]
    if len(alive) > 1:
        # If the match times out with multiple snakes alive, treat it as a shared draw.
        winners = alive

    elimination_reasons: dict[int, str] = {}
    if state.elimination_events is not None:
        for idx, event in state.elimination_events.items():
            elimination_reasons[idx] = str(event.cause)

    return MatchResult(
        winners=winners,
        turns=turn_count,
        alive_order=alive,
        elimination_reasons=elimination_reasons,
    )


def rotate_specs(specs: list[str], offset: int) -> list[str]:
    if not specs:
        return specs
    offset = offset % len(specs)
    return specs[offset:] + specs[:offset]


def compare_agents(agent_specs: list[str], games: int, width: int, height: int):
    results = Counter()
    turns: list[int] = []
    elimination_reasons = defaultdict(Counter)

    seat_wins = Counter()
    seat_draws = Counter()

    if not (2 <= len(agent_specs) <= 4):
        raise ValueError("compare_agents supports between 2 and 4 agents")

    # Alternate seat order to reduce positional bias.
    for game_index in range(games):
        specs = rotate_specs(agent_specs, game_index)
        seat_map = {idx: spec for idx, spec in enumerate(specs)}

        match = run_single_match(specs, width=width, height=height)
        turns.append(match.turns)

        if len(match.winners) == 1:
            winner_spec = seat_map[match.winners[0]]
            results[winner_spec] += 1
            seat_wins[winner_spec] += 1
        else:
            results["draw"] += 1
            for idx in match.winners:
                seat_draws[seat_map[idx]] += 1

        for idx, reason in match.elimination_reasons.items():
            elimination_reasons[seat_map[idx]][reason] += 1

        print(
            f"Game {game_index + 1:4d}/{games}: turns={match.turns:3d}, "
            f"winner={seat_map.get(match.winners[0], 'draw') if len(match.winners) == 1 else 'draw'}"
        )

    avg_turns = sum(turns) / len(turns) if turns else 0.0
    print("\n=== Summary ===")
    print(f"Games: {games}")
    print(f"Average turns: {avg_turns:.2f}")
    print(f"Draws: {results['draw']}")
    for spec in agent_specs:
        print(f"{spec}: {results[spec]}")

    print("\nElimination reasons:")
    for agent in agent_specs:
        if agent not in elimination_reasons:
            continue
        print(f"{agent}:")
        for reason, count in elimination_reasons[agent].most_common():
            print(f"  {reason}: {count}")

    print("\nSeat results:")
    for spec in agent_specs:
        print(f"{spec}: wins={seat_wins[spec]}, draws_as_survivor={seat_draws[spec]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare 2 to 4 Battlesnake agents via many Hisss simulations.")
    parser.add_argument(
        "agents",
        nargs="+",
        help="Agent specs, e.g. 'random_agent:RandomAgent' 'hungry_agent:HungryAgent' 'best_agent:BestAgent'",
    )
    parser.add_argument("--games", type=int, default=100, help="Number of simulated games")
    parser.add_argument("--width", type=int, default=15, help="Board width")
    parser.add_argument("--height", type=int, default=15, help="Board height")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_agents(
        args.agents,
        games=args.games,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()