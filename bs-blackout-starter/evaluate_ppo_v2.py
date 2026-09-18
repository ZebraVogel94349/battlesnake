"""Paired holdout evaluator for the opt-in PPO-MCTS implementations.

The existing ``evaluate_inference.py`` remains unchanged.  This companion uses
its API-faithful state conversion, scoring, and summary functions while adding
``ppo-mcts-v2`` and an explicit ``ppo-mcts-v1`` ablation.  Candidate and
baseline receive identical seeds, checkpoints, layouts, and seat rotations.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import threading
import time

import hisss
import numpy as np
import torch
from hisss.cpp.lib import CPP_LIB

from battlesnake_types import BaseAgent, Direction
from evaluate_inference import (
    DIRECTION_TO_HISSS,
    MatchResult,
    MatchTask,
    _known_fatal_reason,
    _placement_points,
    _seat_kinds,
    _state_for,
    summarize,
)
from obs_config import make_game_config
from ppo4 import DEFAULT_MODEL_PATH, PPOAgent4
from ppo_mcts import PPOMCTSAgent
from ppo_mcts_v2 import PPOMCTSAgentV2, ROOT_DECISIONS


KINDS = ("ppo-mcts-v2", "ppo-mcts-v1", "ppo4")


@dataclass(frozen=True)
class AgentSpecV2:
    kind: str
    model: str
    time_budget_ms: float
    max_iterations: int
    root_decision: str = "visits"


_CANDIDATE_TEMPLATE: BaseAgent | None = None
_BASELINE_TEMPLATE: BaseAgent | None = None


def _make_agent(spec: AgentSpecV2) -> BaseAgent:
    if spec.kind == "ppo-mcts-v2":
        return PPOMCTSAgentV2(
            spec.model,
            time_budget_ms=spec.time_budget_ms,
            max_iterations=spec.max_iterations,
            root_decision=spec.root_decision,
            diagnostics_path=None,
            symmetries=1,
            safety_search=True,
            device="cpu",
        )
    if spec.kind == "ppo-mcts-v1":
        return PPOMCTSAgent(
            spec.model,
            time_budget_ms=spec.time_budget_ms,
            max_iterations=spec.max_iterations,
            symmetries=1,
            safety_search=True,
            device="cpu",
        )
    if spec.kind == "ppo4":
        return PPOAgent4(
            spec.model,
            symmetries=1,
            safety_search=True,
            device="cpu",
        )
    raise ValueError(f"unknown agent kind: {spec.kind}")


def _worker_init(candidate: AgentSpecV2, baseline: AgentSpecV2) -> None:
    global _CANDIDATE_TEMPLATE, _BASELINE_TEMPLATE
    torch.set_num_threads(1)
    _CANDIDATE_TEMPLATE = _make_agent(candidate)
    _BASELINE_TEMPLATE = _make_agent(baseline)


def _clone_agent(template: BaseAgent) -> BaseAgent:
    agent = copy.copy(template)
    if isinstance(agent, PPOMCTSAgentV2):
        agent._sessions = {}
        agent._search_sessions = {}
        agent._session_locks = {}
        agent._lock = threading.RLock()
        agent._model_lock = threading.RLock()
        agent._diagnostic_lock = threading.RLock()
        agent._last_search = None
        agent.diagnostics_path = None
    elif isinstance(agent, PPOMCTSAgent):
        agent._sessions = {}
        agent._mcts_sessions = {}
        agent._session_locks = {}
        agent._lock = threading.RLock()
        agent._model_lock = threading.RLock()
        agent._last_search = None
    elif isinstance(agent, PPOAgent4):
        agent._sessions = {}
        agent._lock = threading.RLock()
    return agent


def _empty_search_stats() -> dict[str, object]:
    return {
        "moves": 0.0,
        "iterations": 0.0,
        "nodes": 0.0,
        "depth": 0.0,
        "max_depth": 0.0,
        "elapsed_ms": 0.0,
        "deadline_hits": 0.0,
        "errors": 0.0,
        "mcts_choices": 0.0,
        "insufficient_searches": 0.0,
        "policy_changes": 0.0,
        "ensemble_moves": 0.0,
        "waves_started": 0.0,
        "waves_completed": 0.0,
        "waves_discarded": 0.0,
        "voting_iterations": 0.0,
        "discarded_iterations": 0.0,
        "stopped_for_wave_guard": 0.0,
        "decision_reason_counts": {},
    }


def _run_match(task: MatchTask) -> MatchResult:
    if _CANDIDATE_TEMPLATE is None or _BASELINE_TEMPLATE is None:
        raise RuntimeError("evaluation worker was not initialized")
    random.seed(task.seed)
    np.random.seed(task.seed)
    torch.manual_seed(task.seed)
    CPP_LIB.lib.set_seed(task.seed)

    seat_kinds = _seat_kinds(task.layout, task.game_in_pair)
    agents = [
        _clone_agent(
            _CANDIDATE_TEMPLATE if kind == "candidate" else _BASELINE_TEMPLATE
        )
        for kind in seat_kinds
    ]
    config = make_game_config()
    config.all_actions_legal = True
    env = hisss.BattleSnakeGame(config)
    game_id = f"mcts-v2-eval-{task.pair}-{task.game_in_pair}-{task.seed}"
    latencies = {"candidate": [], "baseline": []}
    fatal_moves = {"candidate": 0, "baseline": 0}
    avoidable_fatal_moves = {"candidate": 0, "baseline": 0}
    search_stats = {kind: _empty_search_stats() for kind in ("candidate", "baseline")}
    agent_diagnostics = {}
    for seat, kind in enumerate(seat_kinds):
        if kind in agent_diagnostics:
            continue
        diagnostics_fn = getattr(agents[seat], "get_diagnostics", None)
        diagnostics = diagnostics_fn() if callable(diagnostics_fn) else {}
        agent_diagnostics[kind] = diagnostics if isinstance(diagnostics, dict) else {}
    turns = 0
    try:
        for seat, agent in enumerate(agents):
            agent.start(_state_for(env, seat, game_id))
        while not env.is_terminal() and turns < task.max_turns:
            actions = []
            for seat in env.players_at_turn():
                state = _state_for(env, seat, game_id)
                started = time.perf_counter()
                move = agents[seat].move(state)
                kind = seat_kinds[seat]
                latencies[kind].append((time.perf_counter() - started) * 1000.0)
                fatal = _known_fatal_reason(state, move.move)
                if fatal is not None:
                    fatal_moves[kind] += 1
                    if any(
                        _known_fatal_reason(state, direction) is None
                        for direction in Direction
                    ):
                        avoidable_fatal_moves[kind] += 1
                actions.append(DIRECTION_TO_HISSS[move.move])
            env.step(tuple(actions))
            turns += 1

        for seat, agent in enumerate(agents):
            agent.end(_state_for(env, seat, game_id))
        points, winner, causes = _placement_points(env.get_state(), timed_out=not env.is_terminal())
        return MatchResult(
            pair=task.pair,
            game_in_pair=task.game_in_pair,
            seed=task.seed,
            seat_kinds=seat_kinds,
            points=points,
            turns=turns,
            timed_out=not env.is_terminal(),
            winner=winner,
            elimination_causes=causes,
            move_latencies_ms=latencies,
            fatal_moves=fatal_moves,
            avoidable_fatal_moves=avoidable_fatal_moves,
            search_stats=search_stats,
            agent_diagnostics=agent_diagnostics,
        )
    finally:
        env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired evaluation of PPO-MCTS v2, v1, and PPO4."
    )
    parser.add_argument("--candidate", choices=KINDS, default="ppo-mcts-v2")
    parser.add_argument("--baseline", choices=KINDS, default="ppo4")
    parser.add_argument("--candidate-model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--baseline-model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--candidate-time-budget-ms", type=float, default=400.0)
    parser.add_argument("--baseline-time-budget-ms", type=float, default=400.0)
    parser.add_argument("--candidate-max-iterations", type=int, default=50_000)
    parser.add_argument("--baseline-max-iterations", type=int, default=50_000)
    parser.add_argument(
        "--candidate-root-decision", choices=ROOT_DECISIONS, default="visits"
    )
    parser.add_argument(
        "--baseline-root-decision", choices=ROOT_DECISIONS, default="visits"
    )
    parser.add_argument("--layout", choices=("copies", "solo-pair"), default="copies")
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max-turns", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=814_271)
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=None,
        help="Run duration in hours; if set, continuously schedule pairs until this wall-clock time elapses.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.pairs <= 0 or args.workers <= 0 or args.max_turns <= 0:
        raise ValueError("pairs, workers, and max-turns must be positive")
    candidate = AgentSpecV2(
        args.candidate,
        str(Path(args.candidate_model).resolve()),
        args.candidate_time_budget_ms,
        args.candidate_max_iterations,
        args.candidate_root_decision,
    )
    baseline = AgentSpecV2(
        args.baseline,
        str(Path(args.baseline_model).resolve()),
        args.baseline_time_budget_ms,
        args.baseline_max_iterations,
        args.baseline_root_decision,
    )
    results: list[MatchResult] = []
    started = time.perf_counter()
    # If a wall-clock duration was specified, keep scheduling pairs until
    # that duration elapses; otherwise submit a fixed set of pairs.
    if args.duration_hours is None:
        tasks = [
            MatchTask(
                pair=pair,
                game_in_pair=game_in_pair,
                seed=args.seed + pair,
                layout=args.layout,
                max_turns=args.max_turns,
            )
            for pair in range(args.pairs)
            for game_in_pair in range(2)
        ]

        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_worker_init,
            initargs=(candidate, baseline),
        ) as executor:
            futures = {executor.submit(_run_match, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                print(
                    f"[{completed:4d}/{len(tasks)}] pair={result.pair:4d} "
                    f"game={result.game_in_pair} turns={result.turns:4d} "
                    f"points={result.points}",
                    flush=True,
                )
    else:
        duration_seconds = float(args.duration_hours) * 3600.0
        end_time = time.perf_counter() + duration_seconds
        next_pair = 0
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_worker_init,
            initargs=(candidate, baseline),
        ) as executor:
            futures: dict = {}
            # prime the queue with a small backlog
            backlog = max(2, args.workers * 2)
            while len(futures) < backlog and time.perf_counter() < end_time:
                for game_in_pair in (0, 1):
                    task = MatchTask(
                        pair=next_pair,
                        game_in_pair=game_in_pair,
                        seed=args.seed + next_pair,
                        layout=args.layout,
                        max_turns=args.max_turns,
                    )
                    futures[executor.submit(_run_match, task)] = task
                next_pair += 1

            completed = 0
            # consume completed futures and keep refilling until time expires
            while futures:
                for future in as_completed(list(futures)):
                    task = futures.pop(future)
                    result = future.result()
                    completed += 1
                    results.append(result)
                    print(
                        f"[{completed:4d}] pair={result.pair:4d} "
                        f"game={result.game_in_pair} turns={result.turns:4d} "
                        f"points={result.points}",
                        flush=True,
                    )
                    # refill if there's still time remaining
                    if time.perf_counter() < end_time:
                        for game_in_pair in (0, 1):
                            task = MatchTask(
                                pair=next_pair,
                                game_in_pair=game_in_pair,
                                seed=args.seed + next_pair,
                                layout=args.layout,
                                max_turns=args.max_turns,
                            )
                            futures[executor.submit(_run_match, task)] = task
                        next_pair += 1

    summary = summarize(results, args.seed)
    report = {
        "candidate": asdict(candidate),
        "baseline": asdict(baseline),
        "layout": args.layout,
        "seed": args.seed,
        "max_turns": args.max_turns,
        "wall_seconds": time.perf_counter() - started,
        "summary": summary,
        "matches": [
            asdict(result)
            for result in sorted(results, key=lambda item: (item.pair, item.game_in_pair))
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

