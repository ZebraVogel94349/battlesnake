from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import platform
import random
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass
from pathlib import Path

import hisss
import numpy as np
import torch
from hisss.cpp.lib import CPP_LIB

from battlesnake_types import BaseAgent, Direction, GameState
from obs_config import make_game_config
from ppo3 import PPOAgent3
from ppo4 import DEFAULT_MODEL_PATH, PPOAgent4
from ppo5 import DEFAULT_PPO5_MODEL_PATH, PPOAgent5
from ppo_mcts import DEFAULT_MCTS_MODEL_PATH, PPOMCTSAgent


DIRECTION_TO_HISSS = {
    Direction.UP: hisss.UP,
    Direction.DOWN: hisss.DOWN,
    Direction.LEFT: hisss.LEFT,
    Direction.RIGHT: hisss.RIGHT,
}
PLACEMENT_POINTS = (2.0, 1.0, 0.0, 0.0)
REPORT_TYPE = "evaluate-inference"
REPORT_SCHEMA_VERSION = 2
MATCH_RNG = "turn-reseeded-common-random-numbers-v1"
METRIC_SEMANTICS_VERSION = "blackout-evaluation-metrics-v2"
METRIC_SEMANTICS = {
    "placement": {
        "awards": list(PLACEMENT_POINTS),
        "ties": "average occupied placement slots at equal elimination turn",
        "winner": "the unique non-eliminated seat, otherwise null",
    },
    "top2": "fractional credit when a tie crosses the second-place boundary",
    "paired_delta": "candidate_points/8 - baseline_points/8 per two-game pair",
    "known_fatal": (
        "walls, visible bodies, and starvation are known fatal; in Blackout an "
        "enemy tail is never assumed to vacate from its redacted length"
    ),
    "latency": "synchronous wall time around agent.move, in milliseconds",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return "unavailable"


def _named_file_fingerprint(
    files: list[tuple[str, Path]],
) -> tuple[str, dict[str, str]]:
    composite = hashlib.sha256()
    components: dict[str, str] = {}
    available = True
    for name, path in sorted(files):
        fingerprint = _file_sha256(path)
        components[name] = fingerprint
        available = available and fingerprint != "unavailable"
        label = name.encode("utf-8", errors="surrogatepass")
        composite.update(len(label).to_bytes(4, byteorder="big", signed=False))
        composite.update(label)
        if fingerprint != "unavailable":
            composite.update(bytes.fromhex(fingerprint))
    return (composite.hexdigest() if available else "unavailable"), components


def _hisss_version() -> str:
    try:
        return importlib.metadata.version("hisss")
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(hisss, "__version__", "unknown"))


def _build_hisss_provenance() -> dict[str, str]:
    module_file = Path(hisss.__file__).resolve()
    package_root = module_file.parent
    python_files = [
        (path.relative_to(package_root).as_posix(), path)
        for path in package_root.rglob("*.py")
        if path.is_file()
    ]
    python_sha, _ = _named_file_fingerprint(python_files)
    native_name = getattr(CPP_LIB.lib, "_name", None)
    native_sha = (
        _file_sha256(Path(native_name).resolve())
        if isinstance(native_name, (str, os.PathLike)) and native_name
        else "unavailable"
    )
    version = _hisss_version()
    implementation = {
        "version": version,
        "python_sources_sha256": python_sha,
        "native_library_sha256": native_sha,
    }
    implementation_sha = (
        hashlib.sha256(_canonical_json_bytes(implementation)).hexdigest()
        if python_sha != "unavailable" and native_sha != "unavailable"
        else "unavailable"
    )
    return {**implementation, "implementation_sha256": implementation_sha}


_EVALUATOR_CODE_SHA256, _EVALUATOR_COMPONENT_SHA256 = _named_file_fingerprint(
    [
        ("evaluate_inference.py", Path(__file__).resolve()),
        ("obs_config.py", Path(__file__).resolve().with_name("obs_config.py")),
        (
            "battlesnake_types.py",
            Path(__file__).resolve().with_name("battlesnake_types.py"),
        ),
    ]
)
METRIC_SEMANTICS_SHA256 = hashlib.sha256(
    _canonical_json_bytes(METRIC_SEMANTICS)
).hexdigest()
_HISSS_PROVENANCE = _build_hisss_provenance()


def evaluation_provenance() -> dict[str, object]:
    """Return stable provenance captured when this evaluator was imported."""

    return {
        "report_schema": {
            "name": REPORT_TYPE,
            "version": REPORT_SCHEMA_VERSION,
        },
        "evaluator": {
            "code_sha256": _EVALUATOR_CODE_SHA256,
            "component_sha256": dict(_EVALUATOR_COMPONENT_SHA256),
        },
        "metric_semantics": {
            "version": METRIC_SEMANTICS_VERSION,
            "sha256": METRIC_SEMANTICS_SHA256,
            "definitions": copy.deepcopy(METRIC_SEMANTICS),
        },
        "hisss": dict(_HISSS_PROVENANCE),
    }


@dataclass(frozen=True)
class AgentSpec:
    kind: str
    model: str
    time_budget_ms: float | None = None
    max_iterations: int | None = None
    min_iterations: int | None = None


@dataclass(frozen=True)
class MatchTask:
    pair: int
    game_in_pair: int
    seed: int
    layout: str
    max_turns: int


@dataclass
class MatchResult:
    pair: int
    game_in_pair: int
    seed: int
    seat_kinds: list[str]
    points: list[float]
    turns: int
    timed_out: bool
    winner: int | None
    elimination_causes: dict[int, str]
    move_latencies_ms: dict[str, list[float]]
    fatal_moves: dict[str, int]
    avoidable_fatal_moves: dict[str, int]
    search_stats: dict[str, dict[str, object]]
    agent_diagnostics: dict[str, dict[str, object]]


_CANDIDATE_TEMPLATE: BaseAgent | None = None
_BASELINE_TEMPLATE: BaseAgent | None = None


def _make_agent(spec: AgentSpec) -> BaseAgent:
    _validate_agent_spec(spec)
    if spec.kind == "ppo-mcts":
        kwargs = {}
        if spec.time_budget_ms is not None:
            kwargs["time_budget_ms"] = spec.time_budget_ms
        if spec.max_iterations is not None:
            kwargs["max_iterations"] = spec.max_iterations
        if spec.min_iterations is not None:
            kwargs["min_iterations"] = spec.min_iterations
        return PPOMCTSAgent(spec.model, **kwargs)
    if spec.kind == "ppo3":
        return PPOAgent3(spec.model, space_mask=True)
    if spec.kind == "ppo3-no-space":
        return PPOAgent3(spec.model, space_mask=False)
    if spec.kind == "ppo4-policy":
        return PPOAgent4(spec.model, symmetries=1, safety_search=False)
    if spec.kind == "ppo4-search":
        return PPOAgent4(spec.model, symmetries=1, safety_search=True)
    if spec.kind == "ppo4-tta":
        return PPOAgent4(spec.model, symmetries=8, safety_search=False)
    if spec.kind == "ppo4":
        return PPOAgent4(spec.model, symmetries=1, safety_search=True)
    if spec.kind == "ppo5":
        return PPOAgent5(spec.model, symmetries=1, safety_search=True)
    raise ValueError(f"unknown agent kind: {spec.kind}")


def _validate_agent_spec(spec: AgentSpec) -> None:
    if spec.time_budget_ms is not None and (
        not math.isfinite(spec.time_budget_ms) or spec.time_budget_ms < 0.0
    ):
        raise ValueError("time-budget-ms must be finite and non-negative")
    for name, value in (
        ("max-iterations", spec.max_iterations),
        ("min-iterations", spec.min_iterations),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative")


def _worker_init(candidate: AgentSpec, baseline: AgentSpec):
    global _CANDIDATE_TEMPLATE, _BASELINE_TEMPLATE
    torch.set_num_threads(1)
    _CANDIDATE_TEMPLATE = _make_agent(candidate)
    _BASELINE_TEMPLATE = _make_agent(baseline)


def _clone_agent(template: BaseAgent) -> BaseAgent:
    agent = copy.copy(template)
    if isinstance(agent, PPOMCTSAgent):
        agent._sessions = {}
        agent._mcts_sessions = {}
        agent._session_locks = {}
        agent._lock = threading.RLock()
    elif isinstance(agent, PPOAgent4):
        agent._sessions = {}
        agent._lock = threading.RLock()
    elif isinstance(agent, PPOAgent3):
        agent.env = None
        agent._reset_memory()
    return agent


def _seat_kinds(layout: str, game_in_pair: int) -> list[str]:
    if layout == "copies":
        return (
            ["candidate", "baseline", "candidate", "baseline"]
            if game_in_pair == 0
            else ["baseline", "candidate", "baseline", "candidate"]
        )
    if layout == "solo-pair":
        return (
            ["candidate", "baseline", "baseline", "baseline"]
            if game_in_pair == 0
            else ["baseline", "candidate", "candidate", "candidate"]
        )
    raise ValueError(f"unknown layout: {layout}")


def _state_for(env, player: int, game_id: str) -> GameState:
    state_dict = json.loads(hisss.to_battlesnake_json(env, player))
    state_dict["game"]["id"] = game_id
    return GameState.model_validate(state_dict)


def _known_fatal_reason(state: GameState, direction: Direction) -> str | None:
    head = state.you.head
    if head is None:
        return None
    width, height = state.board.width, state.board.height
    target = head.x + direction.dx, head.y + direction.dy
    if not (0 <= target[0] < width and 0 <= target[1] < height):
        return "wall"

    def cells(body):
        return [
            (point.x, point.y)
            for point in body
            if point is not None and 0 <= point.x < width and 0 <= point.y < height
        ]

    own = list(dict.fromkeys(cells(state.you.body)))
    tail_vacates = len(own) >= state.you.length
    if target in set(own[:-1] if tail_vacates else own):
        return "self"
    blackout = (
        state.game.ruleset.name == "blackout"
        or state.game.ruleset.settings.viewRadius is not None
    )
    for snake in state.board.snakes:
        if snake.id == state.you.id:
            continue
        valid = cells(snake.body)
        fully_visible = all(
            point is not None and 0 <= point.x < width and 0 <= point.y < height
            for point in snake.body
        )
        # Blackout rewrites an opponent's API ``length`` to the restricted
        # body-list length.  Even if every returned point is currently visible,
        # that value cannot prove that the physical tail is known or vacating.
        tail_vacates = (
            not blackout and fully_visible and len(valid) >= snake.length
        )
        blocked = valid[:-1] if tail_vacates else valid
        if target in set(blocked):
            return "enemy-body"
    food = {(point.x, point.y) for point in state.board.food}
    if (state.you.health or 0) <= 1 and target not in food:
        return "health"
    return None


_VALID_PLACEMENT_MULTISETS = (
    (2.0, 1.0, 0.0, 0.0),
    (2.0, 0.5, 0.5, 0.0),
    (2.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (1.5, 1.5, 0.0, 0.0),
    (1.0, 1.0, 1.0, 0.0),
    (0.75, 0.75, 0.75, 0.75),
)


def validate_placement_outcome(
    points: list[float],
    winner: int | None,
    elimination_causes: dict[int, str],
    timed_out: bool,
    *,
    context: str = "match",
) -> None:
    """Validate the four-seat placement and survivor/winner relationship."""

    if len(points) != 4:
        raise ValueError(f"{context}: placement must contain four scores")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in points
    ):
        raise ValueError(f"{context}: placement scores must be finite numbers")
    ordered = tuple(sorted((float(value) for value in points), reverse=True))
    if not any(
        all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
            for actual, expected in zip(ordered, allowed)
        )
        for allowed in _VALID_PLACEMENT_MULTISETS
    ):
        raise ValueError(
            f"{context}: scores are not a valid tied allocation of {PLACEMENT_POINTS}"
        )

    eliminated = set(elimination_causes)
    if any(
        isinstance(seat, bool) or not isinstance(seat, int) or seat not in range(4)
        for seat in eliminated
    ):
        raise ValueError(f"{context}: elimination causes contain an invalid seat")
    if isinstance(winner, bool) or (
        winner is not None and (not isinstance(winner, int) or winner not in range(4))
    ):
        raise ValueError(f"{context}: winner must be null or a seat in 0..3")
    if not isinstance(timed_out, bool):
        raise ValueError(f"{context}: timed_out must be boolean")

    alive = set(range(4)) - eliminated
    expected_winner = next(iter(alive)) if len(alive) == 1 else None
    if winner != expected_winner:
        raise ValueError(
            f"{context}: winner {winner!r} disagrees with surviving seats "
            f"{sorted(alive)}"
        )
    if alive:
        best = max(points)
        top = {
            seat
            for seat, value in enumerate(points)
            if math.isclose(value, best, rel_tol=0.0, abs_tol=1e-9)
        }
        if top != alive:
            raise ValueError(
                f"{context}: top-scoring seats {sorted(top)} disagree with "
                f"surviving seats {sorted(alive)}"
            )
    expected_timeout = len(alive) >= 2
    if timed_out != expected_timeout:
        raise ValueError(
            f"{context}: timed_out={timed_out} disagrees with surviving seats "
            f"{sorted(alive)}"
        )


def _placement_points(
    state,
    *,
    timed_out: bool,
) -> tuple[list[float], int | None, dict[int, str]]:
    if len(state.snakes_alive) != len(PLACEMENT_POINTS):
        raise ValueError(
            f"placement expects {len(PLACEMENT_POINTS)} seats, "
            f"got {len(state.snakes_alive)}"
        )
    elimination = state.elimination_events or {}
    if any(seat not in range(len(PLACEMENT_POINTS)) for seat in elimination):
        raise ValueError("elimination events contain an invalid seat")
    rank_values: dict[int, float] = {}
    causes: dict[int, str] = {}
    for seat, alive in enumerate(state.snakes_alive):
        if alive:
            rank_values[seat] = math.inf
        else:
            event = elimination.get(seat)
            rank_values[seat] = float(event.turn if event is not None else -1)
            causes[seat] = event.cause if event is not None else "unknown"

    ordered_values = sorted(set(rank_values.values()), reverse=True)
    points = [0.0] * 4
    rank = 0
    for value in ordered_values:
        seats = [seat for seat, cur in rank_values.items() if cur == value]
        awards = PLACEMENT_POINTS[rank : rank + len(seats)]
        award = sum(awards) / len(seats) if seats else 0.0
        for seat in seats:
            points[seat] = award
        rank += len(seats)
    alive = [seat for seat, is_alive in enumerate(state.snakes_alive) if is_alive]
    winner = alive[0] if len(alive) == 1 else None
    validate_placement_outcome(
        points,
        winner,
        causes,
        timed_out,
        context="simulator placement",
    )
    return points, winner, causes


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
    agent_diagnostics = {}
    for seat, kind in enumerate(seat_kinds):
        if kind in agent_diagnostics:
            continue
        diagnostics_fn = getattr(agents[seat], "get_diagnostics", None)
        diagnostics = diagnostics_fn() if callable(diagnostics_fn) else {}
        agent_diagnostics[kind] = diagnostics if isinstance(diagnostics, dict) else {}
    cfg = make_game_config()
    cfg.all_actions_legal = True
    env = hisss.BattleSnakeGame(cfg)
    game_id = f"inference-eval-{task.pair}-{task.game_in_pair}-{task.seed}"
    latencies = {"candidate": [], "baseline": []}
    fatal_moves = {"candidate": 0, "baseline": 0}
    avoidable_fatal_moves = {"candidate": 0, "baseline": 0}
    search_stats = {
        kind: {
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
        for kind in ("candidate", "baseline")
    }
    turns = 0
    try:
        for seat, agent in enumerate(agents):
            agent.start(_state_for(env, seat, game_id))

        while not env.is_terminal() and turns < task.max_turns:
            actions = []
            for seat in env.players_at_turn():
                state = _state_for(env, seat, game_id)
                start = time.perf_counter()
                result = agents[seat].move(state)
                latencies[seat_kinds[seat]].append(
                    (time.perf_counter() - start) * 1_000.0
                )
                stats_fn = getattr(agents[seat], "get_last_search_stats", None)
                stats = stats_fn() if callable(stats_fn) else None
                if isinstance(stats, dict):
                    totals = search_stats[seat_kinds[seat]]
                    totals["moves"] += 1.0
                    for key in ("iterations", "nodes", "depth", "elapsed_ms"):
                        value = stats.get(key, 0.0)
                        if isinstance(value, (int, float)) and math.isfinite(value):
                            totals[key] += float(value)
                    depth = stats.get("depth", 0.0)
                    if isinstance(depth, (int, float)) and math.isfinite(depth):
                        totals["max_depth"] = max(totals["max_depth"], float(depth))
                    totals["deadline_hits"] += float(bool(stats.get("deadline_hit")))
                    totals["errors"] += float(stats.get("error") is not None)
                    totals["mcts_choices"] += float(stats.get("fallback") == "mcts")
                    totals["insufficient_searches"] += float(
                        stats.get("fallback") == "insufficient-search"
                    )
                    totals["policy_changes"] += float(
                        bool(stats.get("changed_policy"))
                    )
                    ensemble_move = stats.get("search_mode") in {
                        "wave-ensemble",
                        "ensemble",
                    }
                    totals["ensemble_moves"] += float(ensemble_move)
                    for key_name in (
                        "waves_started",
                        "waves_completed",
                        "waves_discarded",
                        "voting_iterations",
                        "discarded_iterations",
                    ):
                        value = stats.get(key_name, 0.0)
                        if (
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(value)
                        ):
                            totals[key_name] += float(value)
                    totals["stopped_for_wave_guard"] += float(
                        ensemble_move and bool(stats.get("stopped_for_wave_guard"))
                    )
                    reason = stats.get("decision_reason")
                    if ensemble_move and isinstance(reason, str) and reason:
                        reason_counts = totals["decision_reason_counts"]
                        reason_counts[reason] = reason_counts.get(reason, 0.0) + 1.0
                if _known_fatal_reason(state, result.move) is not None:
                    fatal_moves[seat_kinds[seat]] += 1
                    if any(
                        _known_fatal_reason(state, direction) is None
                        for direction in Direction
                    ):
                        avoidable_fatal_moves[seat_kinds[seat]] += 1
                actions.append(DIRECTION_TO_HISSS[result.move])
            # The native simulator currently owns one process-global food RNG.
            # Agent-side helper environments (notably PPO3) must not perturb the
            # actual match.  Re-seeding immediately before the authoritative
            # step gives both games in a paired seed common random numbers.
            turn_seed = (task.seed * 1_000_003 + turns * 97_409 + 17) & 0x7FFFFFFF
            CPP_LIB.lib.set_seed(turn_seed)
            env.step(tuple(actions))
            turns += 1

        for seat, agent in enumerate(agents):
            agent.end(_state_for(env, seat, game_id))
        timed_out = not env.is_terminal()
        points, winner, causes = _placement_points(
            env.get_state(),
            timed_out=timed_out,
        )
        return MatchResult(
            pair=task.pair,
            game_in_pair=task.game_in_pair,
            seed=task.seed,
            seat_kinds=seat_kinds,
            points=points,
            turns=turns,
            timed_out=timed_out,
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


def _run_pair(
    pair: int,
    seed: int,
    layout: str,
    max_turns: int,
) -> list[MatchResult]:
    """Run both seat rotations atomically for duration-based evaluations."""

    return [
        _run_match(
            MatchTask(
                pair=pair,
                game_in_pair=game_in_pair,
                seed=seed,
                layout=layout,
                max_turns=max_turns,
            )
        )
        for game_in_pair in range(2)
    ]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), percentile))


def _top_k_credits(points: list[float], k: int) -> list[float]:
    """Fractionally assign top-k slots when a placement boundary is tied."""

    if k < 0:
        raise ValueError("k must be non-negative")
    credits = [0.0] * len(points)
    rank = 0
    for score in sorted(set(points), reverse=True):
        tied = [seat for seat, value in enumerate(points) if value == score]
        slots = min(len(tied), max(0, k - rank))
        credit = slots / len(tied)
        for seat in tied:
            credits[seat] = credit
        rank += len(tied)
    return credits


def _bootstrap_ci(
    values: list[float], seed: int, samples: int = 20_000
) -> tuple[float, float] | None:
    # Resampling a single pair only reproduces the point estimate and conveys
    # no uncertainty.  Report it as unavailable instead of a zero-width CI.
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(data), size=(samples, len(data)))
    means = data[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(results: list[MatchResult], seed: int) -> dict[str, object]:
    results.sort(key=lambda result: (result.pair, result.game_in_pair))
    points = {"candidate": 0.0, "baseline": 0.0}
    seats = {"candidate": 0, "baseline": 0}
    wins = Counter()
    top2 = Counter()
    causes = {"candidate": Counter(), "baseline": Counter()}
    latency = {"candidate": [], "baseline": []}
    fatal = Counter()
    avoidable_fatal = Counter()
    search = {
        "candidate": Counter(),
        "baseline": Counter(),
    }
    decision_reasons = {
        "candidate": Counter(),
        "baseline": Counter(),
    }
    pair_totals: dict[int, dict[str, float]] = {}

    for result in results:
        validate_placement_outcome(
            result.points,
            result.winner,
            result.elimination_causes,
            result.timed_out,
            context=f"pair {result.pair}/game {result.game_in_pair}",
        )
        pair_totals.setdefault(result.pair, {"candidate": 0.0, "baseline": 0.0})
        top2_credits = _top_k_credits(result.points, 2)
        for seat, kind in enumerate(result.seat_kinds):
            seats[kind] += 1
            points[kind] += result.points[seat]
            pair_totals[result.pair][kind] += result.points[seat]
            top2[kind] += top2_credits[seat]
            if result.winner == seat:
                wins[kind] += 1
            if seat in result.elimination_causes:
                causes[kind][result.elimination_causes[seat]] += 1
        for kind in latency:
            latency[kind].extend(result.move_latencies_ms[kind])
            fatal[kind] += result.fatal_moves[kind]
            avoidable_fatal[kind] += result.avoidable_fatal_moves[kind]
            match_search = result.search_stats[kind]
            for key, value in match_search.items():
                if key == "decision_reason_counts":
                    if isinstance(value, dict):
                        decision_reasons[kind].update(value)
                elif (
                    key != "max_depth"
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                ):
                    search[kind][key] += value
            search[kind]["max_depth"] = max(
                search[kind]["max_depth"], match_search.get("max_depth", 0.0)
            )

    pair_deltas = []
    for pair in sorted(pair_totals):
        # Every two-game block contains four seats of each kind under both
        # supported layouts. Normalize each seat's 2/1/0 score to [0,1].
        cur = pair_totals[pair]
        pair_deltas.append(cur["candidate"] / 8.0 - cur["baseline"] / 8.0)
    ci = _bootstrap_ci(pair_deltas, seed)

    summary: dict[str, object] = {
        "games": len(results),
        "pairs": len(pair_deltas),
        "average_turns": statistics.fmean(result.turns for result in results),
        "timeouts": sum(result.timed_out for result in results),
        "paired_normalized_score_delta": statistics.fmean(pair_deltas),
        "paired_delta_ci95": list(ci) if ci is not None else None,
        "fatal_moves": dict(fatal),
        "avoidable_fatal_moves": dict(avoidable_fatal),
        "agents": {},
    }
    for kind in ("candidate", "baseline"):
        values = latency[kind]
        move_count = len(values)
        summary["agents"][kind] = {
            "seats": seats[kind],
            "normalized_score": points[kind] / (2.0 * seats[kind]),
            "wins": wins[kind],
            "win_rate_per_game": wins[kind] / len(results),
            "top2_rate": top2[kind] / seats[kind],
            "moves": move_count,
            "known_fatal_moves": int(fatal[kind]),
            "known_fatal_rate_per_move": (
                fatal[kind] / move_count if move_count else 0.0
            ),
            "avoidable_known_fatal_moves": int(avoidable_fatal[kind]),
            "avoidable_known_fatal_rate_per_move": (
                avoidable_fatal[kind] / move_count if move_count else 0.0
            ),
            "elimination_causes": dict(causes[kind]),
            "latency_ms": {
                "mean": statistics.fmean(values) if values else 0.0,
                "p50": _percentile(values, 50),
                "p95": _percentile(values, 95),
                "p99": _percentile(values, 99),
                "max": max(values, default=0.0),
                "over_450ms": sum(value > 450.0 for value in values),
                "over_500ms": sum(value > 500.0 for value in values),
            },
        }
        search_moves = search[kind]["moves"]
        if search_moves:
            ensemble_moves = search[kind]["ensemble_moves"]
            reason_counts = {
                reason: int(count)
                for reason, count in sorted(decision_reasons[kind].items())
            }
            summary["agents"][kind]["search"] = {
                "moves": int(search_moves),
                "iterations_per_move": search[kind]["iterations"] / search_moves,
                "nodes_per_move": search[kind]["nodes"] / search_moves,
                "mean_depth": search[kind]["depth"] / search_moves,
                "max_depth": int(search[kind]["max_depth"]),
                "mean_elapsed_ms": search[kind]["elapsed_ms"] / search_moves,
                "deadline_hit_rate": search[kind]["deadline_hits"] / search_moves,
                "errors": int(search[kind]["errors"]),
                "mcts_choice_rate": search[kind]["mcts_choices"] / search_moves,
                "insufficient_searches": int(
                    search[kind]["insufficient_searches"]
                ),
                "insufficient_search_rate": (
                    search[kind]["insufficient_searches"] / search_moves
                ),
                "policy_change_rate": search[kind]["policy_changes"] / search_moves,
                "ensemble_moves": int(ensemble_moves),
                "ensemble_move_rate": ensemble_moves / search_moves,
                "waves_started": int(search[kind]["waves_started"]),
                "waves_started_per_move": (
                    search[kind]["waves_started"] / search_moves
                ),
                "waves_completed": int(search[kind]["waves_completed"]),
                "waves_completed_per_move": (
                    search[kind]["waves_completed"] / search_moves
                ),
                "waves_discarded": int(search[kind]["waves_discarded"]),
                "waves_discarded_per_move": (
                    search[kind]["waves_discarded"] / search_moves
                ),
                "voting_iterations": int(search[kind]["voting_iterations"]),
                "voting_iterations_per_move": (
                    search[kind]["voting_iterations"] / search_moves
                ),
                "discarded_iterations": int(
                    search[kind]["discarded_iterations"]
                ),
                "discarded_iterations_per_move": (
                    search[kind]["discarded_iterations"] / search_moves
                ),
                "stopped_for_wave_guard": int(
                    search[kind]["stopped_for_wave_guard"]
                ),
                "stopped_for_wave_guard_rate": (
                    search[kind]["stopped_for_wave_guard"] / ensemble_moves
                    if ensemble_moves
                    else 0.0
                ),
                "decision_reason_counts": reason_counts,
                "decision_reason_rates": {
                    reason: count / ensemble_moves if ensemble_moves else 0.0
                    for reason, count in reason_counts.items()
                },
            }
    return summary


def _runtime_metadata() -> dict[str, object]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_devices: list[str] = []
    if cuda_available:
        try:
            cuda_devices = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
        except (AssertionError, RuntimeError):
            # Metadata must never make an otherwise valid evaluation fail.
            cuda_devices = ["unavailable"]

    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "torch": {
            "version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
        },
        "hisss": dict(_HISSS_PROVENANCE),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "cuda_available": cuda_available,
            "cuda_devices": cuda_devices,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired, API-faithful evaluation of PPO inference agents."
    )
    kinds = (
        "ppo-mcts",
        "ppo3",
        "ppo3-no-space",
        "ppo4-policy",
        "ppo4-search",
        "ppo4-tta",
        "ppo4",
        "ppo5",
    )
    parser.add_argument("--candidate", choices=kinds, default="ppo5")
    parser.add_argument("--baseline", choices=kinds, default="ppo4")
    parser.add_argument(
        "--candidate-model",
        help="Checkpoint (default: PPO_MODEL_PATH or the promoted champion).",
    )
    parser.add_argument(
        "--baseline-model",
        help="Checkpoint (default: PPO_MODEL_PATH or the promoted champion).",
    )
    parser.add_argument(
        "--candidate-time-ms",
        type=float,
        help="Override the candidate MCTS wall-clock budget per move.",
    )
    parser.add_argument(
        "--baseline-time-ms",
        type=float,
        help="Override the baseline MCTS wall-clock budget per move.",
    )
    parser.add_argument("--candidate-max-iterations", type=int)
    parser.add_argument("--baseline-max-iterations", type=int)
    parser.add_argument(
        "--candidate-min-iterations",
        type=int,
        help="Minimum completed MCTS simulations before overriding PPO.",
    )
    parser.add_argument(
        "--baseline-min-iterations",
        type=int,
        help="Minimum completed MCTS simulations before overriding PPO.",
    )
    parser.add_argument("--layout", choices=("copies", "solo-pair"), default="copies")
    parser.add_argument(
        "--pairs",
        type=int,
        help=(
            "Run exactly this many pairs instead of the default time-based "
            "one-hour evaluation."
        ),
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=3_600.0,
        help=(
            "Stop launching new pairs after this many seconds (default: 3600). "
            "Already running pairs finish cleanly. Ignored with --pairs."
        ),
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max-turns", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=814_271)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _run_fixed_evaluation(
    args: argparse.Namespace,
    candidate: AgentSpec,
    baseline: AgentSpec,
) -> list[MatchResult]:
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
    context = mp.get_context("spawn")
    results: list[MatchResult] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
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
    return results


def _run_duration_evaluation(
    args: argparse.Namespace,
    candidate: AgentSpec,
    baseline: AgentSpec,
    started: float,
) -> list[MatchResult]:
    """Keep workers supplied with complete pairs until the soft deadline."""

    deadline = started + args.duration_seconds
    context = mp.get_context("spawn")
    results: list[MatchResult] = []
    next_pair = 0
    completed_pairs = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_worker_init,
        initargs=(candidate, baseline),
    ) as executor:
        pending = {}

        def submit_pair() -> None:
            nonlocal next_pair
            future = executor.submit(
                _run_pair,
                next_pair,
                args.seed + next_pair,
                args.layout,
                args.max_turns,
            )
            pending[future] = next_pair
            next_pair += 1

        while len(pending) < args.workers and time.perf_counter() < deadline:
            submit_pair()

        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                pair = pending.pop(future)
                pair_results = future.result()
                results.extend(pair_results)
                completed_pairs += 1
                elapsed = time.perf_counter() - started
                turns = sum(result.turns for result in pair_results)
                print(
                    f"[pair {completed_pairs:4d}] id={pair:4d} "
                    f"games=2 turns={turns:4d} elapsed={elapsed:7.1f}s",
                    flush=True,
                )
                if time.perf_counter() < deadline:
                    submit_pair()
    return results


def _default_duration_output() -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    return Path(__file__).with_name("eval_results") / (
        f"ppo5_vs_ppo4_1h_{timestamp}.json"
    )


def main() -> None:
    args = parse_args()
    if args.pairs is not None and args.pairs <= 0:
        raise ValueError("pairs must be positive")
    if args.workers <= 0 or args.max_turns <= 0:
        raise ValueError("workers and max-turns must be positive")
    if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0.0:
        raise ValueError("duration-seconds must be finite and positive")
    for name in ("candidate_time_ms", "baseline_time_ms"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(
                f"{name.replace('_', '-')} must be finite and non-negative"
            )
    for name in (
        "candidate_max_iterations",
        "baseline_max_iterations",
        "candidate_min_iterations",
        "baseline_min_iterations",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"{name.replace('_', '-')} must be non-negative")
    default_model = os.environ.get("PPO_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    candidate_default_model = (
        os.environ.get("PPO_MCTS_MODEL_PATH", default_model)
        if args.candidate == "ppo-mcts"
        else (
            os.environ.get("PPO5_MODEL_PATH", str(DEFAULT_PPO5_MODEL_PATH))
            if args.candidate == "ppo5" and "PPO_MODEL_PATH" not in os.environ
            else default_model
        )
    )
    if args.candidate == "ppo-mcts" and "PPO_MODEL_PATH" not in os.environ:
        candidate_default_model = os.environ.get(
            "PPO_MCTS_MODEL_PATH", str(DEFAULT_MCTS_MODEL_PATH)
        )
    baseline_default_model = (
        os.environ.get("PPO_MCTS_MODEL_PATH", default_model)
        if args.baseline == "ppo-mcts"
        else (
            os.environ.get("PPO5_MODEL_PATH", str(DEFAULT_PPO5_MODEL_PATH))
            if args.baseline == "ppo5" and "PPO_MODEL_PATH" not in os.environ
            else default_model
        )
    )
    if args.baseline == "ppo-mcts" and "PPO_MODEL_PATH" not in os.environ:
        baseline_default_model = os.environ.get(
            "PPO_MCTS_MODEL_PATH", str(DEFAULT_MCTS_MODEL_PATH)
        )
    candidate = AgentSpec(
        args.candidate,
        str(Path(args.candidate_model or candidate_default_model).resolve()),
        args.candidate_time_ms,
        args.candidate_max_iterations,
        args.candidate_min_iterations,
    )
    baseline = AgentSpec(
        args.baseline,
        str(Path(args.baseline_model or baseline_default_model).resolve()),
        args.baseline_time_ms,
        args.baseline_max_iterations,
        args.baseline_min_iterations,
    )
    _validate_agent_spec(candidate)
    _validate_agent_spec(baseline)
    started = time.perf_counter()
    duration_mode = args.pairs is None
    if duration_mode and args.output is None:
        args.output = _default_duration_output()
    mode_description = (
        f"duration={args.duration_seconds:.0f}s"
        if duration_mode
        else f"pairs={args.pairs}"
    )
    print(
        f"Starting {args.candidate} vs {args.baseline}: {mode_description}, "
        f"workers={args.workers}, layout={args.layout}\n"
        f"candidate model: {candidate.model}\n"
        f"baseline model:  {baseline.model}\n"
        f"output: {args.output or 'stdout only'}",
        flush=True,
    )
    results = (
        _run_duration_evaluation(args, candidate, baseline, started)
        if duration_mode
        else _run_fixed_evaluation(args, candidate, baseline)
    )

    summary = summarize(results, args.seed)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "evaluation_provenance": evaluation_provenance(),
        "candidate": asdict(candidate),
        "baseline": asdict(baseline),
        "layout": args.layout,
        "seed": args.seed,
        "max_turns": args.max_turns,
        "workers": args.workers,
        "evaluation_mode": "duration" if duration_mode else "fixed-pairs",
        "target_duration_seconds": args.duration_seconds if duration_mode else None,
        "runtime": _runtime_metadata(),
        "match_rng": MATCH_RNG,
        "effective_agent_diagnostics": (
            results[0].agent_diagnostics if results else {}
        ),
        "wall_seconds": time.perf_counter() - started,
        "summary": summary,
        "matches": [
            asdict(result)
            for result in sorted(results, key=lambda r: (r.pair, r.game_in_pair))
        ],
    }
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
