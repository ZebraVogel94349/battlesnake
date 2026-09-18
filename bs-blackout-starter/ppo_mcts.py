"""Deadline-bound information-set MCTS guided by the PPO4 actor.

The deployable ``.zip`` contains a faithful actor but not the privileged
training critic.  Consequently this module never asks the ZIP model for a
value: PPO supplies the root PUCT prior and the anytime fallback, while all
rollout values are terminal or geometric values from :mod:`mcts_simulator`.

The tree is deliberately open loop.  A node represents our action history,
not one guessed full board.  Every simulation starts from one of several
freshly sampled Blackout determinizations, so edge statistics are aggregated
over the information set instead of over a single invented hidden state.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from battlesnake_types import Direction, GameState, MoveAction
from mcts_simulator import (
    ACTION_DELTAS as SIM_ACTION_DELTAS,
    WorldState,
    determinize_world,
    evaluate_world,
    heuristic_action_priors,
    is_terminal,
    legal_actions,
    state_key,
    step_world,
)
from ppo4 import (
    ACTION_DELTAS,
    MODEL_ACTIONS,
    PPOAgent4,
    _DEFAULT_MODEL,
    _Session,
    encode_observation,
)


DEFAULT_TIME_BUDGET_MS = 400.0
DEFAULT_MAX_ITERATIONS = 50_000
DEFAULT_MIN_ITERATIONS = 24
# The selected production ensemble uses short, independently seeded roots.
# Three complete 16-simulation trees gave the best same-seed development
# result while retaining substantially more HTTP headroom than a monolithic
# deadline-sized tree.  The fixed <=32 simulation path below remains a
# separate compatibility mode for model screens.
DEFAULT_WAVE_ITERATIONS = 16
DEFAULT_MIN_COMPLETE_WAVES = 3
LEGACY_SINGLE_TREE_MAX_ITERATIONS = 32
DEFAULT_HORIZON = 12
DEFAULT_PARTICLES = 24
DEFAULT_ROLLOUT_STEPS = 2
DEFAULT_C_PUCT = 1.35
DEFAULT_ADVERSARIAL_FRACTION = 0.20
DEFAULT_SEED = 0
DEFAULT_TIMEOUT_MARGIN_MS = 80.0
WAVE_GUARD_FACTOR = 1.35
WAVE_GUARD_BUFFER_SECONDS = 0.002
DEFAULT_MCTS_MODEL_PATH = Path(__file__).with_name(
    "ppo_bs_lstm_cuda_v25_targeted_finish_champion.zip"
)


def _first_env(names: Sequence[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return None


def _env_int(names: Sequence[str], default: int) -> int:
    value = _first_env(names)
    return default if value is None else int(value)


def _env_float(names: Sequence[str], default: float) -> float:
    value = _first_env(names)
    return default if value is None else float(value)


def _env_bool(names: Sequence[str], default: bool) -> bool:
    value = _first_env(names)
    if value is None:
        return default
    return value.lower() not in ("", "0", "false", "no", "off")


def _composite_fingerprint(paths: Sequence[Path]) -> str:
    """Fingerprint all executable policy/search sources, independent of location."""

    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            return "unavailable"
        component = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                component.update(chunk)
        label = path.name.encode("utf-8", errors="surrogatepass")
        digest.update(len(label).to_bytes(4, byteorder="big", signed=False))
        digest.update(label)
        digest.update(component.digest())
    return digest.hexdigest()[:12]


def _copy_session(session: _Session) -> _Session:
    """Copy recurrent state so an interrupted request cannot partially commit."""

    states = session.lstm_states
    if states is not None:
        # PyTorch's LSTM currently returns fresh tensors, but cloning here makes
        # the transaction independent of that implementation detail.
        states = tuple(state.detach().clone() for state in states)
    return _Session(
        lstm_states=states,
        episode_start=session.episode_start,
        last_turn=session.last_turn,
        last_move=session.last_move,
    )


def _masked_probabilities(
    values: np.ndarray | Sequence[float],
    actions: Iterable[int],
    *,
    logits: bool,
) -> np.ndarray:
    """Return a finite four-action distribution supported on ``actions``."""

    allowed = tuple(dict.fromkeys(int(action) for action in actions))
    result = np.zeros(4, dtype=np.float64)
    if not allowed:
        return result
    source = np.asarray(values, dtype=np.float64).reshape(-1)
    selected = np.asarray(
        [source[action] if action < source.size else math.nan for action in allowed],
        dtype=np.float64,
    )
    if logits:
        finite = np.isfinite(selected)
        if finite.any():
            floor = float(np.min(selected[finite])) - 50.0
            selected = np.where(finite, selected, floor)
            selected = np.exp(selected - float(np.max(selected)))
        else:
            selected = np.ones(len(allowed), dtype=np.float64)
    else:
        selected = np.where(np.isfinite(selected), np.maximum(0.0, selected), 0.0)
    total = float(selected.sum())
    if total <= 0.0 or not math.isfinite(total):
        selected.fill(1.0 / len(allowed))
    else:
        selected /= total
    result[list(allowed)] = selected
    return result


@dataclass(slots=True)
class _Edge:
    action: int
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    child: _Node | None = None

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(slots=True)
class _Node:
    visits: int = 0
    edges: dict[int, _Edge] = field(default_factory=dict)

    def ensure_edges(self, actions: Iterable[int], priors: np.ndarray) -> None:
        for action in actions:
            action = int(action)
            if action not in self.edges:
                self.edges[action] = _Edge(action=action, prior=float(priors[action]))


@dataclass(slots=True)
class SearchDiagnostics:
    iterations: int = 0
    nodes: int = 1
    depth: int = 0
    elapsed_ms: float = 0.0
    particles: int = 0
    adversarial_iterations: int = 0
    deadline: float = 0.0
    deadline_ms: float = 0.0
    deadline_hit: bool = False
    fallback: str = "ppo"
    policy: str = ""
    selected: str = ""
    changed_policy: bool = False
    root_prior: dict[str, float] = field(default_factory=dict)
    root_visits: dict[str, int] = field(default_factory=dict)
    root_q: dict[str, float] = field(default_factory=dict)
    search_mode: str = "single"
    waves: int = 0
    complete_waves: int = 0
    votes: dict[str, int] = field(default_factory=dict)
    waves_started: int = 0
    waves_completed: int = 0
    waves_discarded: int = 0
    wave_votes: dict[str, int] = field(default_factory=dict)
    voting_iterations: int = 0
    discarded_iterations: int = 0
    stopped_for_wave_guard: bool = False
    decision_reason: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class _MCTSSession:
    last_turn: int = -1
    last_search: SearchDiagnostics | None = None


class PPOMCTSAgent(PPOAgent4):
    """PPO4 policy plus open-loop, root-sampled information-set PUCT."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None | object = _DEFAULT_MODEL,
        *,
        time_budget_ms: float | None = None,
        max_iterations: int | None = None,
        min_iterations: int | None = None,
        wave_iterations: int | None = None,
        min_complete_waves: int | None = None,
        horizon: int | None = None,
        particles: int | None = None,
        rollout_steps: int | None = None,
        c_puct: float | None = None,
        adversarial_fraction: float | None = None,
        seed: int | None = None,
        timeout_margin_ms: float | None = None,
        symmetries: int | None = None,
        safety_search: bool | None = None,
        device: str | None = None,
    ):
        if model_path is _DEFAULT_MODEL:
            model_path = _first_env(("PPO_MCTS_MODEL_PATH", "PPO_MODEL_PATH"))
            if model_path is None:
                model_path = DEFAULT_MCTS_MODEL_PATH
        if time_budget_ms is None:
            time_budget_ms = _env_float(
                ("PPO_MCTS_TIME_BUDGET_MS", "MCTS_TIME_BUDGET_MS"),
                DEFAULT_TIME_BUDGET_MS,
            )
        if max_iterations is None:
            max_iterations = _env_int(
                ("PPO_MCTS_MAX_ITERATIONS", "MCTS_MAX_ITERATIONS"),
                DEFAULT_MAX_ITERATIONS,
            )
        if min_iterations is None:
            min_iterations = _env_int(
                ("PPO_MCTS_MIN_ITERATIONS", "MCTS_MIN_ITERATIONS"),
                DEFAULT_MIN_ITERATIONS,
            )
        if wave_iterations is None:
            wave_iterations = _env_int(
                ("PPO_MCTS_WAVE_ITERATIONS", "MCTS_WAVE_ITERATIONS"),
                DEFAULT_WAVE_ITERATIONS,
            )
        if min_complete_waves is None:
            min_complete_waves = _env_int(
                ("PPO_MCTS_MIN_COMPLETE_WAVES", "MCTS_MIN_COMPLETE_WAVES"),
                DEFAULT_MIN_COMPLETE_WAVES,
            )
        if horizon is None:
            horizon = _env_int(
                ("PPO_MCTS_HORIZON", "MCTS_HORIZON"), DEFAULT_HORIZON
            )
        if particles is None:
            particles = _env_int(
                ("PPO_MCTS_PARTICLES", "MCTS_PARTICLES"), DEFAULT_PARTICLES
            )
        if rollout_steps is None:
            rollout_steps = _env_int(
                ("PPO_MCTS_ROLLOUT_STEPS", "MCTS_ROLLOUT_STEPS"),
                DEFAULT_ROLLOUT_STEPS,
            )
        if c_puct is None:
            c_puct = _env_float(
                ("PPO_MCTS_C_PUCT", "MCTS_C_PUCT"), DEFAULT_C_PUCT
            )
        if adversarial_fraction is None:
            adversarial_fraction = _env_float(
                ("PPO_MCTS_ADVERSARIAL_FRACTION", "MCTS_ADVERSARIAL_FRACTION"),
                DEFAULT_ADVERSARIAL_FRACTION,
            )
        if seed is None:
            seed = _env_int(("PPO_MCTS_SEED", "MCTS_SEED"), DEFAULT_SEED)
        if timeout_margin_ms is None:
            timeout_margin_ms = _env_float(
                ("PPO_MCTS_TIMEOUT_MARGIN_MS", "MCTS_TIMEOUT_MARGIN_MS"),
                DEFAULT_TIMEOUT_MARGIN_MS,
            )
        if safety_search is None:
            # Retain PPO4's own-body DFS as conservative root pruning.  MCTS
            # then extends this safe base with stochastic opponent rollouts.
            safety_search = _env_bool(("PPO_MCTS_PPO_SAFETY_SEARCH",), True)

        if not math.isfinite(time_budget_ms) or time_budget_ms < 0:
            raise ValueError("time_budget_ms must be finite and non-negative")
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if min_iterations < 0:
            raise ValueError("min_iterations must be non-negative")
        if max_iterations > 0 and min_iterations > max_iterations:
            raise ValueError(
                "min_iterations must not exceed max_iterations unless "
                "max_iterations is 0 (PPO-only mode)"
            )
        if wave_iterations < 1:
            raise ValueError("wave_iterations must be positive")
        if min_complete_waves < 2:
            raise ValueError("min_complete_waves must be at least 2")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if particles < 1:
            raise ValueError("particles must be positive")
        if rollout_steps < 0:
            raise ValueError("rollout_steps must be non-negative")
        if not math.isfinite(c_puct) or c_puct < 0:
            raise ValueError("c_puct must be finite and non-negative")
        if (
            not math.isfinite(adversarial_fraction)
            or not 0.0 <= adversarial_fraction <= 1.0
        ):
            raise ValueError("adversarial_fraction must be finite and in [0, 1]")
        if not math.isfinite(timeout_margin_ms) or timeout_margin_ms < 0:
            raise ValueError("timeout_margin_ms must be finite and non-negative")

        super().__init__(
            model_path,
            symmetries=symmetries,
            safety_search=safety_search,
            device=device,
        )
        if tuple(ACTION_DELTAS) != tuple(SIM_ACTION_DELTAS):
            raise RuntimeError("PPO4 and MCTS simulator action orders differ")

        self.time_budget_ms = float(time_budget_ms)
        self.max_iterations = int(max_iterations)
        self.min_iterations = int(min_iterations)
        self.wave_iterations = int(wave_iterations)
        self.min_complete_waves = int(min_complete_waves)
        self.horizon = int(horizon)
        self.particles = int(particles)
        self.rollout_steps = min(int(rollout_steps), self.horizon)
        self.c_puct = float(c_puct)
        self.adversarial_fraction = float(adversarial_fraction)
        self.seed = int(seed)
        self.timeout_margin_ms = float(timeout_margin_ms)
        self._mcts_sessions: dict[tuple[str, str], _MCTSSession] = {}
        self._session_locks: dict[tuple[str, str], threading.RLock] = {}
        # Different games can search concurrently.  Only the short shared
        # PyTorch forward is serialized; recurrent tensors themselves live in
        # per-game transactions.
        self._model_lock = threading.RLock()
        self._last_search: SearchDiagnostics | None = None
        # PPOAgent4 fingerprints only its own module.  The deployed decision is
        # composed of the base policy wrapper, search, and simulator semantics.
        source_dir = Path(__file__).resolve().parent
        self._code_fingerprint = _composite_fingerprint(
            (
                source_dir / "ppo_mcts.py",
                source_dir / "mcts_simulator.py",
                source_dir / "ppo4.py",
            )
        )
        # Trigger lazy CPU/CUDA kernels before the first timed /move request.
        # The temporary recurrent state is intentionally discarded.
        with self._model_lock:
            self._policy_logits(
                np.zeros((9, 29, 29), dtype=np.float32),
                _Session(),
            )

    def get_name(self):
        return "Der Snaketürke PPO-MCTS"

    def get_diagnostics(self) -> dict[str, object]:
        result = super().get_diagnostics()
        result.update(
            {
                "time_budget_ms": self.time_budget_ms,
                "max_iterations": self.max_iterations,
                "min_iterations": self.min_iterations,
                "wave_iterations": self.wave_iterations,
                "min_complete_waves": self.min_complete_waves,
                "horizon": self.horizon,
                "particles": self.particles,
                "rollout_steps": self.rollout_steps,
                "c_puct": self.c_puct,
                "adversarial_fraction": self.adversarial_fraction,
                "seed": self.seed,
                "timeout_margin_ms": self.timeout_margin_ms,
                "device": self.device,
                "last_search": (
                    None if self._last_search is None else self._last_search.to_dict()
                ),
            }
        )
        return result

    def get_last_search_stats(self) -> dict[str, object] | None:
        """Return a detached snapshot suitable for evaluation/logging."""

        with self._lock:
            return None if self._last_search is None else self._last_search.to_dict()

    def start(self, game_state: GameState):
        with self._lock:
            key = self._session_key(game_state)
            # Battlesnake may retry /start after play has already begun.  Keep
            # the active recurrent/search transaction intact; after end() has
            # removed it, the same calls naturally create a fresh generation.
            self._sessions.setdefault(key, _Session())
            self._mcts_sessions.setdefault(key, _MCTSSession())
            self._session_locks.setdefault(key, threading.RLock())

    def end(self, game_state: GameState):
        key = self._session_key(game_state)
        with self._lock:
            session_lock = self._session_locks.setdefault(key, threading.RLock())
        with session_lock:
            with self._lock:
                self._sessions.pop(key, None)
                self._mcts_sessions.pop(key, None)
                self._session_locks.pop(key, None)

    def _turn_rng(self, game_state: GameState) -> np.random.Generator:
        material = (
            f"{self.seed}\0{game_state.game.id}\0{game_state.you.id}"
            f"\0{game_state.turn}"
        ).encode("utf-8", errors="surrogatepass")
        digest = hashlib.blake2b(material, digest_size=16).digest()
        return np.random.Generator(
            np.random.PCG64(int.from_bytes(digest, byteorder="little", signed=False))
        )

    def _effective_budget_ms(self, game_state: GameState) -> float:
        timeout = float(game_state.game.timeout or 0)
        if timeout <= 0:
            return self.time_budget_ms
        return min(
            self.time_budget_ms,
            max(0.0, timeout - self.timeout_margin_ms),
        )

    @staticmethod
    def _sample_action(
        rng: np.random.Generator,
        actions: Sequence[int],
        probabilities: np.ndarray,
    ) -> int:
        if len(actions) == 1:
            return int(actions[0])
        probs = _masked_probabilities(probabilities, actions, logits=False)
        allowed_probs = probs[list(actions)]
        return int(rng.choice(np.asarray(actions, dtype=np.int64), p=allowed_probs))

    @staticmethod
    def _advance(state: WorldState, point: tuple[int, int], action: int):
        dx, dy = SIM_ACTION_DELTAS[action]
        x, y = point[0] + dx, point[1] + dy
        if state.wrapped:
            x %= state.width
            y %= state.height
        return x, y

    def _opponent_action(
        self,
        state: WorldState,
        player: int,
        actions: tuple[int, ...],
        priors: np.ndarray,
        our_target: tuple[int, int] | None,
        adversarial: bool,
        rng: np.random.Generator,
    ) -> int:
        if len(actions) <= 1:
            return int(actions[0]) if actions else 0
        if not adversarial or our_target is None:
            return self._sample_action(rng, actions, priors)

        snake = state.snakes[player]
        if snake.head is None:
            return self._sample_action(rng, actions, priors)
        us = state.snakes[0]
        best_action = actions[0]
        best_score = -math.inf
        for action in actions:
            target = self._advance(state, snake.head, action)
            prior = max(1e-12, float(priors[action]))
            score = 0.30 * math.log(prior)
            distance = abs(target[0] - our_target[0]) + abs(target[1] - our_target[1])
            score -= 0.20 * distance
            if target == our_target:
                # Unknown length must be treated as capable of winning the
                # contest; exact shorter snakes do not suicidally attack us.
                dangerous = not snake.length_known or snake.length >= us.length
                score += 12.0 if dangerous else -6.0
            if score > best_score or (score == best_score and action < best_action):
                best_score = score
                best_action = action
        return int(best_action)

    def _joint_actions(
        self,
        state: WorldState,
        our_action: int,
        adversarial: bool,
        rng: np.random.Generator,
        prior_cache: dict[tuple[WorldState, int], np.ndarray],
    ) -> list[int]:
        joint = [0] * len(state.snakes)
        joint[0] = int(our_action)
        us = state.snakes[0]
        our_target = (
            None
            if us.head is None
            else self._advance(state, us.head, int(our_action))
        )
        key_state = state_key(state)
        for player in range(1, len(state.snakes)):
            actions = legal_actions(state, player)
            if not actions:
                continue
            cache_key = key_state, player
            priors = prior_cache.get(cache_key)
            if priors is None:
                priors = heuristic_action_priors(state, player)
                prior_cache[cache_key] = priors
            joint[player] = self._opponent_action(
                state,
                player,
                actions,
                priors,
                our_target,
                adversarial,
                rng,
            )
        return joint

    def _select_edge(
        self,
        node: _Node,
        actions: Sequence[int],
        *,
        visit_all: bool = False,
    ) -> _Edge:
        if visit_all:
            unvisited = [
                node.edges[int(action)]
                for action in actions
                if node.edges[int(action)].visits == 0
            ]
            if unvisited:
                # Root coverage prevents a very sharp PPO prior from turning
                # MCTS into policy-only inference before it tested alternatives.
                return max(unvisited, key=lambda edge: (edge.prior, -edge.action))
        exploration_scale = math.sqrt(max(1, node.visits))
        return max(
            (node.edges[int(action)] for action in actions),
            key=lambda edge: (
                edge.q
                + self.c_puct
                * edge.prior
                * exploration_scale
                / (1 + edge.visits),
                edge.prior,
                -edge.action,
            ),
        )

    def _leaf_value(
        self,
        state: WorldState,
        remaining: int,
        adversarial: bool,
        rng: np.random.Generator,
        deadline: float,
        root_alive_count: int,
        prior_cache: dict[tuple[WorldState, int], np.ndarray],
    ) -> tuple[float, int]:
        """Short heuristic rollout followed by one geometric evaluation."""

        depth = 0
        limit = min(remaining, self.rollout_steps)
        while depth < limit and not is_terminal(state, 0):
            if time.perf_counter() >= deadline:
                break
            actions = legal_actions(state, 0)
            if not actions:
                break
            cache_key = state_key(state), 0
            priors = prior_cache.get(cache_key)
            if priors is None:
                priors = heuristic_action_priors(state, 0)
                prior_cache[cache_key] = priors
            our_action = self._sample_action(rng, actions, priors)
            joint = self._joint_actions(
                state, our_action, adversarial, rng, prior_cache
            )
            state = step_world(state, joint, rng=rng, spawn_food=True)
            depth += 1
        return evaluate_world(state, 0, root_alive_count=root_alive_count), depth

    def _build_particles(
        self,
        game_state: GameState,
        rng: np.random.Generator,
        deadline: float,
    ) -> list[WorldState]:
        base = WorldState.from_game_state(game_state, rng=None)
        result: list[WorldState] = []
        for _ in range(self.particles):
            if result and time.perf_counter() >= deadline:
                break
            result.append(determinize_world(base, rng))
        return result or [base]

    def _run_search(
        self,
        game_state: GameState,
        root_logits: np.ndarray,
        candidates: Sequence[int],
        fallback_action: int,
        rng: np.random.Generator,
        search_started: float,
        deadline: float,
        budget_ms: float,
        *,
        iteration_limit: int | None = None,
        minimum_iterations: int | None = None,
    ) -> tuple[int, SearchDiagnostics]:
        max_iterations = (
            self.max_iterations if iteration_limit is None else int(iteration_limit)
        )
        min_iterations = (
            self.min_iterations
            if minimum_iterations is None
            else int(minimum_iterations)
        )
        priors = _masked_probabilities(root_logits, candidates, logits=True)
        root = _Node()
        root.ensure_edges(candidates, priors)
        diagnostics = SearchDiagnostics(
            deadline=deadline,
            deadline_ms=budget_ms,
            policy=MODEL_ACTIONS[fallback_action].value,
            root_prior={
                direction.value: float(priors[action])
                for action, direction in enumerate(MODEL_ACTIONS)
            },
        )
        if (
            len(candidates) <= 1
            or max_iterations == 0
            or time.perf_counter() >= deadline
        ):
            diagnostics.elapsed_ms = (time.perf_counter() - search_started) * 1000.0
            diagnostics.deadline_hit = time.perf_counter() >= deadline
            diagnostics.selected = MODEL_ACTIONS[fallback_action].value
            diagnostics.root_visits = {
                direction.value: 0 for direction in MODEL_ACTIONS
            }
            diagnostics.root_q = {
                direction.value: 0.0 for direction in MODEL_ACTIONS
            }
            return fallback_action, diagnostics

        particles = self._build_particles(game_state, rng, deadline)
        diagnostics.particles = len(particles)
        root_alive_count = particles[0].root_alive_count
        prior_cache: dict[tuple[WorldState, int], np.ndarray] = {}
        node_count = 1
        max_depth = 0

        while diagnostics.iterations < max_iterations:
            if time.perf_counter() >= deadline:
                break
            state = particles[diagnostics.iterations % len(particles)]
            node = root
            path: list[tuple[_Node, _Edge]] = []
            depth = 0
            adversarial = bool(rng.random() < self.adversarial_fraction)
            if adversarial:
                diagnostics.adversarial_iterations += 1

            while depth < self.horizon and not is_terminal(state, 0):
                if time.perf_counter() >= deadline:
                    break
                legal = legal_actions(state, 0)
                if node is root:
                    available = tuple(action for action in candidates if action in legal)
                    current_priors = priors
                else:
                    available = legal
                    if not available:
                        break
                    current_priors = heuristic_action_priors(state, 0)
                if not available:
                    break
                node.ensure_edges(available, current_priors)
                edge = self._select_edge(
                    node, available, visit_all=node is root
                )
                joint = self._joint_actions(
                    state, edge.action, adversarial, rng, prior_cache
                )
                state = step_world(state, joint, rng=rng, spawn_food=True)
                path.append((node, edge))
                depth += 1

                if edge.child is None:
                    edge.child = _Node()
                    node_count += 1
                    node = edge.child
                    break
                node = edge.child

            remaining = max(0, self.horizon - depth)
            value, rollout_depth = self._leaf_value(
                state,
                remaining,
                adversarial,
                rng,
                deadline,
                root_alive_count,
                prior_cache,
            )
            max_depth = max(max_depth, depth + rollout_depth)
            node.visits += 1
            for parent, edge in reversed(path):
                edge.visits += 1
                edge.value_sum += value
                parent.visits += 1
            diagnostics.iterations += 1

        visited = [edge for edge in root.edges.values() if edge.visits]
        # A handful of forced root visits is exploration, not evidence.  If a
        # slow model/backend or an unusually expensive state leaves too little
        # time, preserve the PPO anytime decision instead of letting noisy
        # Monte-Carlo samples override it.  One full default particle cycle is
        # the deliberately conservative production threshold.
        if visited and diagnostics.iterations >= min_iterations:
            chosen = max(
                visited,
                key=lambda edge: (edge.visits, edge.q, edge.prior, -edge.action),
            ).action
            diagnostics.fallback = "mcts"
        else:
            chosen = fallback_action
            if visited:
                diagnostics.fallback = "insufficient-search"
        now = time.perf_counter()
        diagnostics.nodes = node_count
        diagnostics.depth = max_depth
        diagnostics.elapsed_ms = (now - search_started) * 1000.0
        diagnostics.deadline_hit = now >= deadline
        diagnostics.selected = MODEL_ACTIONS[chosen].value
        diagnostics.changed_policy = chosen != fallback_action
        diagnostics.root_visits = {
            direction.value: root.edges[action].visits if action in root.edges else 0
            for action, direction in enumerate(MODEL_ACTIONS)
        }
        diagnostics.root_q = {
            direction.value: root.edges[action].q if action in root.edges else 0.0
            for action, direction in enumerate(MODEL_ACTIONS)
        }
        return chosen, diagnostics

    def _run_wave_ensemble(
        self,
        game_state: GameState,
        root_logits: np.ndarray,
        candidates: Sequence[int],
        fallback_action: int,
        rng: np.random.Generator,
        search_started: float,
        deadline: float,
        budget_ms: float,
    ) -> tuple[int, SearchDiagnostics]:
        """Run independent bounded trees and require a strict wave majority."""

        priors = _masked_probabilities(root_logits, candidates, logits=True)
        names = tuple(direction.value for direction in MODEL_ACTIONS)
        aggregate = SearchDiagnostics(
            nodes=0,
            deadline=deadline,
            deadline_ms=budget_ms,
            policy=MODEL_ACTIONS[fallback_action].value,
            search_mode="wave-ensemble",
            root_prior={name: float(priors[action]) for action, name in enumerate(names)},
            root_visits={name: 0 for name in names},
            root_q={name: 0.0 for name in names},
        )
        root_value_sums = {name: 0.0 for name in names}
        wave_votes = {name: 0 for name in names}
        complete_durations: list[float] = []
        required_complete_waves = max(
            self.min_complete_waves,
            math.ceil(self.min_iterations / self.wave_iterations),
        )

        def finish(
            chosen: int,
            *,
            fallback: str,
            reason: str,
            now: float | None = None,
        ) -> tuple[int, SearchDiagnostics]:
            finished = time.perf_counter() if now is None else now
            aggregate.elapsed_ms = (finished - search_started) * 1000.0
            aggregate.deadline_hit = aggregate.deadline_hit or finished >= deadline
            aggregate.fallback = fallback
            aggregate.decision_reason = reason
            aggregate.selected = MODEL_ACTIONS[chosen].value
            aggregate.changed_policy = chosen != fallback_action
            aggregate.waves = aggregate.waves_started
            aggregate.complete_waves = aggregate.waves_completed
            aggregate.wave_votes = {
                name: count for name, count in wave_votes.items() if count
            }
            aggregate.votes = dict(aggregate.wave_votes)
            for name in names:
                visits = aggregate.root_visits[name]
                aggregate.root_q[name] = (
                    root_value_sums[name] / visits if visits else 0.0
                )
            return chosen, aggregate

        if (
            len(candidates) <= 1
            or self.max_iterations == 0
            or time.perf_counter() >= deadline
        ):
            return finish(
                fallback_action,
                fallback="ppo",
                reason="no-wave-search",
            )

        while aggregate.iterations < self.max_iterations:
            now = time.perf_counter()
            if now >= deadline:
                break
            if complete_durations:
                required = (
                    WAVE_GUARD_FACTOR * max(complete_durations)
                    + WAVE_GUARD_BUFFER_SECONDS
                )
                if deadline - now < required:
                    aggregate.stopped_for_wave_guard = True
                    break

            remaining = self.max_iterations - aggregate.iterations
            planned_iterations = min(self.wave_iterations, remaining)
            # Exactly one draw from the turn-level master RNG isolates each
            # tree from the variable random consumption of every other tree.
            wave_seed = int(rng.bit_generator.random_raw())
            wave_rng = np.random.Generator(np.random.PCG64(wave_seed))
            wave_started = time.perf_counter()
            aggregate.waves_started += 1
            try:
                wave_action, wave = self._run_search(
                    game_state,
                    root_logits,
                    candidates,
                    fallback_action,
                    wave_rng,
                    wave_started,
                    deadline,
                    budget_ms,
                    iteration_limit=planned_iterations,
                    # The aggregate wave count enforces the global minimum
                    # evidence below.  Each complete tree must still be able
                    # to report its own winner when that minimum spans waves.
                    minimum_iterations=min(self.min_iterations, planned_iterations),
                )
            except Exception as exc:
                aggregate.waves_discarded += 1
                aggregate.error = f"{type(exc).__name__}: {exc}"
                return finish(
                    fallback_action,
                    fallback="ppo",
                    reason="wave-error",
                )

            wave_finished = time.perf_counter()
            aggregate.iterations += wave.iterations
            aggregate.nodes += wave.nodes
            aggregate.depth = max(aggregate.depth, wave.depth)
            aggregate.particles += wave.particles
            aggregate.adversarial_iterations += wave.adversarial_iterations
            aggregate.deadline_hit = aggregate.deadline_hit or wave.deadline_hit
            if wave.error is not None:
                aggregate.waves_discarded += 1
                aggregate.discarded_iterations += wave.iterations
                aggregate.error = wave.error
                return finish(
                    fallback_action,
                    fallback="ppo",
                    reason="wave-error",
                    now=wave_finished,
                )

            complete = (
                planned_iterations == self.wave_iterations
                and wave.iterations == self.wave_iterations
                and wave.particles == self.particles
                and wave_finished < deadline
                and not wave.deadline_hit
                and wave.error is None
            )
            if complete:
                aggregate.waves_completed += 1
                aggregate.voting_iterations += wave.iterations
                complete_durations.append(wave_finished - wave_started)
                vote_name = MODEL_ACTIONS[wave_action].value
                wave_votes[vote_name] += 1
                # Root evidence is meaningful only for a complete, voting
                # tree.  Q is aggregated by its corresponding visit count.
                for name in names:
                    visits = int(wave.root_visits.get(name, 0))
                    aggregate.root_visits[name] += visits
                    root_value_sums[name] += float(wave.root_q.get(name, 0.0)) * visits
            else:
                aggregate.waves_discarded += 1
                aggregate.discarded_iterations += wave.iterations
                # The final partial wave is diagnostic work only.  Starting
                # another after an incomplete tree would add deadline bias.
                break

            if wave.iterations < planned_iterations:
                break

        completed = aggregate.waves_completed
        majority_action: int | None = None
        if completed >= required_complete_waves:
            for action, name in enumerate(names):
                if wave_votes[name] * 2 > completed:
                    majority_action = action
                    break

        if majority_action is not None:
            reason = (
                "policy-wave-majority"
                if majority_action == fallback_action
                else "alternative-wave-majority"
            )
            return finish(majority_action, fallback="mcts", reason=reason)
        if completed < required_complete_waves:
            reason = "insufficient-complete-waves"
        else:
            reason = "no-strict-wave-majority"
        return finish(
            fallback_action,
            fallback="insufficient-search",
            reason=reason,
        )

    def move(self, game_state: GameState) -> MoveAction:
        request_started = time.perf_counter()
        key = self._session_key(game_state)
        with self._lock:
            session_lock = self._session_locks.setdefault(key, threading.RLock())
            committed = self._sessions.setdefault(key, _Session())
            mcts_session = self._mcts_sessions.setdefault(key, _MCTSSession())
        with session_lock:
            with self._lock:
                if (
                    self._session_locks.get(key) is not session_lock
                    or self._sessions.get(key) is not committed
                    or self._mcts_sessions.get(key) is not mcts_session
                ):
                    raise RuntimeError(
                        f"session ended or restarted before move for {key}"
                    )
            # A network retry is a read of the committed transaction: neither
            # its LSTM state nor its RNG/search result advances twice.
            if game_state.turn == committed.last_turn and committed.last_move is not None:
                return committed.last_move
            if game_state.turn < committed.last_turn:
                raise ValueError(
                    f"out-of-order turn for {key}: "
                    f"{game_state.turn} < {committed.last_turn}"
                )

            working = _copy_session(committed)
            obs = encode_observation(game_state)
            try:
                with self._model_lock:
                    logits = self._policy_logits(obs, working)
            except Exception:
                # Even a model/backend failure must obey the certain-death
                # invariant.  Keeping the old recurrent tensors is the only
                # valid transactional outcome when inference did not finish.
                working = _copy_session(committed)
                logits = np.zeros(4, dtype=np.float32)

            hard, soft = self._action_tiers(game_state)
            policy_tier = soft if soft else hard
            if not policy_tier:
                policy_tier = list(range(4))
            policy_action = self._argmax_legal(logits, policy_tier)
            # PPO4's safety result is the root contract.  A merely contested
            # hard move cannot displace an available soft move.  The complete
            # trap search may still return a hard-only escape after proving the
            # PPO line and every soft alternative trapped; with no soft move,
            # hard moves are already the unavoidable search domain.
            candidates, _ = self._safe_candidates(
                game_state, hard, soft, policy_action
            )
            if not candidates:
                candidates = list(policy_tier)
            policy_action = self._argmax_legal(logits, candidates)

            budget_ms = self._effective_budget_ms(game_state)
            deadline = request_started + budget_ms / 1000.0
            rng = self._turn_rng(game_state)
            try:
                # Preserve the evaluated fixed-32 path exactly: no master seed
                # draw and no wrapper logic for bounded model screens.  The
                # production-sized cap uses independent root trees, even
                # though its selected wave size is now smaller than 32.
                search = (
                    self._run_search
                    if self.max_iterations <= LEGACY_SINGLE_TREE_MAX_ITERATIONS
                    else self._run_wave_ensemble
                )
                action, diagnostics = search(
                    game_state,
                    logits,
                    candidates,
                    policy_action,
                    rng,
                    request_started,
                    deadline,
                    budget_ms,
                )
            except Exception as exc:
                action = policy_action
                diagnostics = SearchDiagnostics(
                    elapsed_ms=(time.perf_counter() - request_started) * 1000.0,
                    deadline=deadline,
                    deadline_ms=budget_ms,
                    deadline_hit=time.perf_counter() >= deadline,
                    policy=MODEL_ACTIONS[policy_action].value,
                    selected=MODEL_ACTIONS[action].value,
                    error=f"{type(exc).__name__}: {exc}",
                )

            # Last, independent hard invariant.  No simulator bug, hidden-state
            # sample, exception, or ranking error can emit an observably certain
            # wall/body/starvation/hazard death rejected by PPO4.
            if hard and action not in hard:
                action = self._argmax_legal(logits, hard)
                diagnostics.fallback = "hard-safety"
                diagnostics.selected = MODEL_ACTIONS[action].value
            diagnostics.changed_policy = action != policy_action

            result = MoveAction(move=MODEL_ACTIONS[action])
            working.last_turn = game_state.turn
            working.last_move = result
            with self._lock:
                if (
                    self._session_locks.get(key) is not session_lock
                    or self._sessions.get(key) is not committed
                    or self._mcts_sessions.get(key) is not mcts_session
                ):
                    raise RuntimeError(
                        f"session ended or restarted during move for {key}"
                    )
                # Keep the session object as the generation token.  This is a
                # transactional mutation under both locks: concurrent retries
                # observe either the old committed fields or all new fields.
                committed.lstm_states = working.lstm_states
                committed.episode_start = working.episode_start
                committed.last_turn = working.last_turn
                committed.last_move = working.last_move
                mcts_session.last_turn = game_state.turn
                mcts_session.last_search = diagnostics
                self._last_search = diagnostics
            return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the PPO-guided IS-MCTS agent")
    parser.add_argument("port", type=int, help="HTTP server port")
    parser.add_argument(
        "--model",
        default=None,
        help="PPO .zip model (default: PPO_MCTS_MODEL_PATH/PPO_MODEL_PATH or V24)",
    )
    parser.add_argument("--time-budget-ms", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--min-iterations", type=int, default=None)
    parser.add_argument("--wave-iterations", type=int, default=None)
    parser.add_argument("--min-complete-waves", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    from battlesnake_server import start_server

    args = _build_parser().parse_args(argv)
    kwargs: dict[str, object] = {
        "time_budget_ms": args.time_budget_ms,
        "max_iterations": args.max_iterations,
        "min_iterations": args.min_iterations,
        "wave_iterations": args.wave_iterations,
        "min_complete_waves": args.min_complete_waves,
        "horizon": args.horizon,
        "particles": args.particles,
        "rollout_steps": args.rollout_steps,
        "seed": args.seed,
    }
    # ``None`` is meaningful to PPOAgent4 (random weights), so omit the model
    # argument entirely when the CLI should use its champion default.
    if args.model is None:
        agent = PPOMCTSAgent(**kwargs)
    else:
        agent = PPOMCTSAgent(args.model, **kwargs)
    start_server(agent=agent, port=args.port)


if __name__ == "__main__":
    main(sys.argv[1:])
