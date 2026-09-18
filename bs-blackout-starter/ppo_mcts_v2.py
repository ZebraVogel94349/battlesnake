"""Persistent, history-conditioned PPO-guided MCTS for Battlesnake Blackout.

This is an opt-in successor to ``ppo_mcts.py``.  It deliberately lives in a
new module and has a distinct class/name so importing or editing it cannot
change the agent currently serving the tournament.

Differences from the original search:

* one root-sampled tree consumes the entire deadline; evidence is never reduced
  to votes from independent 16-simulation trees;
* root particles are generated lazily by a per-game belief tracker and useful
  particles/tree descendants survive into the next real turn;
* the recurrent PPO actor supplies priors when an inner node is first expanded;
* rootless simulations are rejected and repeated zero-edge samples terminate
  early instead of being counted as completed MCTS work;
* every processed turn can emit a self-contained JSONL diagnostic record;
* all previously returned turns remain cached, making late HTTP retries reads
  instead of out-of-order exceptions.

The deployable checkpoint has no trustworthy privileged critic, so terminal
and geometric values still come from ``mcts_simulator.evaluate_world``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Iterable, Sequence

import numpy as np

from battlesnake_types import GameState, MoveAction
from mcts_belief_v2 import BeliefTrackerV2
from mcts_simulator import (
    ACTION_DELTAS as SIM_ACTION_DELTAS,
    WorldState,
    encode_world_observation,
    evaluate_world,
    heuristic_action_priors,
    is_terminal,
    legal_actions,
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
DEFAULT_HORIZON = 12
DEFAULT_ROLLOUT_STEPS = 2
DEFAULT_C_PUCT = 1.25
DEFAULT_RISK_PENALTY = 0.12
DEFAULT_CVAR_ALPHA = 0.20
DEFAULT_ADVERSARIAL_FRACTION = 0.25
DEFAULT_TIMEOUT_MARGIN_MS = 80.0
DEFAULT_NETWORK_GUARD_MS = 14.0
DEFAULT_MAX_ROOT_REJECTIONS = 8
DEFAULT_REUSE_DECAY = 0.45
DEFAULT_MAX_RESERVOIR = 96
DEFAULT_BELIEF_REUSE_FRACTION = 0.55
DEFAULT_SEED = 0
DEFAULT_ROOT_DECISION = "visits"
ROOT_DECISIONS = ("visits", "risk", "cvar")


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


def _copy_session(session: _Session | None) -> _Session | None:
    if session is None:
        return None
    states = session.lstm_states
    if states is not None:
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


def _fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            return "unavailable"
        component = hashlib.sha256(path.read_bytes()).digest()
        digest.update(path.name.encode("utf-8", errors="surrogatepass"))
        digest.update(component)
    return digest.hexdigest()[:12]


@dataclass(slots=True)
class _Edge:
    action: int
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    value_square_sum: float = 0.0
    minimum_value: float = 1.0
    samples: list[float] = field(default_factory=list)
    child: _Node | None = None

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    @property
    def std(self) -> float:
        if self.visits <= 1:
            return 0.0
        variance = self.value_square_sum / self.visits - self.q * self.q
        return math.sqrt(max(0.0, variance))

    def risk_q(self, penalty: float) -> float:
        return self.q - penalty * self.std

    def cvar(self, alpha: float) -> float:
        if not self.samples:
            return self.q
        count = max(1, math.ceil(len(self.samples) * alpha))
        return float(sum(sorted(self.samples)[:count]) / count)

    def record(self, value: float, *, keep_sample: bool) -> None:
        self.visits += 1
        self.value_sum += value
        self.value_square_sum += value * value
        self.minimum_value = min(self.minimum_value, value)
        if keep_sample:
            self.samples.append(value)


@dataclass(slots=True)
class _Node:
    visits: int = 0
    edges: dict[int, _Edge] = field(default_factory=dict)
    policy_priors: np.ndarray | None = None
    actor_session: _Session | None = None

    def ensure_edges(
        self,
        actions: Iterable[int],
        priors: np.ndarray,
        *,
        refresh: bool = False,
    ) -> None:
        for raw_action in actions:
            action = int(raw_action)
            if action not in self.edges:
                self.edges[action] = _Edge(action, float(priors[action]))
            elif refresh:
                self.edges[action].prior = float(priors[action])


@dataclass(slots=True)
class SearchDiagnosticsV2:
    iterations: int = 0
    rejected_particles: int = 0
    consecutive_root_rejections: int = 0
    new_nodes: int = 0
    tree_nodes: int = 1
    depth: int = 0
    elapsed_ms: float = 0.0
    deadline_ms: float = 0.0
    deadline_hit: bool = False
    actor_evaluations: int = 0
    actor_fallbacks: int = 0
    adversarial_iterations: int = 0
    tree_reused: bool = False
    reused_root_visits: int = 0
    fallback: str = "ppo"
    decision_reason: str = ""
    policy: str = ""
    selected: str = ""
    changed_policy: bool = False
    root_prior: dict[str, float] = field(default_factory=dict)
    root_visits: dict[str, int] = field(default_factory=dict)
    root_q: dict[str, float] = field(default_factory=dict)
    root_risk_q: dict[str, float] = field(default_factory=dict)
    root_cvar: dict[str, float] = field(default_factory=dict)
    root_min: dict[str, float] = field(default_factory=dict)
    belief: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class _SearchSessionV2:
    belief: BeliefTrackerV2
    root: _Node | None = None
    selected_tree_action: int | None = None
    last_turn: int = -1
    move_cache: dict[int, MoveAction] = field(default_factory=dict)
    diagnostic_cache: dict[int, SearchDiagnosticsV2] = field(default_factory=dict)


class PPOMCTSAgentV2(PPOAgent4):
    """PPO4 plus persistent root-sampled information-set MCTS."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None | object = _DEFAULT_MODEL,
        *,
        time_budget_ms: float | None = None,
        max_iterations: int | None = None,
        min_iterations: int | None = None,
        horizon: int | None = None,
        rollout_steps: int | None = None,
        c_puct: float | None = None,
        risk_penalty: float | None = None,
        cvar_alpha: float | None = None,
        adversarial_fraction: float | None = None,
        timeout_margin_ms: float | None = None,
        network_guard_ms: float | None = None,
        max_root_rejections: int | None = None,
        reuse_decay: float | None = None,
        max_reservoir: int | None = None,
        belief_reuse_fraction: float | None = None,
        root_decision: str | None = None,
        seed: int | None = None,
        diagnostics_path: str | os.PathLike[str] | None = None,
        symmetries: int | None = None,
        safety_search: bool | None = None,
        device: str | None = None,
    ):
        if time_budget_ms is None:
            time_budget_ms = _env_float(("PPO_MCTS_V2_TIME_BUDGET_MS",), DEFAULT_TIME_BUDGET_MS)
        if max_iterations is None:
            max_iterations = _env_int(("PPO_MCTS_V2_MAX_ITERATIONS",), DEFAULT_MAX_ITERATIONS)
        if min_iterations is None:
            min_iterations = _env_int(("PPO_MCTS_V2_MIN_ITERATIONS",), DEFAULT_MIN_ITERATIONS)
        if horizon is None:
            horizon = _env_int(("PPO_MCTS_V2_HORIZON",), DEFAULT_HORIZON)
        if rollout_steps is None:
            rollout_steps = _env_int(("PPO_MCTS_V2_ROLLOUT_STEPS",), DEFAULT_ROLLOUT_STEPS)
        if c_puct is None:
            c_puct = _env_float(("PPO_MCTS_V2_C_PUCT",), DEFAULT_C_PUCT)
        if risk_penalty is None:
            risk_penalty = _env_float(("PPO_MCTS_V2_RISK_PENALTY",), DEFAULT_RISK_PENALTY)
        if cvar_alpha is None:
            cvar_alpha = _env_float(("PPO_MCTS_V2_CVAR_ALPHA",), DEFAULT_CVAR_ALPHA)
        if adversarial_fraction is None:
            adversarial_fraction = _env_float(
                ("PPO_MCTS_V2_ADVERSARIAL_FRACTION",), DEFAULT_ADVERSARIAL_FRACTION
            )
        if timeout_margin_ms is None:
            timeout_margin_ms = _env_float(
                ("PPO_MCTS_V2_TIMEOUT_MARGIN_MS",), DEFAULT_TIMEOUT_MARGIN_MS
            )
        if network_guard_ms is None:
            network_guard_ms = _env_float(
                ("PPO_MCTS_V2_NETWORK_GUARD_MS",), DEFAULT_NETWORK_GUARD_MS
            )
        if max_root_rejections is None:
            max_root_rejections = _env_int(
                ("PPO_MCTS_V2_MAX_ROOT_REJECTIONS",), DEFAULT_MAX_ROOT_REJECTIONS
            )
        if reuse_decay is None:
            reuse_decay = _env_float(("PPO_MCTS_V2_REUSE_DECAY",), DEFAULT_REUSE_DECAY)
        if max_reservoir is None:
            max_reservoir = _env_int(("PPO_MCTS_V2_MAX_RESERVOIR",), DEFAULT_MAX_RESERVOIR)
        if belief_reuse_fraction is None:
            belief_reuse_fraction = _env_float(
                ("PPO_MCTS_V2_BELIEF_REUSE_FRACTION",),
                DEFAULT_BELIEF_REUSE_FRACTION,
            )
        if root_decision is None:
            root_decision = os.environ.get("PPO_MCTS_V2_ROOT_DECISION", DEFAULT_ROOT_DECISION)
        if seed is None:
            seed = _env_int(("PPO_MCTS_V2_SEED",), DEFAULT_SEED)
        if diagnostics_path is None:
            diagnostics_path = os.environ.get("PPO_MCTS_V2_DIAGNOSTICS_PATH")
        if safety_search is None:
            safety_search = True

        if not math.isfinite(time_budget_ms) or time_budget_ms < 0:
            raise ValueError("time_budget_ms must be finite and non-negative")
        if max_iterations < 0 or min_iterations < 0:
            raise ValueError("iteration counts must be non-negative")
        if max_iterations and min_iterations > max_iterations:
            raise ValueError("min_iterations must not exceed max_iterations")
        if horizon < 1 or rollout_steps < 0:
            raise ValueError("horizon must be positive and rollout_steps non-negative")
        if not math.isfinite(c_puct) or c_puct < 0:
            raise ValueError("c_puct must be finite and non-negative")
        if not math.isfinite(risk_penalty) or risk_penalty < 0:
            raise ValueError("risk_penalty must be finite and non-negative")
        if not math.isfinite(cvar_alpha) or not 0 < cvar_alpha <= 1:
            raise ValueError("cvar_alpha must be in (0, 1]")
        if not 0 <= adversarial_fraction <= 1:
            raise ValueError("adversarial_fraction must be in [0, 1]")
        if timeout_margin_ms < 0 or network_guard_ms < 0:
            raise ValueError("deadline margins must be non-negative")
        if max_root_rejections < 1 or max_reservoir < 1:
            raise ValueError("root rejection and reservoir limits must be positive")
        if not 0 <= reuse_decay <= 1 or not 0 <= belief_reuse_fraction <= 1:
            raise ValueError("reuse factors must be in [0, 1]")
        if root_decision not in ROOT_DECISIONS:
            raise ValueError(f"root_decision must be one of {ROOT_DECISIONS}")

        super().__init__(
            model_path,
            symmetries=symmetries,
            safety_search=safety_search,
            device=device,
        )
        if tuple(ACTION_DELTAS) != tuple(SIM_ACTION_DELTAS):
            raise RuntimeError("PPO4 and simulator action orders differ")

        self.time_budget_ms = float(time_budget_ms)
        self.max_iterations = int(max_iterations)
        self.min_iterations = int(min_iterations)
        self.horizon = int(horizon)
        self.rollout_steps = min(int(rollout_steps), self.horizon)
        self.c_puct = float(c_puct)
        self.risk_penalty = float(risk_penalty)
        self.cvar_alpha = float(cvar_alpha)
        self.adversarial_fraction = float(adversarial_fraction)
        self.timeout_margin_ms = float(timeout_margin_ms)
        self.network_guard_ms = float(network_guard_ms)
        self.max_root_rejections = int(max_root_rejections)
        self.reuse_decay = float(reuse_decay)
        self.max_reservoir = int(max_reservoir)
        self.belief_reuse_fraction = float(belief_reuse_fraction)
        self.root_decision = root_decision
        self.seed = int(seed)
        self.diagnostics_path = None if diagnostics_path is None else Path(diagnostics_path)
        self._search_sessions: dict[tuple[str, str], _SearchSessionV2] = {}
        self._session_locks: dict[tuple[str, str], threading.RLock] = {}
        self._model_lock = threading.RLock()
        self._diagnostic_lock = threading.RLock()
        self._last_search: SearchDiagnosticsV2 | None = None
        source_dir = Path(__file__).resolve().parent
        self._code_fingerprint = _fingerprint(
            (
                source_dir / "ppo_mcts_v2.py",
                source_dir / "mcts_belief_v2.py",
                source_dir / "mcts_simulator.py",
                source_dir / "ppo4.py",
            )
        )
        # Warm only this new agent instance.  Merely creating the source file has
        # no effect on the separately running tournament process.
        with self._model_lock:
            self._policy_logits(np.zeros((9, 29, 29), dtype=np.float32), _Session())

    def get_name(self):
        return "Der Snaketürke PPO-MCTS v2 (experimental)"

    def get_diagnostics(self) -> dict[str, object]:
        result = super().get_diagnostics()
        result.update(
            {
                "time_budget_ms": self.time_budget_ms,
                "max_iterations": self.max_iterations,
                "min_iterations": self.min_iterations,
                "horizon": self.horizon,
                "rollout_steps": self.rollout_steps,
                "c_puct": self.c_puct,
                "risk_penalty": self.risk_penalty,
                "cvar_alpha": self.cvar_alpha,
                "adversarial_fraction": self.adversarial_fraction,
                "root_decision": self.root_decision,
                "reuse_decay": self.reuse_decay,
                "max_reservoir": self.max_reservoir,
                "belief_reuse_fraction": self.belief_reuse_fraction,
                "diagnostics_path": (
                    None if self.diagnostics_path is None else str(self.diagnostics_path)
                ),
                "last_search": (
                    None if self._last_search is None else self._last_search.to_dict()
                ),
            }
        )
        return result

    def get_last_search_stats(self) -> dict[str, object] | None:
        with self._lock:
            return None if self._last_search is None else self._last_search.to_dict()

    def _new_search_session(self) -> _SearchSessionV2:
        return _SearchSessionV2(
            belief=BeliefTrackerV2(
                max_reservoir=self.max_reservoir,
                reuse_fraction=self.belief_reuse_fraction,
            )
        )

    def start(self, game_state: GameState):
        key = self._session_key(game_state)
        with self._lock:
            self._sessions.setdefault(key, _Session())
            self._search_sessions.setdefault(key, self._new_search_session())
            self._session_locks.setdefault(key, threading.RLock())

    def end(self, game_state: GameState):
        key = self._session_key(game_state)
        with self._lock:
            session_lock = self._session_locks.setdefault(key, threading.RLock())
        with session_lock:
            with self._lock:
                self._sessions.pop(key, None)
                self._search_sessions.pop(key, None)
                self._session_locks.pop(key, None)

    def _turn_rng(self, game_state: GameState) -> np.random.Generator:
        material = (
            f"v2\0{self.seed}\0{game_state.game.id}\0{game_state.you.id}"
            f"\0{game_state.turn}"
        ).encode("utf-8", errors="surrogatepass")
        digest = hashlib.blake2b(material, digest_size=16).digest()
        return np.random.Generator(
            np.random.PCG64(int.from_bytes(digest, "little", signed=False))
        )

    def _effective_budget_ms(self, game_state: GameState) -> float:
        timeout = float(game_state.game.timeout or 0)
        if timeout <= 0:
            return self.time_budget_ms
        return min(self.time_budget_ms, max(0.0, timeout - self.timeout_margin_ms))

    @staticmethod
    def _sample_action(
        rng: np.random.Generator,
        actions: Sequence[int],
        probabilities: np.ndarray,
    ) -> int:
        if len(actions) == 1:
            return int(actions[0])
        distribution = _masked_probabilities(probabilities, actions, logits=False)
        return int(rng.choice(np.asarray(actions), p=distribution[list(actions)]))

    @staticmethod
    def _advance(state: WorldState, point: tuple[int, int], action: int) -> tuple[int, int]:
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
            distance = abs(target[0] - our_target[0]) + abs(target[1] - our_target[1])
            exits = sum(
                0 <= target[0] + dx < state.width
                and 0 <= target[1] + dy < state.height
                for dx, dy in SIM_ACTION_DELTAS
            )
            score = 0.22 * math.log(prior) - 0.18 * distance + 0.08 * exits
            if target == our_target:
                dangerous = snake.length >= us.length
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
        our_target = None if us.head is None else self._advance(state, us.head, our_action)
        for player in range(1, len(state.snakes)):
            actions = legal_actions(state, player)
            if not actions:
                continue
            key = state, player
            priors = prior_cache.get(key)
            if priors is None:
                priors = heuristic_action_priors(state, player)
                prior_cache[key] = priors
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
        force_coverage: bool,
    ) -> _Edge:
        if force_coverage:
            unvisited = [node.edges[int(action)] for action in actions if not node.edges[int(action)].visits]
            if unvisited:
                return max(unvisited, key=lambda edge: (edge.prior, -edge.action))
        scale = math.sqrt(max(1, node.visits))
        return max(
            (node.edges[int(action)] for action in actions),
            key=lambda edge: (
                edge.risk_q(self.risk_penalty)
                + self.c_puct * edge.prior * scale / (1 + edge.visits),
                edge.prior,
                -edge.action,
            ),
        )

    def _initialize_node_policy(
        self,
        node: _Node,
        parent_session: _Session | None,
        state: WorldState,
        deadline: float,
        diagnostics: SearchDiagnosticsV2,
    ) -> None:
        actions = legal_actions(state, 0)
        heuristic = heuristic_action_priors(state, 0)
        node.policy_priors = heuristic.astype(np.float64)
        node.actor_session = None
        if not actions or parent_session is None:
            diagnostics.actor_fallbacks += 1
            node.ensure_edges(actions, node.policy_priors, refresh=True)
            return
        remaining_ms = (deadline - time.perf_counter()) * 1000.0
        if remaining_ms <= self.network_guard_ms:
            diagnostics.actor_fallbacks += 1
            node.ensure_edges(actions, node.policy_priors, refresh=True)
            return
        child_session = _copy_session(parent_session)
        try:
            observation = encode_world_observation(state, 0, view_radius=None)
            with self._model_lock:
                logits = self._policy_logits(observation, child_session)
            node.policy_priors = _masked_probabilities(logits, actions, logits=True)
            node.actor_session = child_session
            diagnostics.actor_evaluations += 1
        except Exception:
            diagnostics.actor_fallbacks += 1
        node.ensure_edges(actions, node.policy_priors, refresh=True)

    def _leaf_value(
        self,
        state: WorldState,
        node: _Node,
        remaining: int,
        adversarial: bool,
        rng: np.random.Generator,
        deadline: float,
        root_alive_count: int,
        prior_cache: dict[tuple[WorldState, int], np.ndarray],
    ) -> tuple[float, int]:
        depth = 0
        limit = min(remaining, self.rollout_steps)
        first_priors = node.policy_priors
        while depth < limit and not is_terminal(state, 0):
            if time.perf_counter() >= deadline:
                break
            actions = legal_actions(state, 0)
            if not actions:
                break
            if depth == 0 and first_priors is not None:
                priors = first_priors
            else:
                key = state, 0
                priors = prior_cache.get(key)
                if priors is None:
                    priors = heuristic_action_priors(state, 0)
                    prior_cache[key] = priors
            action = self._sample_action(rng, actions, priors)
            joint = self._joint_actions(state, action, adversarial, rng, prior_cache)
            state = step_world(state, joint, rng=rng, spawn_food=True)
            depth += 1
        return evaluate_world(state, 0, root_alive_count=root_alive_count), depth

    @staticmethod
    def _tree_size(root: _Node) -> int:
        seen: set[int] = set()
        todo = [root]
        while todo:
            node = todo.pop()
            identity = id(node)
            if identity in seen:
                continue
            seen.add(identity)
            todo.extend(edge.child for edge in node.edges.values() if edge.child is not None)
        return len(seen)

    @staticmethod
    def _clear_descendant_sessions(root: _Node) -> None:
        todo = [edge.child for edge in root.edges.values() if edge.child is not None]
        seen: set[int] = set()
        while todo:
            node = todo.pop()
            identity = id(node)
            if identity in seen:
                continue
            seen.add(identity)
            node.actor_session = None
            todo.extend(edge.child for edge in node.edges.values() if edge.child is not None)

    @staticmethod
    def _decay_tree(root: _Node, factor: float) -> None:
        todo = [root]
        seen: set[int] = set()
        while todo:
            node = todo.pop()
            identity = id(node)
            if identity in seen:
                continue
            seen.add(identity)
            node.visits = int(round(node.visits * factor))
            for edge in node.edges.values():
                old_visits = edge.visits
                edge.visits = int(round(old_visits * factor))
                effective_factor = edge.visits / old_visits if old_visits else 0.0
                edge.value_sum *= effective_factor
                edge.value_square_sum *= effective_factor
                keep = int(round(len(edge.samples) * factor))
                edge.samples = edge.samples[-keep:] if keep else []
                if not edge.visits:
                    edge.value_sum = 0.0
                    edge.value_square_sum = 0.0
                    edge.minimum_value = 1.0
                if edge.child is not None:
                    todo.append(edge.child)

    def _prepare_tree_root(
        self,
        search_session: _SearchSessionV2,
        current_turn: int,
        root_priors: np.ndarray,
        root_actor_session: _Session | None,
        candidates: Sequence[int],
    ) -> tuple[_Node, bool]:
        root: _Node | None = None
        reused = False
        if (
            search_session.root is not None
            and search_session.selected_tree_action is not None
            and search_session.last_turn + 1 == current_turn
        ):
            previous_edge = search_session.root.edges.get(search_session.selected_tree_action)
            if previous_edge is not None and previous_edge.child is not None:
                root = previous_edge.child
                reused = True
                self._decay_tree(root, self.reuse_decay)
                self._clear_descendant_sessions(root)
        if root is None:
            root = _Node()
        root.actor_session = _copy_session(root_actor_session)
        root.policy_priors = root_priors.copy()
        root.ensure_edges(candidates, root_priors, refresh=True)
        return root, reused

    def _root_choice_key(self, edge: _Edge):
        if self.root_decision == "risk":
            return (
                edge.risk_q(self.risk_penalty),
                edge.visits,
                edge.cvar(self.cvar_alpha),
                edge.prior,
                -edge.action,
            )
        if self.root_decision == "cvar":
            robust = 0.65 * edge.q + 0.35 * edge.cvar(self.cvar_alpha)
            return (robust, edge.visits, edge.prior, -edge.action)
        return (
            edge.visits,
            edge.risk_q(self.risk_penalty),
            edge.q,
            edge.prior,
            -edge.action,
        )

    def _finish_diagnostics(
        self,
        diagnostics: SearchDiagnosticsV2,
        root: _Node,
        chosen: int,
        fallback_action: int,
        search_started: float,
        deadline: float,
        belief: BeliefTrackerV2,
    ) -> None:
        now = time.perf_counter()
        diagnostics.elapsed_ms = (now - search_started) * 1000.0
        diagnostics.deadline_hit = now >= deadline
        diagnostics.tree_nodes = self._tree_size(root)
        diagnostics.selected = MODEL_ACTIONS[chosen].value
        diagnostics.changed_policy = chosen != fallback_action
        diagnostics.belief = belief.diagnostics.to_dict()
        for action, direction in enumerate(MODEL_ACTIONS):
            edge = root.edges.get(action)
            name = direction.value
            diagnostics.root_visits[name] = 0 if edge is None else edge.visits
            diagnostics.root_q[name] = 0.0 if edge is None else edge.q
            diagnostics.root_risk_q[name] = (
                0.0 if edge is None else edge.risk_q(self.risk_penalty)
            )
            diagnostics.root_cvar[name] = (
                0.0 if edge is None else edge.cvar(self.cvar_alpha)
            )
            diagnostics.root_min[name] = (
                0.0 if edge is None or not edge.visits else edge.minimum_value
            )

    def _run_search(
        self,
        game_state: GameState,
        root_logits: np.ndarray,
        candidates: Sequence[int],
        fallback_action: int,
        root_actor_session: _Session | None,
        search_session: _SearchSessionV2,
        rng: np.random.Generator,
        search_started: float,
        deadline: float,
        budget_ms: float,
    ) -> tuple[int, _Node, list[WorldState], SearchDiagnosticsV2]:
        root_priors = _masked_probabilities(root_logits, candidates, logits=True)
        root, reused = self._prepare_tree_root(
            search_session,
            game_state.turn,
            root_priors,
            root_actor_session,
            candidates,
        )
        diagnostics = SearchDiagnosticsV2(
            deadline_ms=budget_ms,
            tree_reused=reused,
            reused_root_visits=sum(root.edges[action].visits for action in candidates),
            policy=MODEL_ACTIONS[fallback_action].value,
            root_prior={
                direction.value: float(root_priors[action])
                for action, direction in enumerate(MODEL_ACTIONS)
            },
        )
        sampled_particles: list[WorldState] = []
        if (
            len(candidates) <= 1
            or self.max_iterations == 0
            or time.perf_counter() >= deadline
        ):
            diagnostics.decision_reason = "no-search-needed"
            self._finish_diagnostics(
                diagnostics,
                root,
                fallback_action,
                fallback_action,
                search_started,
                deadline,
                search_session.belief,
            )
            return fallback_action, root, sampled_particles, diagnostics

        prior_cache: dict[tuple[WorldState, int], np.ndarray] = {}
        max_depth = 0
        root_alive_count: int | None = None
        consecutive_rejections = 0
        while diagnostics.iterations < self.max_iterations:
            if time.perf_counter() >= deadline:
                break
            state = search_session.belief.sample(game_state, rng)
            if len(sampled_particles) < self.max_reservoir:
                sampled_particles.append(state)
            if root_alive_count is None:
                root_alive_count = state.root_alive_count
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
                available = (
                    tuple(action for action in candidates if action in legal)
                    if node is root
                    else legal
                )
                if not available:
                    break
                priors = node.policy_priors
                if priors is None:
                    priors = heuristic_action_priors(state, 0).astype(np.float64)
                    node.policy_priors = priors
                node.ensure_edges(available, priors)
                edge = self._select_edge(
                    node,
                    available,
                    force_coverage=node is root,
                )
                parent_session = node.actor_session
                joint = self._joint_actions(
                    state,
                    edge.action,
                    adversarial,
                    rng,
                    prior_cache,
                )
                state = step_world(state, joint, rng=rng, spawn_food=True)
                path.append((node, edge))
                depth += 1

                if edge.child is None:
                    edge.child = _Node()
                    diagnostics.new_nodes += 1
                    self._initialize_node_policy(
                        edge.child,
                        parent_session,
                        state,
                        deadline,
                        diagnostics,
                    )
                    node = edge.child
                    break
                node = edge.child
                if node.actor_session is None and not is_terminal(state, 0):
                    self._initialize_node_policy(
                        node,
                        parent_session,
                        state,
                        deadline,
                        diagnostics,
                    )

            if not path:
                diagnostics.rejected_particles += 1
                consecutive_rejections += 1
                diagnostics.consecutive_root_rejections = max(
                    diagnostics.consecutive_root_rejections,
                    consecutive_rejections,
                )
                if consecutive_rejections >= self.max_root_rejections:
                    diagnostics.decision_reason = "no-root-edge"
                    break
                continue
            consecutive_rejections = 0
            value, rollout_depth = self._leaf_value(
                state,
                node,
                max(0, self.horizon - depth),
                adversarial,
                rng,
                deadline,
                int(root_alive_count or 1),
                prior_cache,
            )
            max_depth = max(max_depth, depth + rollout_depth)
            node.visits += 1
            for parent, edge in reversed(path):
                edge.record(value, keep_sample=parent is root)
                parent.visits += 1
            diagnostics.iterations += 1

        diagnostics.depth = max_depth
        visited = [root.edges[action] for action in candidates if root.edges[action].visits]
        if visited and diagnostics.iterations >= self.min_iterations:
            chosen = max(visited, key=self._root_choice_key).action
            diagnostics.fallback = "mcts"
            diagnostics.decision_reason = diagnostics.decision_reason or self.root_decision
        else:
            chosen = fallback_action
            diagnostics.fallback = "insufficient-search" if visited else "ppo"
            diagnostics.decision_reason = diagnostics.decision_reason or "insufficient-search"
        self._finish_diagnostics(
            diagnostics,
            root,
            chosen,
            fallback_action,
            search_started,
            deadline,
            search_session.belief,
        )
        return chosen, root, sampled_particles, diagnostics

    def _late_request_policy(self, game_state: GameState) -> MoveAction:
        """Stateless safe response for an unseen old turn; never mutates memory."""

        session = _Session()
        try:
            with self._model_lock:
                logits = self._policy_logits(encode_observation(game_state), session)
        except Exception:
            logits = np.zeros(4, dtype=np.float32)
        hard, soft = self._action_tiers(game_state)
        tier = soft if soft else hard
        if not tier:
            tier = list(range(4))
        action = self._argmax_legal(logits, tier)
        candidates, _ = self._safe_candidates(game_state, hard, soft, action)
        if candidates:
            action = self._argmax_legal(logits, candidates)
        return MoveAction(move=MODEL_ACTIONS[action])

    def _emit_diagnostic(
        self,
        game_state: GameState,
        result: MoveAction,
        diagnostics: SearchDiagnosticsV2,
    ) -> None:
        if self.diagnostics_path is None:
            return
        record = {
            "schema_version": 1,
            "agent": type(self).__name__,
            "game_id": game_state.game.id,
            "you_id": game_state.you.id,
            "turn": game_state.turn,
            "move": result.move.value,
            "recorded_at_unix": time.time(),
            "search": diagnostics.to_dict(),
        }
        rendered = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._diagnostic_lock:
            self.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            with self.diagnostics_path.open("a", encoding="utf-8") as handle:
                handle.write(rendered + "\n")

    def move(self, game_state: GameState) -> MoveAction:
        request_started = time.perf_counter()
        key = self._session_key(game_state)
        with self._lock:
            session_lock = self._session_locks.setdefault(key, threading.RLock())
        with session_lock:
            with self._lock:
                committed = self._sessions.setdefault(key, _Session())
                search_session = self._search_sessions.setdefault(
                    key, self._new_search_session()
                )
                cached = search_session.move_cache.get(game_state.turn)
            if cached is not None:
                return cached
            if game_state.turn < committed.last_turn:
                return self._late_request_policy(game_state)

            working = _copy_session(committed)
            try:
                with self._model_lock:
                    logits = self._policy_logits(encode_observation(game_state), working)
                root_actor_session = working
            except Exception:
                working = _copy_session(committed)
                logits = np.zeros(4, dtype=np.float32)
                root_actor_session = None

            hard, soft = self._action_tiers(game_state)
            tier = soft if soft else hard
            if not tier:
                tier = list(range(4))
            policy_action = self._argmax_legal(logits, tier)
            candidates, _ = self._safe_candidates(
                game_state, hard, soft, policy_action
            )
            if not candidates:
                candidates = list(tier)
            policy_action = self._argmax_legal(logits, candidates)

            budget_ms = self._effective_budget_ms(game_state)
            deadline = request_started + budget_ms / 1000.0
            rng = self._turn_rng(game_state)
            search_session.belief.prepare(game_state, rng)
            try:
                action, root, particles, diagnostics = self._run_search(
                    game_state,
                    logits,
                    candidates,
                    policy_action,
                    root_actor_session,
                    search_session,
                    rng,
                    request_started,
                    deadline,
                    budget_ms,
                )
            except Exception as exc:
                action = policy_action
                root = _Node(policy_priors=_masked_probabilities(logits, candidates, logits=True))
                root.ensure_edges(candidates, root.policy_priors)
                particles = []
                diagnostics = SearchDiagnosticsV2(
                    elapsed_ms=(time.perf_counter() - request_started) * 1000.0,
                    deadline_ms=budget_ms,
                    deadline_hit=time.perf_counter() >= deadline,
                    policy=MODEL_ACTIONS[policy_action].value,
                    selected=MODEL_ACTIONS[action].value,
                    fallback="ppo",
                    decision_reason="search-error",
                    error=f"{type(exc).__name__}: {exc}",
                )

            if hard and action not in hard:
                action = self._argmax_legal(logits, hard)
                diagnostics.fallback = "hard-safety"
                diagnostics.decision_reason = "hard-safety"
                diagnostics.selected = MODEL_ACTIONS[action].value
            diagnostics.changed_policy = action != policy_action
            result = MoveAction(move=MODEL_ACTIONS[action])

            working.last_turn = game_state.turn
            working.last_move = result
            search_session.belief.commit(particles, action, game_state.turn)
            with self._lock:
                committed.lstm_states = working.lstm_states
                committed.episode_start = working.episode_start
                committed.last_turn = working.last_turn
                committed.last_move = working.last_move
                search_session.root = root
                search_session.selected_tree_action = action
                search_session.last_turn = game_state.turn
                search_session.move_cache[game_state.turn] = result
                search_session.diagnostic_cache[game_state.turn] = diagnostics
                self._last_search = diagnostics
            self._emit_diagnostic(game_state, result, diagnostics)
            return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the experimental PPO-MCTS v2 agent")
    parser.add_argument("port", type=int)
    parser.add_argument("--model")
    parser.add_argument("--time-budget-ms", type=float)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--min-iterations", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--root-decision", choices=ROOT_DECISIONS)
    parser.add_argument("--diagnostics-path")
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    from battlesnake_server import start_server

    args = _build_parser().parse_args(argv)
    kwargs = {
        "time_budget_ms": args.time_budget_ms,
        "max_iterations": args.max_iterations,
        "min_iterations": args.min_iterations,
        "horizon": args.horizon,
        "rollout_steps": args.rollout_steps,
        "root_decision": args.root_decision,
        "diagnostics_path": args.diagnostics_path,
        "seed": args.seed,
    }
    # Drop CLI Nones so constructor defaults/environment remain effective.
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    agent = PPOMCTSAgentV2(**kwargs) if args.model is None else PPOMCTSAgentV2(args.model, **kwargs)
    start_server(agent=agent, port=args.port)


if __name__ == "__main__":
    main(sys.argv[1:])
