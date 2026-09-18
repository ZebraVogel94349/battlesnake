from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import threading
import time

import numpy as np
import pytest
import torch

from battlesnake_types import Direction, GameState
from ppo4 import PPOAgent4, _Session
from ppo_mcts import (
    DEFAULT_MCTS_MODEL_PATH,
    PPOMCTSAgent,
    SearchDiagnostics,
    _composite_fingerprint,
    _MCTSSession,
)


def _game_state(
    *,
    head=(7, 7),
    body=((7, 7), (7, 6), (7, 5)),
    health=100,
    food=(),
    opponents=(),
    turn=4,
    game_id="mcts-test",
) -> GameState:
    def snake(snake_id, snake_head, snake_body):
        return {
            "id": snake_id,
            "name": snake_id,
            "length": len(snake_body),
            "latency": "0",
            "squad": None,
            "health": health,
            "head": {"x": snake_head[0], "y": snake_head[1]},
            "body": [{"x": x, "y": y} for x, y in snake_body],
            "customizations": {
                "color": [0, 0, 0],
                "head": None,
                "tail": None,
            },
        }

    you = snake("you", head, body)
    enemies = [
        snake(f"opponent-{index}", enemy[0], enemy[1])
        for index, enemy in enumerate(opponents)
    ]
    return GameState.model_validate(
        {
            "turn": turn,
            "game": {
                "id": game_id,
                "source": "test",
                "timeout": 500,
                "ruleset": {
                    "name": "blackout",
                    "version": "v1",
                    "settings": {
                        "foodSpawnChance": 0,
                        "hazardDamagePerTurn": 14,
                        "minimumFood": 0,
                        "viewRadius": 5,
                        "royale": {"shrinkEveryNTurns": 0},
                        "squad": {
                            "allowBodyCollisions": False,
                            "sharedElimination": False,
                            "sharedHealth": False,
                            "sharedLength": False,
                        },
                    },
                },
            },
            "board": {
                "height": 15,
                "width": 15,
                "food": [
                    {"x": x, "y": y, "spawn_turn": turn} for x, y in food
                ],
                "hazards": [],
                "snakes": [you, *enemies],
            },
            "you": you,
        }
    )


def _bare_agent(**overrides) -> PPOMCTSAgent:
    agent = PPOMCTSAgent.__new__(PPOMCTSAgent)
    defaults = {
        "time_budget_ms": 25.0,
        "max_iterations": 32,
        "min_iterations": 0,
        "wave_iterations": 32,
        "min_complete_waves": 2,
        "horizon": 5,
        "particles": 3,
        "rollout_steps": 1,
        "c_puct": 1.35,
        "adversarial_fraction": 0.2,
        "seed": 19,
        "timeout_margin_ms": 45.0,
        "safety_search": False,
    }
    defaults.update(overrides)
    for name, value in defaults.items():
        setattr(agent, name, value)
    agent._sessions = {}
    agent._mcts_sessions = {}
    agent._session_locks = {}
    agent._last_search = None
    agent._lock = threading.RLock()
    agent._model_lock = threading.RLock()
    return agent


class _ObservedRLock:
    """RLock that exposes an attempted entry without changing lock semantics."""

    def __init__(self):
        self._lock = threading.RLock()
        self.enter_attempted = threading.Event()

    def __enter__(self):
        self.enter_attempted.set()
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._lock.release()


def _patch_lightweight_constructor(monkeypatch) -> None:
    """Avoid loading a real checkpoint in constructor-validation tests."""

    def base_init(
        self,
        _model_path,
        *,
        symmetries=None,
        safety_search=None,
        device=None,
    ):
        self.symmetries = 1 if symmetries is None else symmetries
        self.safety_search = bool(safety_search)
        self.search_horizon = 10
        self.search_node_budget = 6000
        self.device = device or "cpu"
        self._sessions = {}
        self._lock = threading.RLock()
        self._model_fingerprint = "test-model"
        self._code_fingerprint = "base-code"
        self.received_model_path = _model_path

    monkeypatch.setattr(PPOAgent4, "__init__", base_init)
    monkeypatch.setattr(
        PPOMCTSAgent,
        "_policy_logits",
        lambda _self, _obs, _session: np.zeros(4, dtype=np.float32),
    )


@pytest.mark.parametrize(
    "parameter",
    (
        "time_budget_ms",
        "c_puct",
        "adversarial_fraction",
        "timeout_margin_ms",
    ),
)
@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_float_configuration_rejects_non_finite_values(parameter, value):
    with pytest.raises(ValueError, match="finite"):
        PPOMCTSAgent(
            None,
            max_iterations=1,
            min_iterations=0,
            **{parameter: value},
        )


def test_iteration_bounds_and_explicit_ppo_only_mode(monkeypatch):
    with pytest.raises(ValueError, match="min_iterations"):
        PPOMCTSAgent(None, max_iterations=3, min_iterations=4)

    _patch_lightweight_constructor(monkeypatch)
    agent = PPOMCTSAgent(None, max_iterations=0, min_iterations=999)

    assert agent.max_iterations == 0
    assert agent.min_iterations == 999


def test_wave_defaults_and_latency_headroom(monkeypatch):
    _patch_lightweight_constructor(monkeypatch)
    for name in (
        "PPO_MCTS_TIME_BUDGET_MS",
        "MCTS_TIME_BUDGET_MS",
        "PPO_MCTS_TIMEOUT_MARGIN_MS",
        "MCTS_TIMEOUT_MARGIN_MS",
        "PPO_MCTS_WAVE_ITERATIONS",
        "MCTS_WAVE_ITERATIONS",
        "PPO_MCTS_MIN_COMPLETE_WAVES",
        "MCTS_MIN_COMPLETE_WAVES",
    ):
        monkeypatch.delenv(name, raising=False)

    agent = PPOMCTSAgent(None, max_iterations=0)

    assert agent.time_budget_ms == 400.0
    assert agent.timeout_margin_ms == 80.0
    assert agent.wave_iterations == 16
    assert agent.min_complete_waves == 3


def test_mcts_safety_search_defaults_on_and_has_env_override(monkeypatch):
    _patch_lightweight_constructor(monkeypatch)
    monkeypatch.delenv("PPO_MCTS_PPO_SAFETY_SEARCH", raising=False)

    assert PPOMCTSAgent(None, max_iterations=0).safety_search is True

    monkeypatch.setenv("PPO_MCTS_PPO_SAFETY_SEARCH", "false")
    assert PPOMCTSAgent(None, max_iterations=0).safety_search is False


def test_mcts_model_default_and_environment_precedence(monkeypatch):
    _patch_lightweight_constructor(monkeypatch)
    monkeypatch.delenv("PPO_MCTS_MODEL_PATH", raising=False)
    monkeypatch.delenv("PPO_MODEL_PATH", raising=False)

    assert PPOMCTSAgent(max_iterations=0).received_model_path == DEFAULT_MCTS_MODEL_PATH

    monkeypatch.setenv("PPO_MODEL_PATH", "/tmp/shared.zip")
    assert PPOMCTSAgent(max_iterations=0).received_model_path == "/tmp/shared.zip"

    monkeypatch.setenv("PPO_MCTS_MODEL_PATH", "/tmp/mcts.zip")
    assert PPOMCTSAgent(max_iterations=0).received_model_path == "/tmp/mcts.zip"


def test_composite_fingerprint_changes_with_every_source(tmp_path):
    paths = tuple(
        tmp_path / name
        for name in ("ppo_mcts.py", "mcts_simulator.py", "ppo4.py")
    )
    for index, path in enumerate(paths):
        path.write_bytes(f"source-{index}".encode())
    original = _composite_fingerprint(paths)

    for path in paths:
        content = path.read_bytes()
        path.write_bytes(content + b"-changed")
        assert _composite_fingerprint(paths) != original
        path.write_bytes(content)

    assert _composite_fingerprint((*paths, tmp_path / "missing.py")) == "unavailable"


def test_constructor_fingerprints_all_runtime_sources(monkeypatch):
    _patch_lightweight_constructor(monkeypatch)
    captured = []

    def fingerprint(paths):
        captured.extend(paths)
        return "composite-code"

    monkeypatch.setattr("ppo_mcts._composite_fingerprint", fingerprint)
    agent = PPOMCTSAgent(None, max_iterations=0)

    assert agent._code_fingerprint == "composite-code"
    assert [Path(path).name for path in captured] == [
        "ppo_mcts.py",
        "mcts_simulator.py",
        "ppo4.py",
    ]


def test_extreme_ppo_prior_still_visits_every_root_candidate():
    state = _game_state()
    agent = _bare_agent(
        max_iterations=3,
        horizon=1,
        particles=1,
        rollout_steps=0,
        adversarial_fraction=0.0,
    )
    started = time.perf_counter()
    _, stats = agent._run_search(
        state,
        np.asarray([1000.0, -1000.0, -1000.0, -1000.0]),
        candidates=(0, 2, 3),
        fallback_action=0,
        rng=np.random.default_rng(7),
        search_started=started,
        deadline=started + 1.0,
        budget_ms=1000.0,
    )

    assert stats.iterations == 3
    assert stats.root_visits["up"] == 1
    assert stats.root_visits["left"] == 1
    assert stats.root_visits["right"] == 1


def test_retry_commits_lstm_once_and_final_hard_safety_wins():
    # UP is a visible body collision, DOWN is our non-vacating tail, LEFT
    # starves; only RIGHT reaches food.
    state = _game_state(
        head=(1, 1),
        body=((1, 1), (1, 0)),
        health=1,
        food=((2, 1),),
        opponents=(((1, 2), ((1, 2),)),),
    )
    agent = _bare_agent()
    calls = 0

    def policy(_obs, session):
        nonlocal calls
        calls += 1
        session.lstm_states = (torch.tensor([calls]), torch.tensor([calls]))
        session.episode_start = False
        return np.asarray([100.0, 0.0, 0.0, -100.0])

    def unsafe_search(*_args, **_kwargs):
        return 0, SearchDiagnostics(selected="up", fallback="mcts")

    agent._policy_logits = policy
    agent._run_search = unsafe_search
    agent.start(state)

    first = agent.move(state)
    committed = agent._sessions[agent._session_key(state)]
    retry = agent.move(state)

    assert first.move is Direction.RIGHT
    assert retry is first
    assert calls == 1
    assert committed.last_turn == state.turn
    assert committed.lstm_states[0].item() == 1
    assert agent.get_last_search_stats()["fallback"] == "hard-safety"


def test_duplicate_start_during_move_preserves_active_transaction_and_lock():
    state = _game_state(game_id="duplicate-start")
    agent = _bare_agent()
    entered_search = threading.Event()
    release_search = threading.Event()
    policy_calls = 0
    search_calls = 0

    def policy(_obs, _session):
        nonlocal policy_calls
        policy_calls += 1
        return np.asarray([3.0, 0.0, -1.0, 1.0])

    def search(
        _state,
        _logits,
        _candidates,
        fallback_action,
        *_args,
    ):
        nonlocal search_calls
        search_calls += 1
        entered_search.set()
        assert release_search.wait(timeout=2.0)
        return fallback_action, SearchDiagnostics(
            policy=Direction.UP.value,
            selected=Direction.UP.value,
            fallback="mcts",
        )

    agent._policy_logits = policy
    agent._run_search = search
    agent.start(state)
    key = agent._session_key(state)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(agent.move, state)
        assert entered_search.wait(timeout=2.0)
        active_session = agent._sessions[key]
        active_mcts_session = agent._mcts_sessions[key]
        active_lock = agent._session_locks[key]

        agent.start(state)

        assert agent._sessions[key] is active_session
        assert agent._mcts_sessions[key] is active_mcts_session
        assert agent._session_locks[key] is active_lock
        release_search.set()
        first = future.result(timeout=2.0)

    retry = agent.move(state)
    assert retry is first
    assert policy_calls == 1
    assert search_calls == 1


def test_start_after_end_creates_fresh_session_generation():
    state = _game_state(game_id="restart-after-end")
    agent = _bare_agent()
    policy_calls = 0

    def policy(_obs, _session):
        nonlocal policy_calls
        policy_calls += 1
        return np.asarray([3.0, 0.0, -1.0, 1.0])

    def search(
        _state,
        _logits,
        _candidates,
        fallback_action,
        *_args,
    ):
        return fallback_action, SearchDiagnostics(
            policy=Direction.UP.value,
            selected=Direction.UP.value,
            fallback="mcts",
        )

    agent._policy_logits = policy
    agent._run_search = search
    agent.start(state)
    key = agent._session_key(state)
    first = agent.move(state)
    old_session = agent._sessions[key]
    old_mcts_session = agent._mcts_sessions[key]
    old_lock = agent._session_locks[key]

    agent.end(state)
    assert key not in agent._sessions
    assert key not in agent._mcts_sessions
    assert key not in agent._session_locks

    agent.start(state)
    assert agent._sessions[key] is not old_session
    assert agent._mcts_sessions[key] is not old_mcts_session
    assert agent._session_locks[key] is not old_lock
    assert agent._sessions[key].last_turn == -1
    assert agent._mcts_sessions[key].last_turn == -1

    restarted = agent.move(state)
    assert restarted is not first
    assert restarted.move is first.move
    assert policy_calls == 2


@pytest.mark.parametrize("restart_before_stale_entry", (False, True))
def test_move_captured_before_end_cannot_recreate_or_overwrite_session(
    restart_before_stale_entry,
):
    state = _game_state(game_id="stale-pre-lock-move")
    agent = _bare_agent()
    policy_calls = 0
    search_calls = 0

    def policy(_obs, _session):
        nonlocal policy_calls
        policy_calls += 1
        return np.asarray([3.0, 0.0, -1.0, 1.0])

    def search(
        _state,
        _logits,
        _candidates,
        fallback_action,
        *_args,
    ):
        nonlocal search_calls
        search_calls += 1
        return fallback_action, SearchDiagnostics(
            policy=Direction.UP.value,
            selected=Direction.UP.value,
            fallback="mcts",
        )

    agent._policy_logits = policy
    agent._run_search = search
    agent.start(state)
    key = agent._session_key(state)
    observed_lock = _ObservedRLock()
    with agent._lock:
        agent._session_locks[key] = observed_lock
        ended_session = agent._sessions[key]
        ended_mcts_session = agent._mcts_sessions[key]

    restarted_session = None
    restarted_mcts_session = None
    restarted_lock = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        with observed_lock:
            observed_lock.enter_attempted.clear()
            stale = pool.submit(agent.move, state)
            assert observed_lock.enter_attempted.wait(timeout=2.0)

            agent.end(state)
            assert key not in agent._sessions
            assert key not in agent._mcts_sessions
            assert key not in agent._session_locks
            if restart_before_stale_entry:
                agent.start(state)
                restarted_session = agent._sessions[key]
                restarted_mcts_session = agent._mcts_sessions[key]
                restarted_lock = agent._session_locks[key]

        with pytest.raises(RuntimeError, match="session ended or restarted"):
            stale.result(timeout=2.0)

    assert policy_calls == 0
    assert search_calls == 0
    assert ended_session.last_turn == -1
    assert ended_mcts_session.last_turn == -1
    if not restart_before_stale_entry:
        assert key not in agent._sessions
        assert key not in agent._mcts_sessions
        assert key not in agent._session_locks
        return

    assert agent._sessions[key] is restarted_session
    assert agent._mcts_sessions[key] is restarted_mcts_session
    assert agent._session_locks[key] is restarted_lock
    assert restarted_session.last_turn == -1
    assert restarted_mcts_session.last_turn == -1
    assert restarted_lock is not observed_lock

    assert agent.move(state).move is Direction.UP
    assert policy_calls == 1
    assert search_calls == 1


def test_move_without_start_lazily_creates_one_retryable_generation():
    state = _game_state(game_id="lazy-move")
    agent = _bare_agent()
    policy_calls = 0

    def policy(_obs, _session):
        nonlocal policy_calls
        policy_calls += 1
        return np.asarray([3.0, 0.0, -1.0, 1.0])

    agent._policy_logits = policy
    agent._run_search = lambda *_args, **_kwargs: (
        0,
        SearchDiagnostics(
            policy=Direction.UP.value,
            selected=Direction.UP.value,
            fallback="mcts",
        ),
    )

    first = agent.move(state)
    key = agent._session_key(state)
    session = agent._sessions[key]
    mcts_session = agent._mcts_sessions[key]
    session_lock = agent._session_locks[key]
    retry = agent.move(state)

    assert retry is first
    assert agent._sessions[key] is session
    assert agent._mcts_sessions[key] is mcts_session
    assert agent._session_locks[key] is session_lock
    assert session.last_turn == state.turn
    assert mcts_session.last_turn == state.turn
    assert policy_calls == 1


def test_root_gate_excludes_unproven_contested_hard_move():
    state = _game_state()
    agent = _bare_agent(safety_search=True)
    captured = {}
    agent._policy_logits = lambda _obs, _session: np.asarray(
        [3.0, 2.0, 1.0, 0.0]
    )
    agent._action_tiers = lambda _state: ([0, 1, 2], [0, 1])
    agent._safe_candidates = lambda *_args: ([1], {})

    def search(_state, _logits, candidates, fallback_action, *_args):
        captured["candidates"] = tuple(candidates)
        captured["fallback"] = fallback_action
        return fallback_action, SearchDiagnostics(policy="down", selected="down")

    agent._run_search = search
    agent.start(state)

    assert agent.move(state).move is Direction.DOWN
    assert captured == {"candidates": (1,), "fallback": 1}


def test_root_gate_accepts_hard_escape_returned_by_complete_trap_search():
    state = _game_state()
    agent = _bare_agent(safety_search=True)
    captured = {}
    agent._policy_logits = lambda _obs, _session: np.asarray(
        [3.0, 2.0, 1.0, 0.0]
    )
    # Action 2 is hard-only in the tier contract.  It becomes admissible only
    # because PPO4's complete trap search explicitly returned it as the escape.
    agent._action_tiers = lambda _state: ([0, 1, 2], [0, 1])
    agent._safe_candidates = lambda *_args: ([2], {})

    def search(_state, _logits, candidates, fallback_action, *_args):
        captured["candidates"] = tuple(candidates)
        captured["fallback"] = fallback_action
        return fallback_action, SearchDiagnostics(policy="left", selected="left")

    agent._run_search = search
    agent.start(state)

    assert agent.move(state).move is Direction.LEFT
    assert captured == {"candidates": (2,), "fallback": 2}


def test_different_game_searches_overlap_while_model_forward_is_serialized():
    first_state = _game_state(game_id="parallel-one")
    second_state = _game_state(game_id="parallel-two")
    agent = _bare_agent(time_budget_ms=100.0)
    counters_lock = threading.Lock()
    searches_ready = threading.Barrier(2)
    active_model = 0
    max_active_model = 0
    active_search = 0
    max_active_search = 0

    def policy(_obs, _session):
        nonlocal active_model, max_active_model
        with counters_lock:
            active_model += 1
            max_active_model = max(max_active_model, active_model)
        try:
            time.sleep(0.02)
            return np.asarray([2.0, 0.0, -1.0, 1.0])
        finally:
            with counters_lock:
                active_model -= 1

    def search(
        _state,
        _logits,
        _candidates,
        fallback_action,
        _rng,
        _started,
        _deadline,
        _budget_ms,
    ):
        nonlocal active_search, max_active_search
        with counters_lock:
            active_search += 1
            max_active_search = max(max_active_search, active_search)
        try:
            searches_ready.wait(timeout=2.0)
            time.sleep(0.03)
            return fallback_action, SearchDiagnostics(
                selected=Direction.UP.value,
                fallback="mcts",
            )
        finally:
            with counters_lock:
                active_search -= 1

    agent._policy_logits = policy
    agent._run_search = search
    agent.start(first_state)
    agent.start(second_state)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(agent.move, state) for state in (first_state, second_state)
        ]
        moves = [future.result(timeout=3.0) for future in futures]

    assert all(move.move in Direction for move in moves)
    assert max_active_model == 1
    assert max_active_search == 2
    assert len(agent._sessions) == 2


def test_fixed32_move_uses_legacy_search_without_ensemble_or_master_draw():
    state = _game_state()
    # Production roots are 16 simulations, but the historical fixed-32 model
    # screen remains a single tree for exact backwards comparability.
    agent = _bare_agent(max_iterations=32, wave_iterations=16)
    calls = []
    agent._policy_logits = lambda _obs, _session: np.asarray(
        [2.0, 0.0, -1.0, 1.0]
    )

    def legacy(
        _state,
        _logits,
        _candidates,
        fallback_action,
        *_args,
    ):
        calls.append("legacy")
        return fallback_action, SearchDiagnostics(policy="up", selected="up")

    def ensemble(*_args, **_kwargs):
        raise AssertionError("fixed32 must not enter the ensemble wrapper")

    agent._run_search = legacy
    agent._run_wave_ensemble = ensemble
    agent.start(state)

    agent.move(state)

    assert calls == ["legacy"]


def _script_wave_search(agent, specifications):
    """Install deterministic wave results and return the requested limits."""

    pending = list(specifications)
    limits = []

    def search(
        _state,
        _logits,
        _candidates,
        _fallback_action,
        _rng,
        _started,
        _deadline,
        _budget_ms,
        *,
        iteration_limit=None,
        minimum_iterations=None,
    ):
        assert minimum_iterations == min(agent.min_iterations, iteration_limit)
        limits.append(iteration_limit)
        specification = pending.pop(0)
        action = specification[0]
        iterations = (
            iteration_limit if len(specification) < 2 else specification[1]
        )
        particles = agent.particles if len(specification) < 3 else specification[2]
        deadline_hit = False if len(specification) < 4 else specification[3]
        names = tuple(direction.value for direction in Direction)
        visits = {name: 0 for name in names}
        q = {name: 0.0 for name in names}
        action_name = (
            Direction.UP,
            Direction.DOWN,
            Direction.LEFT,
            Direction.RIGHT,
        )[action].value
        visits[action_name] = iterations
        q[action_name] = 0.25 + action * 0.1
        return action, SearchDiagnostics(
            iterations=iterations,
            nodes=iterations + 1,
            depth=4 + action,
            particles=particles,
            deadline_hit=deadline_hit,
            fallback="mcts",
            selected=action_name,
            root_visits=visits,
            root_q=q,
        )

    agent._run_search = search
    return limits


def _run_scripted_ensemble(agent, state, rng=None, deadline=None):
    started = time.perf_counter()
    return agent._run_wave_ensemble(
        state,
        np.asarray([2.0, 0.0, -1.0, 1.0]),
        candidates=(0, 2, 3),
        fallback_action=0,
        rng=np.random.default_rng(31) if rng is None else rng,
        search_started=started,
        deadline=started + 1.0 if deadline is None else deadline,
        budget_ms=1000.0,
    )


def test_three_complete_waves_override_ppo_with_two_to_one_majority():
    state = _game_state()
    agent = _bare_agent(max_iterations=96, wave_iterations=32)
    limits = _script_wave_search(agent, ((2,), (0,), (2,)))
    master = np.random.default_rng(31)
    reference = np.random.default_rng(31)

    action, stats = _run_scripted_ensemble(agent, state, rng=master)

    assert action == 2
    assert limits == [32, 32, 32]
    assert stats.iterations == 96
    assert stats.nodes == 99
    assert stats.waves_started == stats.waves == 3
    assert stats.waves_completed == stats.complete_waves == 3
    assert stats.waves_discarded == 0
    assert stats.wave_votes == stats.votes == {"up": 1, "left": 2}
    assert stats.fallback == "mcts"
    assert stats.decision_reason == "alternative-wave-majority"
    assert stats.root_visits["left"] == 64
    assert stats.root_q["left"] == pytest.approx(0.45)
    for _ in range(3):
        reference.bit_generator.random_raw()
    assert master.bit_generator.random_raw() == reference.bit_generator.random_raw()


def test_min_iterations_65_requires_three_complete_waves():
    state = _game_state()
    agent = _bare_agent(
        max_iterations=96,
        min_iterations=65,
        wave_iterations=32,
        min_complete_waves=2,
    )
    limits = _script_wave_search(agent, ((2,), (2,), (2,)))

    action, stats = _run_scripted_ensemble(agent, state)

    assert action == 2
    assert limits == [32, 32, 32]
    assert stats.complete_waves == 3
    assert stats.votes == {"left": 3}
    assert stats.fallback == "mcts"
    assert stats.decision_reason == "alternative-wave-majority"


def test_min_iterations_65_rejects_only_two_complete_waves():
    state = _game_state()
    agent = _bare_agent(
        max_iterations=96,
        min_iterations=65,
        wave_iterations=32,
        min_complete_waves=2,
    )
    _script_wave_search(agent, ((2,), (2,), (2, 10)))

    action, stats = _run_scripted_ensemble(agent, state)

    assert action == 0
    assert stats.complete_waves == 2
    assert stats.votes == {"left": 2}
    assert stats.fallback == "insufficient-search"
    assert stats.decision_reason == "insufficient-complete-waves"


def test_split_complete_waves_preserve_ppo():
    state = _game_state()
    agent = _bare_agent(max_iterations=64, wave_iterations=32)
    _script_wave_search(agent, ((2,), (3,)))

    action, stats = _run_scripted_ensemble(agent, state)

    assert action == 0
    assert stats.complete_waves == 2
    assert stats.wave_votes == {"left": 1, "right": 1}
    assert stats.fallback == "insufficient-search"
    assert stats.decision_reason == "no-strict-wave-majority"


def test_only_one_complete_wave_preserves_ppo_and_ignores_partial_vote():
    state = _game_state()
    agent = _bare_agent(max_iterations=64, wave_iterations=32)
    _script_wave_search(agent, ((2,), (2, 10)))

    action, stats = _run_scripted_ensemble(agent, state)

    assert action == 0
    assert stats.iterations == 42
    assert stats.voting_iterations == 32
    assert stats.discarded_iterations == 10
    assert stats.waves_completed == 1
    assert stats.waves_discarded == 1
    assert stats.wave_votes == {"left": 1}
    assert stats.fallback == "insufficient-search"
    assert stats.decision_reason == "insufficient-complete-waves"


def test_full_iteration_wave_with_short_particle_set_is_discarded():
    state = _game_state()
    agent = _bare_agent(max_iterations=64, wave_iterations=32, particles=4)
    _script_wave_search(agent, ((2, 32, 3),))

    action, stats = _run_scripted_ensemble(agent, state)

    assert action == 0
    assert stats.iterations == 32
    assert stats.complete_waves == 0
    assert stats.waves_discarded == 1
    assert stats.votes == {}
    assert stats.root_visits == {"up": 0, "down": 0, "left": 0, "right": 0}


def test_partial_last_wave_is_ignored_after_valid_majority():
    state = _game_state()
    agent = _bare_agent(max_iterations=96, wave_iterations=32)
    _script_wave_search(agent, ((2,), (2,), (0, 10)))

    action, stats = _run_scripted_ensemble(agent, state)

    assert action == 2
    assert stats.iterations == 74
    assert stats.complete_waves == 2
    assert stats.waves_discarded == 1
    assert stats.votes == {"left": 2}
    assert stats.root_visits["up"] == 0
    assert stats.root_visits["left"] == 64


def test_wave_iteration_cap_includes_but_does_not_vote_short_remainder():
    state = _game_state()
    agent = _bare_agent(max_iterations=70, wave_iterations=32)
    limits = _script_wave_search(agent, ((2,), (2,), (0,)))

    action, stats = _run_scripted_ensemble(agent, state)

    assert action == 2
    assert limits == [32, 32, 6]
    assert stats.iterations == 70
    assert stats.complete_waves == 2
    assert stats.discarded_iterations == 6
    assert stats.votes == {"left": 2}


def test_wave_guard_does_not_start_expensive_second_tree(monkeypatch):
    state = _game_state()
    agent = _bare_agent(max_iterations=96, wave_iterations=32)
    clock = [10.0]
    calls = []

    monkeypatch.setattr("ppo_mcts.time.perf_counter", lambda: clock[0])

    def search(
        _state,
        _logits,
        _candidates,
        _fallback_action,
        _rng,
        _started,
        _deadline,
        _budget_ms,
        **_kwargs,
    ):
        calls.append(1)
        clock[0] += 0.100
        return 2, SearchDiagnostics(
            iterations=32,
            nodes=33,
            particles=agent.particles,
            fallback="mcts",
            selected="left",
            root_visits={"up": 8, "down": 0, "left": 24, "right": 0},
            root_q={"up": 0.1, "down": 0.0, "left": 0.4, "right": 0.0},
        )

    agent._run_search = search
    action, stats = _run_scripted_ensemble(
        agent,
        state,
        rng=np.random.default_rng(2),
        deadline=10.200,
    )

    assert action == 0
    assert len(calls) == 1
    assert stats.iterations == 32
    assert stats.stopped_for_wave_guard
    assert stats.complete_waves == 1
    assert stats.decision_reason == "insufficient-complete-waves"


def test_deadline_partial_wave_is_discarded_without_extension(monkeypatch):
    state = _game_state()
    agent = _bare_agent(max_iterations=64, wave_iterations=32)
    clock = [20.0]
    monkeypatch.setattr("ppo_mcts.time.perf_counter", lambda: clock[0])

    def search(*_args, **_kwargs):
        clock[0] = 20.051
        return 2, SearchDiagnostics(
            iterations=12,
            nodes=13,
            particles=agent.particles,
            deadline_hit=True,
            fallback="insufficient-search",
            selected="left",
        )

    agent._run_search = search
    action, stats = _run_scripted_ensemble(
        agent,
        state,
        rng=np.random.default_rng(4),
        deadline=20.050,
    )

    assert action == 0
    assert stats.iterations == stats.discarded_iterations == 12
    assert stats.complete_waves == 0
    assert stats.waves_started == stats.waves_discarded == 1
    assert stats.deadline_hit
    assert stats.fallback == "insufficient-search"


def test_wave_error_discards_entire_ensemble_and_returns_ppo():
    state = _game_state()
    agent = _bare_agent(max_iterations=96, wave_iterations=32)

    def search(*_args, **_kwargs):
        raise RuntimeError("wave failed")

    agent._run_search = search
    action, stats = _run_scripted_ensemble(agent, state)

    assert action == 0
    assert stats.fallback == "ppo"
    assert stats.decision_reason == "wave-error"
    assert stats.waves_started == stats.waves_discarded == 1
    assert stats.error == "RuntimeError: wave failed"


def test_small_deadline_returns_search_stats_without_large_overrun():
    state = _game_state(
        opponents=(((10, 10), ((10, 10), (10, 11), (10, 12))),),
    )
    agent = _bare_agent(
        time_budget_ms=6.0,
        max_iterations=50_000,
        horizon=8,
        particles=4,
        rollout_steps=2,
    )
    agent._policy_logits = lambda _obs, _session: np.asarray(
        [2.0, 0.0, -1.0, 1.0]
    )
    agent.start(state)

    started = time.perf_counter()
    agent.move(state)
    wall_ms = (time.perf_counter() - started) * 1000.0
    stats = agent.get_last_search_stats()

    assert stats is not None
    assert stats["deadline_ms"] == 6.0
    assert stats["iterations"] >= 1
    assert stats["nodes"] >= 1
    assert stats["depth"] >= 1
    assert set(stats["root_prior"]) == {"up", "down", "left", "right"}
    assert set(stats["root_visits"]) == {"up", "down", "left", "right"}
    assert set(stats["root_q"]) == {"up", "down", "left", "right"}
    assert stats["policy"] in {"up", "down", "left", "right"}
    assert isinstance(stats["changed_policy"], bool)
    assert stats["deadline_hit"]
    assert wall_ms < 75.0

    # Public metrics are detached; callers cannot mutate agent state.
    stats["iterations"] = -1
    assert agent.get_last_search_stats()["iterations"] >= 1


def test_undersampled_search_preserves_ppo_fallback():
    state = _game_state()
    agent = _bare_agent(
        max_iterations=3,
        min_iterations=4,
        horizon=1,
        particles=1,
        rollout_steps=0,
        adversarial_fraction=0.0,
    )
    started = time.perf_counter()
    action, stats = agent._run_search(
        state,
        np.asarray([-1000.0, -1000.0, 1000.0, -1000.0]),
        candidates=(0, 2, 3),
        fallback_action=3,
        rng=np.random.default_rng(11),
        search_started=started,
        deadline=started + 1.0,
        budget_ms=1000.0,
    )

    assert stats.iterations == 3
    assert action == 3
    assert stats.fallback == "insufficient-search"
