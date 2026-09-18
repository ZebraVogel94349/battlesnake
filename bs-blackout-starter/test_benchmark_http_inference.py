from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pytest

import benchmark_http_inference as benchmark_module
from battlesnake_types import Direction, MoveAction


class _FakeMCTSAgent:
    latest: _FakeMCTSAgent | None = None
    fail_moves = False

    def __init__(self, model=None, **_kwargs):
        type(self).latest = self
        self.model_path = Path(model or "fake-model.zip")
        self.init_kwargs = dict(_kwargs)
        self.sessions: dict[tuple[str, str], dict] = {}
        self.last_stats: dict | None = None
        self.search_calls = 0

    def get_name(self):
        return "benchmark-test"

    def get_author(self):
        return "test"

    def get_color(self):
        return "#000000"

    def get_diagnostics(self):
        return {
            "active_sessions": len(self.sessions),
            "search_calls": self.search_calls,
        }

    def get_last_search_stats(self):
        return copy.deepcopy(self.last_stats)

    @staticmethod
    def _key(state):
        return state.game.id, state.you.id

    def start(self, state):
        self.sessions[self._key(state)] = {
            "turn": -1,
            "move": None,
        }

    def move(self, state):
        if self.fail_moves:
            raise RuntimeError("injected move failure")
        session = self.sessions[self._key(state)]
        if state.turn == session["turn"]:
            return session["move"]
        if state.turn < session["turn"]:
            raise ValueError("out-of-order test turn")
        self.search_calls += 1
        direction = Direction.RIGHT if state.turn % 2 == 0 else Direction.UP
        move = MoveAction(move=direction)
        session.update(turn=state.turn, move=move)
        self.last_stats = {
            "selected": direction.value,
            "search_call": self.search_calls,
        }
        return move

    def end(self, state):
        self.sessions.pop(self._key(state), None)


def _args(*, moves: int = 3) -> argparse.Namespace:
    return argparse.Namespace(
        model=None,
        moves=moves,
        time_budget_ms=None,
        max_iterations=None,
        min_iterations=None,
        seed=12_345,
        sla_ms=500.0,
        fail_on_sla=True,
        http_timeout_seconds=2.0,
        output=None,
    )


def _event(
    request_id: str,
    phase: str,
    *,
    turn: int = 0,
    game_id: str = "game",
) -> dict:
    event = {
        "request_id": request_id,
        "phase": phase,
        "endpoint": "/move",
        "game_id": game_id,
        "turn": turn,
        "you_id": "you",
    }
    if phase == "completed":
        event.update(
            latency_ms=10.0,
            result={
                "agent_latency_ms": 9.0,
                "move": {"move": "right"},
            },
        )
    return event


def test_request_log_validation_requires_exact_paired_moves():
    events = [
        _event("a", "received", turn=0),
        _event("a", "completed", turn=0),
        _event("b", "received", turn=1),
        _event("b", "completed", turn=1),
    ]

    benchmark_module._validate_request_pairs(events)
    completed = benchmark_module._completed_move_events(
        events,
        game_id="game",
        expected_turns=(0, 1),
    )

    assert len(completed) == 2


def test_request_log_validation_rejects_failed_unmatched_and_missing_metric():
    failed = _event("a", "failed")
    failed.update(error_type="RuntimeError", error="boom")
    with pytest.raises(RuntimeError, match="failed"):
        benchmark_module._validate_request_pairs(
            [_event("a", "received"), failed]
        )

    with pytest.raises(RuntimeError, match="unmatched"):
        benchmark_module._validate_request_pairs([_event("a", "received")])

    events = [_event("a", "received"), _event("a", "completed")]
    events[1]["result"].pop("agent_latency_ms")
    with pytest.raises(RuntimeError, match="agent latency"):
        benchmark_module._completed_move_events(
            events,
            game_id="game",
            expected_turns=(0,),
        )


def test_compaction_validation_fails_when_move_sidecar_is_missing(tmp_path):
    summary = tmp_path / "game.json"
    summary.write_text(
        '{"game_id":"game","ended_at":"now","moves":[{}]}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="sidecar is missing"):
        benchmark_module._wait_for_compaction(
            tmp_path,
            "game",
            expected_moves=1,
            timeout_seconds=0.1,
        )


def test_http_benchmark_separates_retry_and_primary_samples(monkeypatch):
    _FakeMCTSAgent.fail_moves = False
    monkeypatch.setattr(benchmark_module, "PPOMCTSAgent", _FakeMCTSAgent)

    report = benchmark_module.benchmark(_args(moves=3))

    assert report["schema_version"] == 4
    assert report["sla_metric"] == "client_round_trip_ms"
    assert report["sla"] == {
        "metric": "client_round_trip_ms",
        "threshold_ms": 500.0,
        "enforced": True,
        "passed": True,
        "violations": 0,
        "scope": "primary-and-retry-probe-/move-requests",
        "samples": 6,
        "breakdown": {
            "primary": {
                "metric": "client_round_trip_ms",
                "threshold_ms": 500.0,
                "enforced": True,
                "passed": True,
                "violations": 0,
                "samples": 3,
            },
            "retry_probe": {
                "metric": "client_round_trip_ms",
                "threshold_ms": 500.0,
                "enforced": True,
                "passed": True,
                "violations": 0,
                "samples": 3,
            },
        },
    }
    assert report["client_round_trip_ms"]["count"] == 3
    assert len(report["client_round_trip_samples_ms"]) == 3
    assert report["all_move_client_round_trip_ms"]["count"] == 6
    assert len(report["all_move_client_round_trip_samples_ms"]) == 6
    assert report["server_pre_completion_log_ms"]["count"] == 3
    assert report["server_to_agent_return_ms"]["count"] == 3
    assert "server_handler_ms" not in report
    assert "agent_move_ms" not in report
    assert report["log_integrity"] == {
        "all_requests_paired": True,
        "failed_requests": 0,
        "paired_requests": 11,
        "primary_completed_moves": 3,
        "retry_probe_completed_moves": 3,
        "async_writer": {
            "pending": 0,
            "dropped": 0,
            "errors": 0,
            "worker_alive": False,
        },
        "game_log_store": {"move_lock_contention_drops": 0},
    }
    assert report["retry_probe"]["duplicate_start_preserved_retry_cache"] is True
    assert report["retry_probe"]["same_turn_response_identical"] is True
    assert report["retry_probe"]["same_turn_stats_identical"] is True
    assert report["retry_probe"]["next_turn_succeeded"] is True
    assert report["retry_probe"]["client_round_trip_summary_ms"]["count"] == 3
    assert len(report["retry_probe"]["client_round_trip_samples_ms"]) == 3
    # Three primary turns plus two actual retry-game searches. The duplicate
    # same-turn request must not advance the fake recurrent transaction.
    assert report["agent_diagnostics"] == {
        "active_sessions": 0,
        "search_calls": 5,
    }
    assert _FakeMCTSAgent.latest is not None
    assert _FakeMCTSAgent.latest.init_kwargs["seed"] == 12_345
    assert "node" not in report["runtime"]
    assert "hostname" not in report["runtime"]
    assert "state-complexity corpus" in report["workload_limit"]
    assert report["workload"]["seed"] == 12_345
    assert report["workload"]["primary"]["game_id"] == (
        "http-benchmark-seed-12345"
    )
    assert report["workload"]["retry_probe"]["game_id"] == (
        "http-retry-probe-seed-12345"
    )
    assert report["workload"]["retry_probe"]["duplicate_start_after_first_move"]
    assert set(report["source_sha256"]) == {
        "benchmark_http_inference.py",
        "battlesnake_server.py",
    }
    assert all(len(value) == 64 for value in report["source_sha256"].values())


def test_seeded_request_workload_is_reproducible():
    game_id = "deterministic-http-game"
    first = benchmark_module._request_state(game_id, 91)
    second = benchmark_module._request_state(game_id, 91)

    assert first == second
    assert benchmark_module._sha256_json(first) == benchmark_module._sha256_json(second)


def test_sla_summary_uses_strict_500ms_limit():
    status = benchmark_module._sla_summary(
        [499.0, 500.0, 500.001], threshold_ms=500.0, enforced=True
    )

    assert status == {
        "metric": "client_round_trip_ms",
        "threshold_ms": 500.0,
        "enforced": True,
        "passed": False,
        "violations": 1,
    }


def test_combined_sla_fails_when_only_retry_probe_is_slow():
    status = benchmark_module._combined_sla_summary(
        [10.0, 20.0, 30.0],
        [12.0, 500.001, 8.0],
        threshold_ms=500.0,
        enforced=True,
    )

    assert status["passed"] is False
    assert status["violations"] == 1
    assert status["samples"] == 6
    assert status["breakdown"]["primary"]["passed"] is True
    assert status["breakdown"]["primary"]["violations"] == 0
    assert status["breakdown"]["retry_probe"]["passed"] is False
    assert status["breakdown"]["retry_probe"]["violations"] == 1


def test_benchmark_gates_injected_slow_retry_with_fast_primary(monkeypatch):
    _FakeMCTSAgent.fail_moves = False
    monkeypatch.setattr(benchmark_module, "PPOMCTSAgent", _FakeMCTSAgent)
    real_retry_probe = benchmark_module._retry_probe

    def retry_probe_with_slow_sample(**kwargs):
        report, responses = real_retry_probe(**kwargs)
        samples = [10.0, 501.0, 10.0]
        report["client_round_trip_samples_ms"] = samples
        report["client_round_trip_summary_ms"] = (
            benchmark_module._latency_summary(samples)
        )
        return report, responses

    monkeypatch.setattr(
        benchmark_module,
        "_retry_probe",
        retry_probe_with_slow_sample,
    )

    report = benchmark_module.benchmark(_args(moves=2))

    assert report["sla"]["breakdown"]["primary"]["passed"] is True
    assert report["sla"]["breakdown"]["retry_probe"]["passed"] is False
    assert report["sla"]["passed"] is False
    assert report["sla"]["violations"] == 1


def test_main_fails_when_only_retry_probe_violates_sla(monkeypatch):
    def retry_failed_report(_args):
        return {
            "sla": benchmark_module._combined_sla_summary(
                [10.0, 20.0],
                [10.0, 501.0, 10.0],
                threshold_ms=500.0,
                enforced=True,
            )
        }

    monkeypatch.setattr(benchmark_module, "benchmark", retry_failed_report)

    with pytest.raises(SystemExit, match="1 request.*exceeded 500 ms"):
        benchmark_module.main(["--moves", "1"])


def test_main_fails_on_sla_violation_by_default(monkeypatch):
    def failed_report(_args):
        return {
            "sla": {
                "threshold_ms": 500.0,
                "enforced": True,
                "passed": False,
                "violations": 1,
            }
        }

    monkeypatch.setattr(benchmark_module, "benchmark", failed_report)

    with pytest.raises(SystemExit, match="1 request.*exceeded 500 ms"):
        benchmark_module.main(["--moves", "1"])


def test_main_allows_explicit_sla_gate_disable(monkeypatch):
    def failed_report(args):
        return {
            "sla": {
                "threshold_ms": args.sla_ms,
                "enforced": args.fail_on_sla,
                "passed": False,
                "violations": 2,
            }
        }

    monkeypatch.setattr(benchmark_module, "benchmark", failed_report)

    assert benchmark_module.main(["--moves", "1", "--no-fail-on-sla"]) is None


def test_exception_path_ends_session_and_closes_server(monkeypatch):
    real_make_server = benchmark_module.make_server
    closed = []

    def tracked_make_server(*args, **kwargs):
        server = real_make_server(*args, **kwargs)
        original_close = server.server_close

        def tracked_close():
            closed.append(True)
            original_close()

        server.server_close = tracked_close
        return server

    _FakeMCTSAgent.fail_moves = True
    monkeypatch.setattr(benchmark_module, "PPOMCTSAgent", _FakeMCTSAgent)
    monkeypatch.setattr(benchmark_module, "make_server", tracked_make_server)

    with pytest.raises(Exception, match="500: INTERNAL SERVER ERROR"):
        benchmark_module.benchmark(_args(moves=1))

    # Werkzeug may also close from its serve_forever() finally block; our
    # explicit close is intentionally idempotent.
    assert closed
    assert _FakeMCTSAgent.latest is not None
    assert _FakeMCTSAgent.latest.sessions == {}
    _FakeMCTSAgent.fail_moves = False
