from __future__ import annotations

import json
import threading
import time

from battlesnake_server import AsyncLogWriter, GameLogStore, start_server
from battlesnake_types import Direction, MoveAction
from test_mcts_simulator import _blackout_request


class _Agent:
    def get_name(self):
        return "test"

    def get_author(self):
        return "test"

    def get_color(self):
        return "#000000"

    def get_diagnostics(self):
        return {"kind": "test"}

    def start(self, _game_state):
        return None

    def move(self, _game_state):
        return MoveAction(move=Direction.RIGHT)

    def end(self, _game_state):
        return None


class _CountingAgent(_Agent):
    def __init__(self):
        self.start_calls = 0
        self.move_calls = 0

    def start(self, _game_state):
        self.start_calls += 1

    def move(self, game_state):
        self.move_calls += 1
        return super().move(game_state)


def test_async_log_writer_is_fail_open_and_keeps_draining() -> None:
    writer = AsyncLogWriter(max_pending=4)
    completed = []

    def fail() -> None:
        raise OSError("disk full")

    assert writer.submit(fail)
    assert writer.submit(completed.append, "after-error")
    assert writer.flush()
    assert completed == ["after-error"]
    assert writer.stats()["errors"] == 1
    assert writer.close()


def test_async_log_writer_drops_full_queue_without_blocking() -> None:
    writer = AsyncLogWriter(max_pending=1)
    entered = threading.Event()
    release = threading.Event()

    def block() -> None:
        entered.set()
        release.wait(timeout=1.0)

    assert writer.submit(block)
    assert entered.wait(timeout=0.5)
    assert writer.submit(lambda: None)
    started = time.perf_counter()
    assert writer.submit(lambda: None) is False
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert elapsed_ms < 50.0

    release.set()
    assert writer.close(timeout_seconds=2.0)
    assert writer.stats()["dropped"] == 1


def test_move_logging_is_append_only_in_request_path(tmp_path) -> None:
    state = _blackout_request()
    store = GameLogStore(_Agent(), log_dir=str(tmp_path))
    move = MoveAction(move=Direction.RIGHT)

    store.log_start(state)
    assert store.flush()
    json_path = next(tmp_path.glob("*.json"))
    store.log_move(state, move, 0.44)
    store.log_move(state, move, 0.43)
    assert store.flush()

    # The conventional document is compacted at /end, not rewritten on every
    # timed move.  Individual turns are already durable in a bounded JSONL append.
    assert json.loads(json_path.read_text(encoding="utf-8"))["moves"] == []
    sidecar = next(tmp_path.glob("*.moves.jsonl"))
    records = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert [record["move"]["move"] for record in records] == ["right", "right"]
    assert store.close()


def test_flask_app_can_be_exercised_without_opening_a_socket(tmp_path) -> None:
    state = _blackout_request()
    app = start_server(_Agent(), 0, run=False, log_dir=str(tmp_path))
    client = app.test_client()

    assert client.post("/start", json=state.model_dump(mode="json")).status_code == 200
    response = client.post("/move", json=state.model_dump(mode="json"))

    assert response.status_code == 200
    assert response.get_json() == {"move": "right"}
    assert app.extensions["battlesnake_log_writer"].close()


def test_end_compacts_buffered_moves_asynchronously(tmp_path) -> None:
    state = _blackout_request()
    store = GameLogStore(_Agent(), log_dir=str(tmp_path))
    store.log_start(state)
    assert store.flush()
    store.log_move(state, MoveAction(move=Direction.RIGHT), 0.44)

    started = time.perf_counter()
    assert store.log_end(state)
    call_ms = (time.perf_counter() - started) * 1000.0
    json_path = next(tmp_path.glob("*.json"))
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline:
        document = json.loads(json_path.read_text(encoding="utf-8"))
        if document["ended_at"] is not None:
            break
        time.sleep(0.005)

    assert call_ms < 100.0
    assert document["ended_at"] is not None
    assert len(document["moves"]) == 1
    assert store.close()


def test_move_response_is_fail_open_when_game_logging_breaks(tmp_path, monkeypatch) -> None:
    state = _blackout_request()
    app = start_server(_Agent(), 0, run=False, log_dir=str(tmp_path))
    game_logs = app.extensions["battlesnake_game_logs"]
    monkeypatch.setattr(
        game_logs,
        "log_move",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    response = app.test_client().post(
        "/move", json=state.model_dump(mode="json")
    )

    assert response.status_code == 200
    assert response.get_json() == {"move": "right"}
    assert app.extensions["battlesnake_log_writer"].close()


def test_duplicate_start_does_not_reset_active_agent_session(tmp_path) -> None:
    state = _blackout_request()
    agent = _CountingAgent()
    app = start_server(agent, 0, run=False, log_dir=str(tmp_path))
    client = app.test_client()
    payload = state.model_dump(mode="json")

    assert client.post("/start", json=payload).status_code == 200
    assert client.post("/move", json=payload).status_code == 200
    assert client.post("/start", json=payload).status_code == 200
    assert client.post("/move", json=payload).status_code == 200
    assert client.post("/end", json=payload).status_code == 200
    assert client.post("/start", json=payload).status_code == 200

    assert agent.start_calls == 2
    assert agent.move_calls == 2
    writer = app.extensions["battlesnake_log_writer"]
    assert writer.close()
    events = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    starts = [
        event
        for event in events
        if event["endpoint"] == "/start" and event["phase"] == "completed"
    ]
    assert [event["result"]["duplicate"] for event in starts] == [
        False,
        True,
        False,
    ]


def test_slow_request_log_io_does_not_delay_move_response(tmp_path, monkeypatch) -> None:
    state = _blackout_request()
    app = start_server(_Agent(), 0, run=False, log_dir=str(tmp_path))
    client = app.test_client()
    writer = app.extensions["battlesnake_log_writer"]
    request_logs = app.extensions["battlesnake_request_logs"]
    payload = state.model_dump(mode="json")

    assert client.post("/start", json=payload).status_code == 200
    assert writer.flush()

    entered = threading.Event()
    release = threading.Event()
    original_append = request_logs._append_line

    def slow_append(line: str) -> None:
        event = json.loads(line)
        if event.get("endpoint") == "/move" and event.get("phase") == "received":
            entered.set()
            release.wait(timeout=1.0)
        original_append(line)

    monkeypatch.setattr(request_logs, "_append_line", slow_append)
    started = time.perf_counter()
    response = client.post("/move", json=payload)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert response.status_code == 200
    assert response.get_json() == {"move": "right"}
    assert entered.wait(timeout=0.5)
    assert elapsed_ms < 250.0

    release.set()
    assert writer.close(timeout_seconds=2.0)
    assert writer.stats()["dropped"] == 0
    assert writer.stats()["errors"] == 0
    events = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    move_phases = [
        event["phase"] for event in events if event.get("endpoint") == "/move"
    ]
    assert move_phases == ["received", "completed"]


def test_contended_game_log_lock_does_not_delay_move_response(tmp_path) -> None:
    state = _blackout_request()
    app = start_server(_Agent(), 0, run=False, log_dir=str(tmp_path))
    client = app.test_client()
    writer = app.extensions["battlesnake_log_writer"]
    game_logs = app.extensions["battlesnake_game_logs"]
    payload = state.model_dump(mode="json")

    assert client.post("/start", json=payload).status_code == 200
    assert writer.flush()

    held = threading.Event()
    release = threading.Event()

    def hold_game_log_lock() -> None:
        with game_logs._lock:
            held.set()
            release.wait(timeout=1.0)

    holder = threading.Thread(target=hold_game_log_lock)
    holder.start()
    assert held.wait(timeout=0.5)
    started = time.perf_counter()
    response = client.post("/move", json=payload)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert response.status_code == 200
    assert response.get_json() == {"move": "right"}
    assert elapsed_ms < 250.0
    assert game_logs.stats()["move_lock_contention_drops"] == 1

    release.set()
    holder.join(timeout=1.0)
    assert not holder.is_alive()
    assert writer.close(timeout_seconds=2.0)
