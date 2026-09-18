import copy
import json
import logging
import os
import queue
import re
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask
from flask import request

from battlesnake_types import GameState, BaseAgent


logger = logging.getLogger(__name__)


class AsyncLogWriter:
    """Single-writer, bounded queue for fail-open diagnostic filesystem I/O.

    Request handlers only render their small records and enqueue a callable.
    They never wait for the filesystem or for another log write to finish.  The
    daemon worker preserves FIFO ordering across request and game logs.  A full
    queue drops diagnostics instead of risking a competition timeout; callers
    that need durable, complete logs (the HTTP benchmark) explicitly close the
    writer and validate its counters before reading the files.
    """

    def __init__(self, max_pending: int = 8192):
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self._queue = queue.Queue(maxsize=max_pending)
        self._state_lock = threading.Lock()
        self._accepting = True
        self._dropped = 0
        self._errors = 0
        self._thread = threading.Thread(
            target=self._run,
            name="battlesnake-log-writer",
            daemon=True,
        )
        self._thread.start()

    def submit(self, operation, *args) -> bool:
        """Enqueue one write without waiting; return whether it was accepted."""

        with self._state_lock:
            if not self._accepting:
                self._dropped += 1
                return False
            try:
                self._queue.put_nowait((operation, args))
            except queue.Full:
                self._dropped += 1
                return False
        return True

    def flush(self, timeout_seconds: float = 5.0) -> bool:
        """Wait at most ``timeout_seconds`` for all writes queued so far."""

        timeout_seconds = max(0.0, float(timeout_seconds))
        if not self._thread.is_alive():
            return self._queue.empty()
        barrier = threading.Event()
        deadline = time.monotonic() + timeout_seconds
        try:
            self._queue.put((barrier.set, ()), timeout=timeout_seconds)
        except queue.Full:
            return False
        remaining = max(0.0, deadline - time.monotonic())
        return barrier.wait(remaining)

    def close(self, timeout_seconds: float = 5.0) -> bool:
        """Stop accepting writes and drain/stop the daemon within a bound."""

        with self._state_lock:
            self._accepting = False
        self._thread.join(timeout=max(0.0, float(timeout_seconds)))
        return not self._thread.is_alive() and self._queue.empty()

    def stats(self) -> dict[str, int | bool]:
        with self._state_lock:
            return {
                "pending": self._queue.qsize(),
                "dropped": self._dropped,
                "errors": self._errors,
                "worker_alive": self._thread.is_alive(),
            }

    def _run(self) -> None:
        while True:
            try:
                operation, args = self._queue.get(timeout=0.1)
            except queue.Empty:
                with self._state_lock:
                    if not self._accepting:
                        return
                continue
            try:
                operation(*args)
            except Exception:
                with self._state_lock:
                    self._errors += 1
                try:
                    logger.exception("Asynchronous Battlesnake log write failed")
                except Exception:
                    # A broken logging handler must not terminate the only
                    # writer thread; counters still expose the failed task.
                    pass
            finally:
                self._queue.task_done()


class RequestLogStore:
    """Append-only request log for diagnosing dropped/slow requests.

    The per-game JSON logs only contain completed moves. This JSONL log records
    request arrival before any game parsing or agent work happens, so a missing
    completed/failed entry points at a process/network interruption instead of a
    slow policy call.
    """

    def __init__(self, log_dir: Path, writer: AsyncLogWriter | None = None):
        self.log_dir = log_dir
        self.path = self.log_dir / "requests.jsonl"
        self._writer = writer or AsyncLogWriter()
        self._owns_writer = writer is None

    def received(self, endpoint: str, payload: dict | None) -> str:
        request_id = uuid.uuid4().hex
        self._write(
            {
                "request_id": request_id,
                "endpoint": endpoint,
                "phase": "received",
                **self._payload_meta(payload),
            }
        )
        return request_id

    def completed(
        self,
        request_id: str,
        endpoint: str,
        payload: dict | None,
        latency_seconds: float,
        result: dict | None = None,
    ) -> None:
        event = {
            "request_id": request_id,
            "endpoint": endpoint,
            "phase": "completed",
            "latency_ms": round(latency_seconds * 1000, 3),
            **self._payload_meta(payload),
        }
        if result is not None:
            event["result"] = result
        self._write(event)

    def failed(
        self,
        request_id: str,
        endpoint: str,
        payload: dict | None,
        latency_seconds: float,
        exc: Exception,
    ) -> None:
        self._write(
            {
                "request_id": request_id,
                "endpoint": endpoint,
                "phase": "failed",
                "latency_ms": round(latency_seconds * 1000, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                **self._payload_meta(payload),
            }
        )

    def _write(self, event: dict) -> None:
        event = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            **event,
        }
        line = json.dumps(event, sort_keys=True, default=str)
        self._writer.submit(self._append_line, line)

    def _append_line(self, line: str) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")

    def flush(self, timeout_seconds: float = 5.0) -> bool:
        return self._writer.flush(timeout_seconds)

    def close(self, timeout_seconds: float = 5.0) -> bool:
        if not self._owns_writer:
            return True
        return self._writer.close(timeout_seconds)

    def _payload_meta(self, payload: dict | None) -> dict:
        if not isinstance(payload, dict):
            return {"payload_type": type(payload).__name__}

        game = payload.get("game") or {}
        board = payload.get("board") or {}
        you = payload.get("you") or {}
        snakes = board.get("snakes") or []

        return {
            "game_id": game.get("id"),
            "turn": payload.get("turn"),
            "you_id": you.get("id"),
            "timeout": game.get("timeout"),
            "snake_count": len(snakes) if isinstance(snakes, list) else None,
        }


class GameLogStore:
    def __init__(
        self,
        agent: BaseAgent,
        log_dir: str | None = None,
        writer: AsyncLogWriter | None = None,
    ):
        self.agent = agent
        self.log_dir = Path(log_dir or os.getenv("BATTLESNAKE_LOG_DIR", "game_logs"))
        self.active: dict[str, tuple[Path, dict]] = {}
        self.completed: set[str] = set()
        self._lock = threading.RLock()
        self._stats_lock = threading.Lock()
        self._contention_drops = 0
        self._writer = writer or AsyncLogWriter()
        self._owns_writer = writer is None

    def log_start(self, game_state: GameState) -> None:
        with self._lock:
            self.completed.discard(game_state.game.id)
            path, entry = self._entry_for(game_state)
            entry["started_at"] = self._now()
            entry["start_state"] = self._state_dump(game_state)
            snapshot = copy.deepcopy(entry)
        self._writer.submit(self._write_entry, path, snapshot)

    def log_move(self, game_state: GameState, move, latency_seconds: float) -> bool:
        if not self._lock.acquire(blocking=False):
            with self._stats_lock:
                self._contention_drops += 1
            return False
        try:
            path, entry = self._entry_for(game_state)
            record = {
                "turn": game_state.turn,
                "received_at": self._now(),
                "latency_ms": round(latency_seconds * 1000, 3),
                "move": move.model_dump(mode="json"),
                "state": self._state_dump(game_state),
            }
            entry["moves"].append(record)
            # Re-rendering the complete growing match here eventually costs
            # longer than the 500 ms /move deadline.  Keep the in-memory final
            # document, but make the request-path write append-only and O(size
            # of this turn).  The sidecar also preserves completed moves if the
            # process exits before /end compaction.
        finally:
            self._lock.release()
        return self._writer.submit(self._append_move, path, record)

    def log_end(self, game_state: GameState) -> bool:
        with self._lock:
            game_id = game_state.game.id
            if game_id not in self.active:
                # Do not create confusing 0-move summary files for delayed
                # duplicate /end callbacks. The request log still records them.
                return False

            path, entry = self._entry_for(game_state)
            entry["ended_at"] = self._now()
            entry["end_state"] = self._state_dump(game_state)
            entry["result"] = self._result_summary(game_state)
            self.active.pop(game_id, None)
            self.completed.add(game_id)
            snapshot = copy.deepcopy(entry)
        # FIFO ordering guarantees all accepted move sidecars precede the final
        # compacted document without creating one thread per completed game.
        return self._writer.submit(self._write_entry, path, snapshot)

    def _entry_for(self, game_state: GameState) -> tuple[Path, dict]:
        game_id = game_state.game.id
        if game_id not in self.active:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", game_id)[:80] or "game"
            path = self.log_dir / f"{timestamp}_{safe_id}.json"
            agent_info = {
                "name": self.agent.get_name(),
                "author": self.agent.get_author(),
                "color": self.agent.get_color(),
            }
            diagnostics_fn = getattr(self.agent, "get_diagnostics", None)
            if callable(diagnostics_fn):
                diagnostics = diagnostics_fn()
                if diagnostics:
                    agent_info["diagnostics"] = diagnostics
            self.active[game_id] = (
                path,
                {
                    "schema_version": 1,
                    "game_id": game_id,
                    "agent": agent_info,
                    "started_at": None,
                    "ended_at": None,
                    "start_state": None,
                    "moves": [],
                    "end_state": None,
                    "result": None,
                },
            )
        return self.active[game_id]

    def _write(self, game_state: GameState) -> None:
        path, entry = self._entry_for(game_state)
        self._write_entry(path, entry)

    def _write_entry(self, path: Path, entry: dict) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(entry, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)

    def _write_entry_safely(self, path: Path, entry: dict) -> None:
        try:
            self._write_entry(path, entry)
        except Exception:
            logger.exception("Failed compacting game log %s", path)

    def flush(self, timeout_seconds: float = 5.0) -> bool:
        return self._writer.flush(timeout_seconds)

    def close(self, timeout_seconds: float = 5.0) -> bool:
        if not self._owns_writer:
            return True
        return self._writer.close(timeout_seconds)

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return {"move_lock_contention_drops": self._contention_drops}

    def _append_move(self, path: Path, record: dict) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        sidecar = path.with_suffix(".moves.jsonl")
        with sidecar.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            handle.write("\n")

    def _state_dump(self, game_state: GameState) -> dict:
        return game_state.model_dump(mode="json")

    def _result_summary(self, game_state: GameState) -> dict:
        snakes = {snake.id: snake for snake in game_state.board.snakes}
        snakes.setdefault(game_state.you.id, game_state.you)

        alive_ids = [
            snake.id
            for snake in game_state.board.snakes
            if snake.elimination_event is None and snake.head is not None
        ]
        you_alive = game_state.you.id in alive_ids and game_state.you.elimination_event is None
        if you_alive and len(alive_ids) == 1:
            outcome = "win"
        elif you_alive:
            outcome = "draw"
        else:
            outcome = "loss"

        return {
            "turn": game_state.turn,
            "outcome": outcome,
            "you": game_state.you.id,
            "alive_snakes": alive_ids,
            "snakes": [
                {
                    "id": snake.id,
                    "name": snake.name,
                    "length": snake.length,
                    "health": snake.health,
                    "elimination_event": (
                        snake.elimination_event.model_dump(mode="json")
                        if snake.elimination_event is not None
                        else None
                    ),
                }
                for snake in snakes.values()
            ],
        }

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()


def start_server(
    agent: BaseAgent,
    port,
    *,
    run: bool = True,
    log_dir: str | None = None,
):
    if port is None:
        raise ValueError('please select your port')

    app = Flask("Battlesnake")
    log_writer = AsyncLogWriter()
    game_logs = GameLogStore(agent, log_dir=log_dir, writer=log_writer)
    request_logs = RequestLogStore(game_logs.log_dir, writer=log_writer)
    lifecycle_lock = threading.RLock()
    started_sessions: set[tuple[str, str]] = set()
    app.extensions["battlesnake_game_logs"] = game_logs
    app.extensions["battlesnake_request_logs"] = request_logs
    app.extensions["battlesnake_log_writer"] = log_writer

    def _request_json() -> dict:
        data = request.get_json(silent=True)
        if data is None:
            raise ValueError("Request JSON body is missing or invalid")
        return data

    @app.get("/")
    def on_info():
        # TIP: If you open your Battlesnake URL in browser you should see this data
        data = {
            "author": agent.get_author(),
            "color": agent.get_color(),
        }

        # filter None values
        data = {k: v for k, v in data.items() if v is not None}

        if 'kilab' in request.args:
            name = agent.get_name()
            data['name'] = name

        return data

    @app.post("/start")
    def on_start():
        """start is called when your Battlesnake begins a game"""
        start = time.perf_counter()
        data = request.get_json(silent=True)
        request_id = request_logs.received("/start", data)
        try:
            data = data if data is not None else _request_json()
            game_state = GameState(**data)
            session_key = (game_state.game.id, game_state.you.id)
            with lifecycle_lock:
                duplicate = session_key in started_sessions
                if not duplicate:
                    agent.start(game_state)
                    started_sessions.add(session_key)
            if not duplicate:
                try:
                    game_logs.log_start(game_state)
                except Exception:
                    log_writer.submit(
                        logger.error,
                        "Failed preparing game-start log; continuing\n%s",
                        traceback.format_exc(),
                    )
            elapsed = time.perf_counter() - start
            request_logs.completed(
                request_id,
                "/start",
                data,
                elapsed,
                {"duplicate": duplicate},
            )
            log_writer.submit(
                logger.info,
                "START game=%s duplicate=%s",
                game_state.game.id,
                duplicate,
            )
            return "ok"
        except Exception as exc:
            elapsed = time.perf_counter() - start
            request_logs.failed(request_id, "/start", data, elapsed, exc)
            log_writer.submit(
                logger.error,
                "Failed handling /start request %s\n%s",
                request_id,
                traceback.format_exc(),
            )
            raise

    @app.post("/move")
    def on_move():
        """move is called on every turn and returns your next move"""
        start = time.perf_counter()
        data = request.get_json(silent=True)
        request_id = request_logs.received("/move", data)
        try:
            data = data if data is not None else _request_json()
            game_state = GameState(**data)
            move = agent.move(game_state)
            agent_elapsed = time.perf_counter() - start
            try:
                game_logs.log_move(game_state, move, agent_elapsed)
            except Exception:
                log_writer.submit(
                    logger.error,
                    "Failed preparing game-move log; returning move\n%s",
                    traceback.format_exc(),
                )
            move_data = move.model_dump()
            handler_elapsed = time.perf_counter() - start
            request_logs.completed(
                request_id,
                "/move",
                data,
                handler_elapsed,
                {
                    "move": move.model_dump(mode="json"),
                    "agent_latency_ms": round(agent_elapsed * 1000, 3),
                },
            )

            elapsed = time.perf_counter() - start
            if logger.isEnabledFor(logging.DEBUG):
                log_writer.submit(
                    logger.debug,
                    "MOVE game=%s turn=%s move=%s elapsed=%.6f",
                    game_state.game.id,
                    game_state.turn,
                    move,
                    elapsed,
                )
            return move_data
        except Exception as exc:
            elapsed = time.perf_counter() - start
            request_logs.failed(request_id, "/move", data, elapsed, exc)
            log_writer.submit(
                logger.error,
                "Failed handling /move request %s\n%s",
                request_id,
                traceback.format_exc(),
            )
            raise

    @app.post("/end")
    def on_end():
        """end is called when your Battlesnake finishes a game"""
        start = time.perf_counter()
        data = request.get_json(silent=True)
        request_id = request_logs.received("/end", data)
        try:
            data = data if data is not None else _request_json()
            game_state = GameState(**data)
            session_key = (game_state.game.id, game_state.you.id)
            with lifecycle_lock:
                duplicate = session_key not in started_sessions
                if not duplicate:
                    agent.end(game_state)
                    started_sessions.discard(session_key)
            try:
                summary_logged = game_logs.log_end(game_state)
            except Exception:
                log_writer.submit(
                    logger.error,
                    "Failed preparing game-end log; continuing\n%s",
                    traceback.format_exc(),
                )
                summary_logged = False
            elapsed = time.perf_counter() - start
            request_logs.completed(
                request_id,
                "/end",
                data,
                elapsed,
                {"summary_logged": summary_logged, "duplicate": duplicate},
            )
            log_writer.submit(
                logger.info,
                "END game=%s duplicate=%s",
                game_state.game.id,
                duplicate,
            )
            return "ok"
        except Exception as exc:
            elapsed = time.perf_counter() - start
            request_logs.failed(request_id, "/end", data, elapsed, exc)
            log_writer.submit(
                logger.error,
                "Failed handling /end request %s\n%s",
                request_id,
                traceback.format_exc(),
            )
            raise

    if not run:
        # Useful for an end-to-end Flask test client without opening a socket.
        return app

    host = "0.0.0.0"

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    print(f"\nRunning Battlesnake at http://{host}:{port}")
    print(f"Logging games to {game_logs.log_dir.resolve()}")
    try:
        app.run(host=host, port=port)
    finally:
        if not log_writer.close(timeout_seconds=5.0):
            logger.error("Timed out draining asynchronous Battlesnake logs")
