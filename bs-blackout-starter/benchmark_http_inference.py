"""Measure the complete local HTTP path of the production PPO-MCTS agent.

Unlike ``evaluate_inference.py``, this benchmark deliberately includes Flask,
JSON/Pydantic parsing, request and game logging, response serialization, and a
real loopback HTTP connection.  It is a latency check, not a strength eval.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import tempfile
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import hisss
import numpy as np
import torch
from hisss.cpp.lib import CPP_LIB
from werkzeug.serving import make_server

import battlesnake_server as battlesnake_server_module
from battlesnake_server import start_server
from obs_config import make_game_config
from ppo_mcts import PPOMCTSAgent


VALID_MOVES = {"up", "down", "left", "right"}
COMPACTION_TIMEOUT_SECONDS = 5.0
DEFAULT_SEED = 2_606_811
DEFAULT_SLA_MS = 500.0
MAX_NATIVE_SEED = 2**31 - 1
REPORT_SCHEMA_VERSION = 4


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    CPP_LIB.lib.set_seed(seed)


def _derived_seed(seed: int, stream: int) -> int:
    return (seed * 1_000_003 + stream * 97_409 + 17) & MAX_NATIVE_SEED


def _request_state(game_id: str, seed: int) -> dict:
    # Hisss owns initial placement and food randomness in its native library.
    # Seed immediately before construction so unrelated library calls cannot
    # perturb the workload.
    CPP_LIB.lib.set_seed(seed)
    cfg = make_game_config()
    env = hisss.BattleSnakeGame(cfg)
    try:
        state = json.loads(hisss.to_battlesnake_json(env, 0))
    finally:
        env.close()
    state["game"]["id"] = game_id
    # The competition contract grants 500 ms even if a local Hisss build uses
    # a different metadata default.
    state["game"]["timeout"] = 500
    return state


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_sha256() -> dict[str, str]:
    benchmark_path = Path(__file__).resolve()
    server_file = getattr(battlesnake_server_module, "__file__", None)
    if not server_file:
        raise RuntimeError("cannot resolve battlesnake_server source path")
    return {
        "benchmark_http_inference.py": _sha256_file(benchmark_path),
        "battlesnake_server.py": _sha256_file(Path(server_file).resolve()),
    }


def _post_json(url: str, payload: dict, timeout_seconds: float) -> dict | str:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {body!r}")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body.decode("utf-8")


def _latency_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "over_450ms": 0,
            "over_500ms": 0,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        "over_450ms": int(np.count_nonzero(array > 450.0)),
        "over_500ms": int(np.count_nonzero(array > 500.0)),
    }


def _sla_summary(
    values: Sequence[float], *, threshold_ms: float, enforced: bool
) -> dict[str, object]:
    violations = sum(float(value) > threshold_ms for value in values)
    return {
        "metric": "client_round_trip_ms",
        "threshold_ms": threshold_ms,
        "enforced": enforced,
        "passed": violations == 0,
        "violations": violations,
    }


def _combined_sla_summary(
    primary_values: Sequence[float],
    retry_values: Sequence[float],
    *,
    threshold_ms: float,
    enforced: bool,
) -> dict[str, object]:
    """Gate every measured /move while retaining per-workload attribution."""

    primary = _sla_summary(
        primary_values,
        threshold_ms=threshold_ms,
        enforced=enforced,
    )
    retry_probe = _sla_summary(
        retry_values,
        threshold_ms=threshold_ms,
        enforced=enforced,
    )
    combined_values = [*primary_values, *retry_values]
    combined = _sla_summary(
        combined_values,
        threshold_ms=threshold_ms,
        enforced=enforced,
    )
    return {
        **combined,
        "scope": "primary-and-retry-probe-/move-requests",
        "samples": len(combined_values),
        "breakdown": {
            "primary": {**primary, "samples": len(primary_values)},
            "retry_probe": {**retry_probe, "samples": len(retry_values)},
        },
    }


def _request_events(log_dir: Path) -> list[dict]:
    path = log_dir / "requests.jsonl"
    if not path.is_file():
        raise RuntimeError(f"request log is missing: {path}")
    events = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid request-log JSON on line {line_number}"
            ) from exc
        if not isinstance(event, dict):
            raise RuntimeError(f"request-log line {line_number} is not an object")
        events.append(event)
    return events


def _validate_request_pairs(events: Sequence[dict]) -> None:
    """Require one received and one completed record for every request."""

    if not events:
        raise RuntimeError("request log is empty")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        request_id = event.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError("request-log event has no request_id")
        if event.get("phase") == "failed":
            raise RuntimeError(
                f"request {request_id} failed: {event.get('error_type')}: "
                f"{event.get('error')}"
            )
        grouped[request_id].append(event)

    for request_id, records in grouped.items():
        phases = Counter(record.get("phase") for record in records)
        if phases != Counter({"received": 1, "completed": 1}):
            raise RuntimeError(
                f"request {request_id} has unmatched log phases: {dict(phases)}"
            )
        received = next(record for record in records if record["phase"] == "received")
        completed = next(
            record for record in records if record["phase"] == "completed"
        )
        for field in ("endpoint", "game_id", "turn", "you_id"):
            if received.get(field) != completed.get(field):
                raise RuntimeError(
                    f"request {request_id} changed {field!r} between log phases"
                )


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _completed_move_events(
    events: Sequence[dict],
    *,
    game_id: str,
    expected_turns: Sequence[int],
) -> list[dict]:
    completed = [
        event
        for event in events
        if event.get("endpoint") == "/move"
        and event.get("phase") == "completed"
        and event.get("game_id") == game_id
    ]
    expected = Counter(int(turn) for turn in expected_turns)
    actual = Counter(event.get("turn") for event in completed)
    if len(completed) != len(expected_turns) or actual != expected:
        raise RuntimeError(
            f"game {game_id!r} completed move log mismatch: "
            f"expected turns {dict(expected)}, got {dict(actual)}"
        )
    for event in completed:
        if not _finite_nonnegative(event.get("latency_ms")):
            raise RuntimeError("completed /move has invalid handler latency")
        result = event.get("result")
        if not isinstance(result, dict) or not _finite_nonnegative(
            result.get("agent_latency_ms")
        ):
            raise RuntimeError("completed /move has no valid agent latency")
        logged_move = result.get("move")
        if (
            not isinstance(logged_move, dict)
            or logged_move.get("move") not in VALID_MOVES
        ):
            raise RuntimeError("completed /move has no valid logged response")
    return completed


def _valid_move_response(result: object) -> bool:
    return isinstance(result, dict) and result.get("move") in VALID_MOVES


def _wait_for_compaction(
    log_dir: Path,
    game_id: str,
    *,
    expected_moves: int | None,
    timeout_seconds: float = COMPACTION_TIMEOUT_SECONDS,
) -> Path:
    deadline = time.perf_counter() + timeout_seconds
    last_error: Exception | None = None
    while time.perf_counter() < deadline:
        for path in log_dir.glob("*.json"):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                last_error = exc
                continue
            if document.get("game_id") != game_id or not document.get("ended_at"):
                continue
            moves = document.get("moves")
            if expected_moves is not None and (
                not isinstance(moves, list) or len(moves) != expected_moves
            ):
                raise RuntimeError(
                    f"compacted game {game_id!r} has "
                    f"{len(moves) if isinstance(moves, list) else 'invalid'} moves; "
                    f"expected {expected_moves}"
                )
            if expected_moves is not None:
                sidecar = path.with_suffix(".moves.jsonl")
                if not sidecar.is_file():
                    raise RuntimeError(
                        f"game-log move sidecar is missing for {game_id!r}"
                    )
                lines = sidecar.read_text(encoding="utf-8").splitlines()
                if len(lines) != expected_moves:
                    raise RuntimeError(
                        f"game-log sidecar for {game_id!r} has {len(lines)} "
                        f"moves; expected {expected_moves}"
                    )
                try:
                    for line in lines:
                        json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"game-log sidecar for {game_id!r} is invalid"
                    ) from exc
            return path
        time.sleep(0.005)
    detail = f": {last_error}" if last_error is not None else ""
    raise RuntimeError(f"game-log compaction timed out for {game_id!r}{detail}")


def _runtime_metadata() -> dict[str, object]:
    try:
        hisss_version = importlib.metadata.version("hisss")
    except importlib.metadata.PackageNotFoundError:
        hisss_version = str(getattr(hisss, "__version__", "unknown"))
    cuda_available = bool(torch.cuda.is_available())
    devices: list[str] = []
    if cuda_available:
        try:
            devices = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
        except (AssertionError, RuntimeError):
            devices = ["unavailable"]
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "hisss": hisss_version,
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "cuda_available": cuda_available,
        "cuda_devices": devices,
    }


def _end_game(
    *,
    base_url: str,
    state: dict,
    log_dir: Path,
    expected_moves: int,
    timeout_seconds: float,
) -> None:
    _post_json(f"{base_url}/end", state, timeout_seconds)
    _wait_for_compaction(
        log_dir,
        state["game"]["id"],
        expected_moves=expected_moves,
    )


def _retry_probe(
    *,
    agent: PPOMCTSAgent,
    base_url: str,
    state: dict,
    log_dir: Path,
    timeout_seconds: float,
) -> tuple[dict[str, object], list[dict]]:
    """Exercise cached same-turn retry semantics outside primary samples."""

    _post_json(f"{base_url}/start", state, timeout_seconds)
    state["turn"] = 0

    first_started = time.perf_counter()
    first = _post_json(f"{base_url}/move", state, timeout_seconds)
    first_ms = (time.perf_counter() - first_started) * 1_000.0
    if not _valid_move_response(first):
        raise RuntimeError(f"invalid retry-probe response: {first!r}")
    first_stats = copy.deepcopy(agent.get_last_search_stats())
    if not isinstance(first_stats, dict):
        raise RuntimeError("retry probe produced no search diagnostics")

    # A delayed duplicate /start must not reset the recurrent/search session.
    # Keep it between the original and duplicate /move so a reset invalidates
    # the same-turn cache and is observable to this probe.
    duplicate_start = _post_json(f"{base_url}/start", state, timeout_seconds)
    if duplicate_start != "ok":
        raise RuntimeError(
            f"duplicate retry-probe /start returned {duplicate_start!r}"
        )

    retry_started = time.perf_counter()
    retry = _post_json(f"{base_url}/move", state, timeout_seconds)
    retry_ms = (time.perf_counter() - retry_started) * 1_000.0
    retry_stats = copy.deepcopy(agent.get_last_search_stats())
    if retry != first:
        raise RuntimeError("same-turn retry returned a different move")
    if retry_stats != first_stats:
        raise RuntimeError("same-turn retry changed search/LSTM diagnostics")

    state["turn"] = 1
    next_started = time.perf_counter()
    next_result = _post_json(f"{base_url}/move", state, timeout_seconds)
    next_ms = (time.perf_counter() - next_started) * 1_000.0
    if not _valid_move_response(next_result):
        raise RuntimeError(f"post-retry next turn failed: {next_result!r}")
    next_stats = agent.get_last_search_stats()
    if not isinstance(next_stats, dict):
        raise RuntimeError("post-retry next turn produced no search diagnostics")
    if next_stats.get("selected") != next_result["move"]:
        raise RuntimeError("post-retry response disagrees with search diagnostics")

    _end_game(
        base_url=base_url,
        state=state,
        log_dir=log_dir,
        expected_moves=3,
        timeout_seconds=timeout_seconds,
    )
    client_samples = [first_ms, retry_ms, next_ms]
    return (
        {
            "duplicate_start_preserved_retry_cache": True,
            "same_turn_response_identical": True,
            "same_turn_stats_identical": True,
            "next_turn_succeeded": True,
            "client_round_trip_ms": {
                "initial": first_ms,
                "cached_retry": retry_ms,
                "next_turn": next_ms,
            },
            "client_round_trip_summary_ms": _latency_summary(client_samples),
            "client_round_trip_samples_ms": client_samples,
        },
        [first, retry, next_result],
    )


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    seed = int(args.seed)
    source_sha256 = _source_sha256()
    _seed_everything(seed)
    kwargs: dict[str, object] = {"seed": seed}
    if args.time_budget_ms is not None:
        kwargs["time_budget_ms"] = args.time_budget_ms
    if args.max_iterations is not None:
        kwargs["max_iterations"] = args.max_iterations
    if args.min_iterations is not None:
        kwargs["min_iterations"] = args.min_iterations
    agent = (
        PPOMCTSAgent(**kwargs)
        if args.model is None
        else PPOMCTSAgent(args.model, **kwargs)
    )

    primary_seed = _derived_seed(seed, 0)
    retry_seed = _derived_seed(seed, 1)
    game_id = f"http-benchmark-seed-{seed}"
    retry_game_id = f"http-retry-probe-seed-{seed}"
    state = _request_state(game_id, primary_seed)
    retry_state = _request_state(retry_game_id, retry_seed)
    workload = {
        "generator": "seeded-hisss-initial-state-v1",
        "rng_protocol": "python-numpy-torch-hisss-cpp-and-agent-seeded-v1",
        "seed": seed,
        "primary": {
            "native_seed": primary_seed,
            "game_id": game_id,
            "turns": list(range(args.moves)),
            "initial_request_sha256": _sha256_json(state),
        },
        "retry_probe": {
            "native_seed": retry_seed,
            "game_id": retry_game_id,
            "turns": [0, 0, 1],
            "duplicate_start_after_first_move": True,
            "initial_request_sha256": _sha256_json(retry_state),
        },
    }
    # Leave all process-level generators at the public seed before inference.
    _seed_everything(seed)
    client_ms: list[float] = []
    primary_responses: list[dict] = []
    retry_report: dict[str, object]
    retry_responses: list[dict]
    retry_client_ms: list[float]
    with tempfile.TemporaryDirectory(prefix="ppo-mcts-http-") as tmp:
        log_dir = Path(tmp)
        app = start_server(agent, 0, run=False, log_dir=str(log_dir))
        log_writer = app.extensions["battlesnake_log_writer"]
        game_logs = app.extensions["battlesnake_game_logs"]
        werkzeug_logger = logging.getLogger("werkzeug")
        previous_werkzeug_level = werkzeug_logger.level
        werkzeug_logger.setLevel(logging.ERROR)
        try:
            server = make_server("127.0.0.1", 0, app, threaded=True)
        except Exception:
            werkzeug_logger.setLevel(previous_werkzeug_level)
            log_writer.close(timeout_seconds=COMPACTION_TIMEOUT_SECONDS)
            raise
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        try:
            thread.start()
        except Exception:
            server.server_close()
            werkzeug_logger.setLevel(previous_werkzeug_level)
            log_writer.close(timeout_seconds=COMPACTION_TIMEOUT_SECONDS)
            raise
        base_url = f"http://127.0.0.1:{server.server_port}"
        active_games: dict[str, dict] = {}
        log_writer_closed = False
        try:
            active_games[game_id] = state
            _post_json(f"{base_url}/start", state, args.http_timeout_seconds)
            for turn in range(args.moves):
                state["turn"] = turn
                started = time.perf_counter()
                result = _post_json(
                    f"{base_url}/move", state, args.http_timeout_seconds
                )
                client_ms.append((time.perf_counter() - started) * 1_000.0)
                if not _valid_move_response(result):
                    raise RuntimeError(f"invalid move response: {result!r}")
                primary_responses.append(result)
            _end_game(
                base_url=base_url,
                state=state,
                log_dir=log_dir,
                expected_moves=args.moves,
                timeout_seconds=args.http_timeout_seconds,
            )
            active_games.pop(game_id, None)

            # Keep retries out of the primary latency distribution and in a
            # distinct game, so the exact-N assertion cannot count them.
            active_games[retry_game_id] = retry_state
            retry_report, retry_responses = _retry_probe(
                agent=agent,
                base_url=base_url,
                state=retry_state,
                log_dir=log_dir,
                timeout_seconds=args.http_timeout_seconds,
            )
            retry_samples = retry_report.get("client_round_trip_samples_ms")
            if (
                not isinstance(retry_samples, list)
                or len(retry_samples) != 3
                or any(not _finite_nonnegative(value) for value in retry_samples)
            ):
                raise RuntimeError(
                    "retry probe did not expose three valid client latency samples"
                )
            retry_client_ms = [
                float(value)
                for value in retry_samples
            ]
            active_games.pop(retry_game_id, None)
        finally:
            # On an exceptional request path, release recurrent sessions and
            # give async compaction a bounded chance to finish before tmpdir
            # cleanup.  The triggering exception remains authoritative.
            for active_state in list(active_games.values()):
                try:
                    _post_json(
                        f"{base_url}/end",
                        active_state,
                        args.http_timeout_seconds,
                    )
                    _wait_for_compaction(
                        log_dir,
                        active_state["game"]["id"],
                        expected_moves=None,
                    )
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "best-effort benchmark /end cleanup failed: %s", exc
                    )
            try:
                server.shutdown()
            finally:
                server.server_close()
                thread.join(timeout=5.0)
                werkzeug_logger.setLevel(previous_werkzeug_level)
                log_writer_closed = log_writer.close(
                    timeout_seconds=COMPACTION_TIMEOUT_SECONDS
                )
                if thread.is_alive():
                    raise RuntimeError("benchmark HTTP server thread did not stop")

        if not log_writer_closed:
            raise RuntimeError("asynchronous benchmark log writer did not stop")
        log_writer_stats = log_writer.stats()
        if log_writer_stats["dropped"] or log_writer_stats["errors"]:
            raise RuntimeError(
                "asynchronous benchmark logging lost integrity: "
                f"{log_writer_stats}"
            )
        game_log_stats = game_logs.stats()
        if game_log_stats["move_lock_contention_drops"]:
            raise RuntimeError(
                f"benchmark game logging lost integrity: {game_log_stats}"
            )

        events = _request_events(log_dir)
        _validate_request_pairs(events)
        # primary start/moves/end + retry start/duplicate-start/3 moves/end
        expected_requests = args.moves + 8
        if len(events) != expected_requests * 2:
            raise RuntimeError(
                f"request-log event count mismatch: expected "
                f"{expected_requests * 2}, got {len(events)}"
            )
        for current_game_id in (game_id, retry_game_id):
            for endpoint in ("/start", "/end"):
                completed_lifecycle = [
                    event
                    for event in events
                    if event.get("game_id") == current_game_id
                    and event.get("endpoint") == endpoint
                    and event.get("phase") == "completed"
                ]
                expected_lifecycle = (
                    2
                    if current_game_id == retry_game_id and endpoint == "/start"
                    else 1
                )
                if len(completed_lifecycle) != expected_lifecycle:
                    raise RuntimeError(
                        f"game {current_game_id!r} has "
                        f"{len(completed_lifecycle)} completed {endpoint} events"
                    )
        move_game_ids = {
            event.get("game_id")
            for event in events
            if event.get("endpoint") == "/move"
        }
        if move_game_ids != {game_id, retry_game_id}:
            raise RuntimeError(
                f"unexpected /move game IDs in request log: {move_game_ids}"
            )
        primary_events = _completed_move_events(
            events,
            game_id=game_id,
            expected_turns=range(args.moves),
        )
        if [event["turn"] for event in primary_events] != list(range(args.moves)):
            raise RuntimeError("primary move logs are out of order")
        retry_events = _completed_move_events(
            events,
            game_id=retry_game_id,
            expected_turns=(0, 0, 1),
        )
        if [event["turn"] for event in retry_events] != [0, 0, 1]:
            raise RuntimeError("retry-probe move logs are out of order")
        primary_by_turn = {event["turn"]: event for event in primary_events}
        for turn, response in enumerate(primary_responses):
            logged = primary_by_turn[turn]["result"]["move"]["move"]
            if logged != response["move"]:
                raise RuntimeError(f"logged move disagrees at primary turn {turn}")
        for event, response in zip(retry_events, retry_responses, strict=True):
            if event["result"]["move"]["move"] != response["move"]:
                raise RuntimeError("logged move disagrees in retry probe")

    pre_completion_ms = [float(event["latency_ms"]) for event in primary_events]
    agent_ms = [
        float(event["result"]["agent_latency_ms"]) for event in primary_events
    ]
    client_summary = _latency_summary(client_ms)
    all_client_ms = [*client_ms, *retry_client_ms]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "endpoint": "loopback-http-/move",
        "model": str(agent.model_path) if agent.model_path is not None else None,
        "agent_diagnostics": agent.get_diagnostics(),
        "runtime": _runtime_metadata(),
        "source_sha256": source_sha256,
        "workload": workload,
        "moves": args.moves,
        "sla_metric": "client_round_trip_ms",
        "sla": _combined_sla_summary(
            client_ms,
            retry_client_ms,
            threshold_ms=args.sla_ms,
            enforced=bool(args.fail_on_sla),
        ),
        "latency_semantics": {
            "client_round_trip_ms": (
                "Primary-workload monotonic client wall time from JSON "
                "serialization through loopback HTTP, Flask/Pydantic, agent "
                "execution, bounded log enqueue, response serialization and "
                "body decode; asynchronous filesystem writes are deliberately "
                "outside the move deadline."
            ),
            "all_move_client_round_trip_ms": (
                "The same client metric over every primary and retry-probe "
                "/move request; this is the default SLA-gated population."
            ),
            "server_pre_completion_log_ms": (
                "Server wall time from handler entry to immediately before the "
                "completed request-log enqueue; excludes that enqueue, response "
                "and asynchronous filesystem writes."
            ),
            "server_to_agent_return_ms": (
                "Server wall time from handler entry through JSON/Pydantic, the "
                "received request-log enqueue, and agent return; not pure search time."
            ),
        },
        "workload_limit": (
            "Synthetic Hisss initial board with monotonically increasing turns; "
            "HTTP/deadline smoke only, not a representative state-complexity corpus."
        ),
        "client_round_trip_ms": client_summary,
        "client_round_trip_samples_ms": client_ms,
        "all_move_client_round_trip_ms": _latency_summary(all_client_ms),
        "all_move_client_round_trip_samples_ms": all_client_ms,
        "server_pre_completion_log_ms": _latency_summary(pre_completion_ms),
        "server_to_agent_return_ms": _latency_summary(agent_ms),
        "retry_probe": retry_report,
        "log_integrity": {
            "all_requests_paired": True,
            "failed_requests": 0,
            "paired_requests": expected_requests,
            "primary_completed_moves": len(primary_events),
            "retry_probe_completed_moves": len(retry_events),
            "async_writer": log_writer_stats,
            "game_log_store": game_log_stats,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the complete local HTTP path of PPO-MCTS."
    )
    parser.add_argument("--model")
    parser.add_argument("--moves", type=int, default=30)
    parser.add_argument("--time-budget-ms", type=float)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--min-iterations", type=int)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="seed for Python/NumPy/Torch/Hisss, deterministic game IDs and MCTS",
    )
    parser.add_argument(
        "--sla-ms",
        type=float,
        default=DEFAULT_SLA_MS,
        help="maximum allowed client /move round-trip latency (default: 500)",
    )
    parser.add_argument(
        "--no-fail-on-sla",
        dest="fail_on_sla",
        action="store_false",
        help="report SLA violations without returning a failing process status",
    )
    parser.set_defaults(fail_on_sla=True)
    parser.add_argument("--http-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.moves <= 0:
        parser.error("--moves must be positive")
    if args.seed < 0 or args.seed > MAX_NATIVE_SEED:
        parser.error(f"--seed must be between 0 and {MAX_NATIVE_SEED}")
    if not math.isfinite(args.sla_ms) or args.sla_ms <= 0:
        parser.error("--sla-ms must be finite and positive")
    if (
        not math.isfinite(args.http_timeout_seconds)
        or args.http_timeout_seconds <= 0
    ):
        parser.error("--http-timeout-seconds must be finite and positive")
    if args.time_budget_ms is not None and (
        not math.isfinite(args.time_budget_ms) or args.time_budget_ms < 0
    ):
        parser.error("--time-budget-ms must be finite and non-negative")
    for name in ("max_iterations", "min_iterations"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = benchmark(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    sla = report.get("sla")
    if args.fail_on_sla and (
        not isinstance(sla, dict) or sla.get("passed") is not True
    ):
        violations = sla.get("violations", "unknown") if isinstance(sla, dict) else "unknown"
        threshold = sla.get("threshold_ms", args.sla_ms) if isinstance(sla, dict) else args.sla_ms
        raise SystemExit(
            f"HTTP /move SLA failed: {violations} request(s) exceeded "
            f"{threshold:g} ms"
        )


if __name__ == "__main__":
    main()
