#!/usr/bin/env python3
"""Reconstruct four-player Battlesnake placements from game JSON logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO


CSV_HEADER = ("Platz1", "Platz2", "Platz3", "Platz4")
UNKNOWN_PLACEMENTS = ("X", "X", "X", "X")


class PlacementError(ValueError):
    """Raised when a placement cannot be reconstructed without guessing."""


@dataclass
class SnakeTrace:
    snake_id: str
    number: str | None = None
    elimination_turn: int | None = None


def _elimination_event(snake: dict[str, Any]) -> dict[str, Any] | None:
    event = snake.get("elimination_event") or snake.get("elimination")
    return event if isinstance(event, dict) else None


def _snake_number(snake: dict[str, Any]) -> str | None:
    """Return the numeric part of a snake name such as ``24`` or ``Snake 24``."""
    name = snake.get("name")
    if name is not None:
        text = str(name).strip()
        if text.isdigit():
            return str(int(text))
        matches = re.findall(r"\d+", text)
        if matches:
            return str(int(matches[-1]))

    # Locally simulated games use IDs like "snake-3".
    snake_id = str(snake.get("id") or "")
    match = re.fullmatch(r"snake-(\d+)", snake_id)
    return str(int(match.group(1))) if match else None


def _turn(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _states(data: dict[str, Any]) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield all logged states in chronological order."""
    states: list[tuple[int, int, dict[str, Any]]] = []
    sequence = 0

    def add(state: Any) -> None:
        nonlocal sequence
        if not isinstance(state, dict):
            return
        state_turn = _turn(state.get("turn"))
        if state_turn is None:
            return
        states.append((state_turn, sequence, state))
        sequence += 1

    add(data.get("start_state"))
    for move in data.get("moves") or []:
        if isinstance(move, dict):
            add(move.get("state"))
    add(data.get("end_state"))

    for state_turn, _, state in sorted(states):
        yield state_turn, state


def reconstruct_placements(data: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return snake numbers ordered from first through fourth place.

    A snake that disappears between two state snapshots is considered eliminated
    on the turn of the later snapshot. Explicit elimination events take priority.
    Multiple surviving snakes or simultaneous eliminations are parenthesized
    because the log contains no information with which to order them. Unknown
    snake identities are represented by ``X``.
    """
    if not isinstance(data, dict):
        return UNKNOWN_PLACEMENTS

    traces: dict[str, SnakeTrace] = {}
    snapshots: list[tuple[int, set[str]]] = []
    saw_state = False

    def record_snake(
        snake: Any,
        state_turn: int | None,
        alive: bool,
        alive_ids: set[str] | None = None,
    ) -> None:
        if not isinstance(snake, dict):
            return
        snake_id = snake.get("id")
        if not isinstance(snake_id, str) or not snake_id:
            return

        trace = traces.setdefault(snake_id, SnakeTrace(snake_id=snake_id))
        number = _snake_number(snake)
        if number is not None:
            trace.number = number

        event = _elimination_event(snake)
        event_turn = _turn(event.get("turn")) if event else None
        if event_turn is not None:
            trace.elimination_turn = event_turn
            alive = False

        if alive and state_turn is not None and alive_ids is not None:
            alive_ids.add(snake_id)

    for state_turn, state in _states(data):
        saw_state = True
        alive_ids: set[str] = set()
        board = state.get("board")
        board_snakes = board.get("snakes") if isinstance(board, dict) else []
        for snake in board_snakes or []:
            record_snake(snake, state_turn, alive=True, alive_ids=alive_ids)

        # The own snake can be omitted from board.snakes once it is eliminated.
        you = state.get("you")
        you_alive = isinstance(you, dict) and _elimination_event(you) is None
        record_snake(you, state_turn, alive=you_alive, alive_ids=alive_ids)
        snapshots.append((state_turn, alive_ids))

    result = data.get("result")
    if isinstance(result, dict):
        result_turn = _turn(result.get("turn"))
        for snake in result.get("snakes") or []:
            record_snake(snake, result_turn, alive=False)

    previous_alive: set[str] | None = None
    for state_turn, current_alive in snapshots:
        if previous_alive is not None:
            for snake_id in previous_alive - current_alive:
                trace = traces[snake_id]
                if trace.elimination_turn is None:
                    trace.elimination_turn = state_turn
        previous_alive = current_alive

    if not saw_state and not traces:
        return UNKNOWN_PLACEMENTS
    if len(traces) > 4:
        return UNKNOWN_PLACEMENTS

    groups: dict[int | None, list[SnakeTrace]] = defaultdict(list)
    for trace in traces.values():
        groups[trace.elimination_turn].append(trace)

    def number_key(trace: SnakeTrace) -> tuple[int, int | str]:
        if trace.number is not None and trace.number.isdigit():
            return 0, int(trace.number)
        if trace.number is not None:
            return 1, trace.number
        return 2, trace.snake_id

    ordered_groups: list[list[SnakeTrace]] = []
    survivors = groups.pop(None, [])
    if survivors:
        ordered_groups.append(survivors)
    for elimination_turn in sorted(groups, reverse=True):
        ordered_groups.append(groups[elimination_turn])

    numbers: list[str] = []
    for group in ordered_groups:
        ambiguous = len(group) > 1
        for trace in sorted(group, key=number_key):
            label = trace.number or "X"
            if ambiguous and label != "X":
                label = f"({label})"
            numbers.append(label)

    numbers.extend("X" for _ in range(4 - len(numbers)))
    return numbers[0], numbers[1], numbers[2], numbers[3]


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise PlacementError(f"Datei nicht lesbar: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PlacementError(f"ungueltiges JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PlacementError("Top-Level-JSON ist kein Objekt")
    return data


def extract_directory(log_dir: Path, output: TextIO) -> tuple[int, int]:
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(CSV_HEADER)

    written = 0
    skipped = 0
    for path in sorted(log_dir.glob("*.json"), key=lambda item: item.name):
        try:
            writer.writerow(reconstruct_placements(load_json(path)))
            written += 1
        except PlacementError:
            writer.writerow(UNKNOWN_PLACEMENTS)
            written += 1
            skipped += 1
    return written, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rekonstruiert die Platzierungen aus allen JSON-Spiellogs eines "
            "Ordners und gibt Platz1,Platz2,Platz3,Platz4 als CSV aus."
        )
    )
    parser.add_argument("log_dir", type=Path, help="Ordner mit den JSON-Logs")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="CSV-Datei (ohne diese Option erfolgt die Ausgabe auf stdout)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.log_dir.is_dir():
        print(f"FEHLER: Log-Ordner nicht gefunden: {args.log_dir}", file=sys.stderr)
        return 2

    try:
        if args.output is None:
            extract_directory(args.log_dir, sys.stdout)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8", newline="") as handle:
                extract_directory(args.log_dir, handle)
    except OSError as exc:
        print(f"FEHLER: CSV konnte nicht geschrieben werden: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
