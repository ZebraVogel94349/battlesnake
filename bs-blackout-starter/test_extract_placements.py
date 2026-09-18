from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("extract_placements.py")
SPEC = importlib.util.spec_from_file_location("extract_placements", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
extract_placements = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extract_placements
SPEC.loader.exec_module(extract_placements)


def snake(number: int, eliminated_at: int | None = None) -> dict:
    event = None
    if eliminated_at is not None:
        event = {"cause": "snake-collision", "turn": eliminated_at, "by": None}
    return {
        "id": f"id-{number}",
        "name": str(number),
        "elimination_event": event,
    }


def state(
    turn: int,
    alive: tuple[int, ...],
    *,
    you: int = 2,
    you_eliminated_at: int | None = None,
) -> dict:
    return {
        "turn": turn,
        "board": {"snakes": [snake(number) for number in alive]},
        "you": snake(you, you_eliminated_at),
    }


def completed_log(end_state: dict) -> dict:
    return {
        "start_state": state(0, (1, 2, 3, 4)),
        "moves": [
            {"state": state(3, (1, 2, 3))},
            {"state": state(5, (1, 2))},
        ],
        "end_state": end_state,
        "result": {"turn": end_state["turn"], "snakes": []},
    }


def test_reconstructs_four_unique_places() -> None:
    data = completed_log(state(8, (1,), you_eliminated_at=8))

    assert extract_placements.reconstruct_placements(data) == (
        "1",
        "2",
        "3",
        "4",
    )


def test_parenthesizes_opponents_that_are_still_alive() -> None:
    data = {
        "start_state": state(0, (1, 2, 3, 24), you=24),
        "moves": [],
        "end_state": state(8, (1, 2, 3), you=24, you_eliminated_at=8),
        "result": {"turn": 8, "snakes": []},
    }

    assert extract_placements.reconstruct_placements(data) == (
        "(1)",
        "(2)",
        "(3)",
        "24",
    )


def test_parenthesizes_simultaneous_eliminations() -> None:
    data = {
        "start_state": state(0, (1, 2, 3, 24), you=24),
        "moves": [{"state": state(3, (2, 3, 24), you=24)}],
        "end_state": state(8, (24,), you=24),
        "result": {"turn": 8, "snakes": []},
    }

    assert extract_placements.reconstruct_placements(data) == (
        "24",
        "(2)",
        "(3)",
        "1",
    )


def test_uses_x_for_an_unknown_snake_number() -> None:
    data = completed_log(state(8, (1,), you_eliminated_at=8))
    states = [data["start_state"], *(move["state"] for move in data["moves"])]
    for game_state in states:
        for current_snake in game_state["board"]["snakes"]:
            if current_snake["id"] == "id-3":
                current_snake["name"] = "unbekannt"

    assert extract_placements.reconstruct_placements(data) == ("1", "2", "X", "4")


def test_directory_output_has_requested_csv_shape(tmp_path: Path) -> None:
    (tmp_path / "game.json").write_text(
        json.dumps(completed_log(state(8, (1,), you_eliminated_at=8))),
        encoding="utf-8",
    )
    output = io.StringIO()

    assert extract_placements.extract_directory(tmp_path, output) == (1, 0)
    assert output.getvalue() == "Platz1,Platz2,Platz3,Platz4\n1,2,3,4\n"


def test_invalid_json_is_silent_and_uses_placeholders(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    output = io.StringIO()

    assert extract_placements.extract_directory(tmp_path, output) == (1, 1)
    assert output.getvalue() == "Platz1,Platz2,Platz3,Platz4\nX,X,X,X\n"
    assert capsys.readouterr().err == ""
