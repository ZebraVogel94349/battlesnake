from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_matchups.py")
SPEC = importlib.util.spec_from_file_location("analyze_matchups", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
analyze_matchups = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(MODULE_PATH.parent))
sys.modules[SPEC.name] = analyze_matchups
SPEC.loader.exec_module(analyze_matchups)


def record(
    name: str,
    placements: tuple[str, str, str, str],
    *,
    own: str | None = "24",
) -> analyze_matchups.GameRecord:
    return analyze_matchups.GameRecord(
        path=Path(name),
        game_id=name,
        placements=placements,
        detected_own=own,
        quality=(1, 1, 1, 1, name),
    )


def test_analyzes_pairwise_winrates_and_game_wins() -> None:
    records = [
        record("win", ("24", "1", "2", "3")),
        record("loss", ("1", "24", "2", "3")),
        record("draw", ("2", "3", "(1)", "(24)")),
    ]

    matchups, own_totals, skipped = analyze_matchups.analyze_records(records)

    stats = matchups[("24", "1")]
    assert (stats.games, stats.wins, stats.draws, stats.losses) == (3, 1, 1, 1)
    assert stats.win_rate == 1 / 3
    assert stats.score_rate == 0.5
    assert stats.game_wins == 1
    assert own_totals["24"].games == 3
    assert own_totals["24"].game_wins == 1
    assert skipped == 0


def test_explicit_filters_support_multiple_own_snakes_and_opponents() -> None:
    records = [
        record("a", ("24", "1", "2", "3"), own=None),
        record("b", ("2", "25", "1", "4"), own=None),
    ]

    matchups, own_totals, _ = analyze_matchups.analyze_records(
        records,
        own_filter={"24", "25"},
        opponent_filter={"1"},
    )

    assert set(own_totals) == {"24", "25"}
    assert set(matchups) == {("24", "1"), ("25", "1")}
    assert matchups[("24", "1")].wins == 1
    assert matchups[("25", "1")].wins == 1


def test_load_records_keeps_richest_log_for_duplicate_game_id(
    tmp_path: Path,
) -> None:
    sparse = {
        "game_id": "same-game",
        "start_state": {"turn": 0, "board": {"snakes": []}},
        "moves": [],
    }
    rich = {
        "game_id": "same-game",
        "start_state": {"turn": 0, "board": {"snakes": []}},
        "moves": [{"state": {"turn": 1, "board": {"snakes": []}}}],
        "result": {"turn": 1, "snakes": []},
    }
    (tmp_path / "a.json").write_text(json.dumps(sparse), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(rich), encoding="utf-8")

    summary = analyze_matchups.load_records(tmp_path)

    assert summary.files_seen == 2
    assert summary.duplicate_skips == 1
    assert [item.path.name for item in summary.records] == ["b.json"]


def test_csv_contains_machine_readable_rates() -> None:
    records = [
        record("win", ("24", "1", "2", "3")),
        record("loss", ("1", "24", "2", "3")),
    ]
    matchups, _, _ = analyze_matchups.analyze_records(
        records,
        opponent_filter={"1"},
    )
    output = io.StringIO()

    analyze_matchups.write_csv(matchups, output)

    lines = output.getvalue().splitlines()
    assert lines[0].startswith("eigene_snake,gegner,spiele")
    assert lines[1] == "24,1,2,1,0,1,50.00,50.00,50.00,1.500"


def test_hardest_ranking_prefers_more_evidence_when_rates_match() -> None:
    few_games = analyze_matchups.MatchupStats(
        own="24", opponent="1", games=1, losses=1
    )
    many_games = analyze_matchups.MatchupStats(
        own="24", opponent="2", games=5, losses=5
    )

    ranked = sorted([few_games, many_games], key=analyze_matchups.hardest_key)

    assert ranked[0] is many_games
