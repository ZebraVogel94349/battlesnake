#!/usr/bin/env python3
"""Analyze Battlesnake matchups based on ``extract_placements`` results."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO

from extract_placements import PlacementError, load_json, reconstruct_placements


@dataclass(frozen=True)
class Placement:
    snake: str
    rank: float
    tied: bool
    tie_group: int | None


@dataclass(frozen=True)
class GameRecord:
    path: Path
    game_id: str
    placements: tuple[str, str, str, str]
    detected_own: str | None
    quality: tuple[int, int, int, int, str]


@dataclass
class MatchupStats:
    own: str
    opponent: str
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    game_wins: int = 0
    placement_sum: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def score_rate(self) -> float:
        if not self.games:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def game_win_rate(self) -> float:
        return self.game_wins / self.games if self.games else 0.0

    @property
    def average_placement(self) -> float:
        return self.placement_sum / self.games if self.games else 0.0


@dataclass
class OwnStats:
    games: int = 0
    game_wins: int = 0
    placement_sum: float = 0.0

    @property
    def game_win_rate(self) -> float:
        return self.game_wins / self.games if self.games else 0.0

    @property
    def average_placement(self) -> float:
        return self.placement_sum / self.games if self.games else 0.0


@dataclass(frozen=True)
class LoadSummary:
    records: list[GameRecord]
    files_seen: int
    duplicate_skips: int
    invalid_files: int


def normalize_number(value: Any) -> str | None:
    """Use the same snake-number convention as ``extract_placements``."""
    if value is None:
        return None
    text = str(value).strip()
    if text.isdigit():
        return str(int(text))
    matches = re.findall(r"\d+", text)
    return str(int(matches[-1])) if matches else None


def infer_own_snake(data: dict[str, Any]) -> str | None:
    """Find the numeric name of the snake represented by ``you`` in a log."""
    own_id: str | None = None

    result = data.get("result")
    if isinstance(result, dict) and isinstance(result.get("you"), str):
        own_id = result["you"]

    def own_snakes() -> Iterable[dict[str, Any]]:
        for state_key in ("end_state", "start_state"):
            state = data.get(state_key)
            if isinstance(state, dict) and isinstance(state.get("you"), dict):
                yield state["you"]
        for move in reversed(data.get("moves") or []):
            if not isinstance(move, dict):
                continue
            state = move.get("state")
            if isinstance(state, dict) and isinstance(state.get("you"), dict):
                yield state["you"]

    for snake in own_snakes():
        snake_id = snake.get("id")
        if isinstance(snake_id, str) and snake_id:
            own_id = own_id or snake_id
        number = normalize_number(snake.get("name"))
        if number is not None:
            return number

    if own_id is None:
        return None

    states: list[Any] = [data.get("end_state"), data.get("start_state")]
    states.extend(
        move.get("state")
        for move in reversed(data.get("moves") or [])
        if isinstance(move, dict)
    )
    for state in states:
        if not isinstance(state, dict):
            continue
        board = state.get("board")
        snakes = board.get("snakes") if isinstance(board, dict) else []
        for snake in snakes or []:
            if isinstance(snake, dict) and snake.get("id") == own_id:
                number = normalize_number(snake.get("name"))
                if number is not None:
                    return number
    return None


def game_id(data: dict[str, Any], path: Path) -> str:
    if data.get("game_id"):
        return str(data["game_id"])
    for state_key in ("end_state", "start_state"):
        state = data.get(state_key)
        game = state.get("game") if isinstance(state, dict) else None
        if isinstance(game, dict) and game.get("id"):
            return str(game["id"])
    return path.stem


def game_quality(data: dict[str, Any], path: Path) -> tuple[int, int, int, int, str]:
    moves = data.get("moves") or []
    return (
        int(bool(data.get("result"))),
        int(bool(moves)),
        len(moves),
        int(bool(data.get("end_state"))),
        path.name,
    )


def make_record(
    path: Path,
    data: dict[str, Any],
    *,
    detect_own: bool = True,
) -> GameRecord:
    return GameRecord(
        path=path,
        game_id=game_id(data, path),
        placements=reconstruct_placements(data),
        detected_own=infer_own_snake(data) if detect_own else None,
        quality=game_quality(data, path),
    )


def load_records(
    log_dir: Path,
    *,
    include_duplicates: bool = False,
    limit: int | None = None,
    detect_own: bool = True,
) -> LoadSummary:
    paths = sorted(log_dir.glob("*.json"), key=lambda path: path.name)
    if limit is not None:
        paths = paths[-limit:]

    records: list[GameRecord] = []
    best_by_game: dict[str, GameRecord] = {}
    invalid_files = 0
    valid_files = 0

    for path in paths:
        try:
            record = make_record(path, load_json(path), detect_own=detect_own)
        except PlacementError:
            invalid_files += 1
            continue
        valid_files += 1
        if include_duplicates:
            records.append(record)
            continue
        current = best_by_game.get(record.game_id)
        if current is None or record.quality > current.quality:
            best_by_game[record.game_id] = record

    if not include_duplicates:
        records = sorted(best_by_game.values(), key=lambda record: record.path.name)
    duplicate_skips = valid_files - len(records)
    return LoadSummary(
        records=records,
        files_seen=len(paths),
        duplicate_skips=duplicate_skips,
        invalid_files=invalid_files,
    )


def parse_placements(values: Iterable[str]) -> dict[str, Placement]:
    """Turn flat placement labels into ranks, averaging contiguous ties."""
    raw = list(values)
    parsed: dict[str, Placement] = {}
    tie_group = 0
    index = 0
    while index < len(raw):
        value = raw[index]
        tied = value.startswith("(") and value.endswith(")")
        end = index + 1
        if tied:
            while end < len(raw):
                following = raw[end]
                if not (following.startswith("(") and following.endswith(")")):
                    break
                end += 1
        rank = ((index + 1) + end) / 2.0
        group = tie_group if tied else None
        if tied:
            tie_group += 1
        for position in range(index, end):
            label = raw[position].strip("()")
            if label != "X" and label not in parsed:
                parsed[label] = Placement(
                    snake=label,
                    rank=rank,
                    tied=tied,
                    tie_group=group,
                )
        index = end
    return parsed


def is_game_win(placement: Placement) -> bool:
    return placement.rank == 1.0 and not placement.tied


def head_to_head(own: Placement, opponent: Placement) -> str:
    if own.tie_group is not None and own.tie_group == opponent.tie_group:
        return "draw"
    if own.rank < opponent.rank:
        return "win"
    if own.rank > opponent.rank:
        return "loss"
    return "draw"


def analyze_records(
    records: Iterable[GameRecord],
    *,
    own_filter: set[str] | None = None,
    opponent_filter: set[str] | None = None,
) -> tuple[dict[tuple[str, str], MatchupStats], dict[str, OwnStats], int]:
    matchups: dict[tuple[str, str], MatchupStats] = {}
    own_totals: dict[str, OwnStats] = {}
    skipped_without_own = 0

    for record in records:
        placements = parse_placements(record.placements)
        if own_filter is None:
            own_snakes = (
                [record.detected_own]
                if record.detected_own in placements
                else []
            )
        else:
            own_snakes = sorted(own_filter.intersection(placements), key=snake_key)

        if not own_snakes:
            skipped_without_own += 1
            continue

        excluded_opponents = set(own_snakes)
        if own_filter is not None:
            excluded_opponents.update(own_filter)

        for own_name in own_snakes:
            own_placement = placements[own_name]
            own_stats = own_totals.setdefault(own_name, OwnStats())
            own_stats.games += 1
            own_stats.placement_sum += own_placement.rank
            if is_game_win(own_placement):
                own_stats.game_wins += 1

            opponents = set(placements).difference(excluded_opponents)
            if opponent_filter is not None:
                opponents.intersection_update(opponent_filter)
            for opponent_name in sorted(opponents, key=snake_key):
                key = own_name, opponent_name
                stats = matchups.setdefault(
                    key,
                    MatchupStats(own=own_name, opponent=opponent_name),
                )
                stats.games += 1
                stats.placement_sum += own_placement.rank
                if is_game_win(own_placement):
                    stats.game_wins += 1
                outcome = head_to_head(own_placement, placements[opponent_name])
                if outcome == "win":
                    stats.wins += 1
                elif outcome == "draw":
                    stats.draws += 1
                else:
                    stats.losses += 1

    return matchups, own_totals, skipped_without_own


def snake_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def best_key(stats: MatchupStats) -> tuple[float, float, int, tuple[int, int | str]]:
    return stats.score_rate, stats.win_rate, stats.games, snake_key(stats.opponent)


def hardest_key(
    stats: MatchupStats,
) -> tuple[float, float, int, tuple[int, int | str]]:
    return stats.score_rate, stats.win_rate, -stats.games, snake_key(stats.opponent)


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        )
    return "\n".join(lines)


def matchup_row(stats: MatchupStats) -> list[str]:
    return [
        stats.opponent,
        str(stats.games),
        f"{stats.wins}-{stats.draws}-{stats.losses}",
        percent(stats.win_rate),
        percent(stats.score_rate),
        percent(stats.game_win_rate),
        f"{stats.average_placement:.2f}",
    ]


def print_report(
    load_summary: LoadSummary,
    matchups: dict[tuple[str, str], MatchupStats],
    own_totals: dict[str, OwnStats],
    *,
    min_games: int,
    top: int,
    skipped_without_own: int,
    output: TextIO = sys.stdout,
) -> None:
    print("=== Battlesnake-Matchups ===", file=output)
    print(
        f"Logs: {load_summary.files_seen}, eindeutige Spiele: "
        f"{len(load_summary.records)}, Duplikate: {load_summary.duplicate_skips}, "
        f"ungueltig: {load_summary.invalid_files}",
        file=output,
    )
    if skipped_without_own:
        print(f"Ohne passende eigene Snake uebersprungen: {skipped_without_own}", file=output)

    if not own_totals:
        print("Keine Spiele mit einer passenden eigenen Snake gefunden.", file=output)
        return

    headers = [
        "Gegner",
        "Spiele",
        "S-U-N",
        "H2H-Winrate",
        "H2H-Score",
        "Spiel-Winrate",
        "Avg Platz",
    ]
    for own_name in sorted(own_totals, key=snake_key):
        total = own_totals[own_name]
        print(f"\n--- Eigene Snake {own_name} ---", file=output)
        print(
            f"Spiele: {total.games}, Gesamtsiege: {total.game_wins} "
            f"({percent(total.game_win_rate)}), Avg Platz: "
            f"{total.average_placement:.2f}",
            file=output,
        )
        rows = [
            stats
            for (current_own, _), stats in matchups.items()
            if current_own == own_name and stats.games >= min_games
        ]
        rows.sort(key=hardest_key)
        visible = rows[:top] if top > 0 else rows
        if not visible:
            print(
                f"Keine Gegner mit mindestens {min_games} gemeinsamen Spielen.",
                file=output,
            )
            continue
        print(table(headers, [matchup_row(stats) for stats in visible]), file=output)
        best = max(rows, key=best_key)
        hardest = rows[0]
        print(
            f"Beste Bilanz: gegen {best.opponent} ({percent(best.score_rate)} "
            f"H2H-Score, n={best.games})",
            file=output,
        )
        print(
            f"Schwierigster Gegner: {hardest.opponent} "
            f"({percent(hardest.score_rate)} H2H-Score, n={hardest.games})",
            file=output,
        )

    print(
        "\nH2H-Winrate = Siege / gemeinsame Spiele; "
        "H2H-Score wertet ein Unentschieden als halben Sieg.",
        file=output,
    )


CSV_HEADER = (
    "eigene_snake",
    "gegner",
    "spiele",
    "siege",
    "unentschieden",
    "niederlagen",
    "h2h_winrate_prozent",
    "h2h_score_prozent",
    "spiel_winrate_prozent",
    "durchschnittsplatz",
)


def write_csv(
    matchups: dict[tuple[str, str], MatchupStats],
    output: TextIO,
    *,
    min_games: int = 1,
) -> None:
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(CSV_HEADER)
    rows = [stats for stats in matchups.values() if stats.games >= min_games]
    rows.sort(key=lambda stats: (snake_key(stats.own), snake_key(stats.opponent)))
    for stats in rows:
        writer.writerow(
            (
                stats.own,
                stats.opponent,
                stats.games,
                stats.wins,
                stats.draws,
                stats.losses,
                f"{stats.win_rate * 100:.2f}",
                f"{stats.score_rate * 100:.2f}",
                f"{stats.game_win_rate * 100:.2f}",
                f"{stats.average_placement:.3f}",
            )
        )


def normalized_cli_numbers(values: list[str] | None, option: str) -> set[str] | None:
    if values is None:
        return None
    result: set[str] = set()
    for value in values:
        number = normalize_number(value)
        if number is None:
            raise ValueError(f"{option}: keine Snake-Nummer: {value!r}")
        result.add(number)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Berechnet Winrates und Platzierungen eigener Snakes gegen einzelne "
            "Gegner. Die Platzierungen werden mit extract_placements rekonstruiert."
        )
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        default=Path("game_logs"),
        type=Path,
        help="Ordner mit JSON-Logs (Standard: game_logs)",
    )
    parser.add_argument(
        "--own",
        nargs="+",
        metavar="NUMMER",
        help=(
            "eigene Snake-Nummer(n); ohne Option wird pro Log state.you verwendet"
        ),
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        metavar="NUMMER",
        help="nur diese Gegner auswerten",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=1,
        help="Mindestzahl gemeinsamer Spiele pro Tabellenzeile (Standard: 1)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help=(
            "maximale Gegnerzeilen je eigener Snake, schwierigste zuerst; "
            "0 zeigt alle (Standard: 10)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="nur die neuesten N JSON-Dateien betrachten",
    )
    parser.add_argument(
        "--include-duplicates",
        action="store_true",
        help="mehrere Logs derselben game_id mehrfach zaehlen",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="zusaetzlich alle gefilterten Matchups als CSV schreiben",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.log_dir.is_dir():
        print(f"FEHLER: Log-Ordner nicht gefunden: {args.log_dir}", file=sys.stderr)
        return 2
    if args.min_games < 1:
        print("FEHLER: --min-games muss mindestens 1 sein", file=sys.stderr)
        return 2
    if args.top < 0:
        print("FEHLER: --top darf nicht negativ sein", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("FEHLER: --limit muss mindestens 1 sein", file=sys.stderr)
        return 2

    try:
        own_filter = normalized_cli_numbers(args.own, "--own")
        opponent_filter = normalized_cli_numbers(args.opponents, "--opponents")
    except ValueError as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 2

    summary = load_records(
        args.log_dir,
        include_duplicates=args.include_duplicates,
        limit=args.limit,
        detect_own=own_filter is None,
    )
    matchups, own_totals, skipped_without_own = analyze_records(
        summary.records,
        own_filter=own_filter,
        opponent_filter=opponent_filter,
    )
    print_report(
        summary,
        matchups,
        own_totals,
        min_games=args.min_games,
        top=args.top,
        skipped_without_own=skipped_without_own,
    )

    if args.csv_out is not None:
        try:
            args.csv_out.parent.mkdir(parents=True, exist_ok=True)
            with args.csv_out.open("w", encoding="utf-8", newline="") as handle:
                write_csv(matchups, handle, min_games=args.min_games)
        except OSError as exc:
            print(f"FEHLER: CSV konnte nicht geschrieben werden: {exc}", file=sys.stderr)
            return 2
        print(f"CSV geschrieben: {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
