"""Combine compatible, complete ``evaluate_inference.py`` JSON reports.

The combiner validates the effective model/code/config identity rather than
machine-dependent checkpoint paths.  It then renumbers complete pairs and
reuses the current evaluator's summary and paired bootstrap implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

from evaluate_inference import (
    METRIC_SEMANTICS_SHA256,
    METRIC_SEMANTICS_VERSION,
    REPORT_SCHEMA_VERSION,
    REPORT_TYPE,
    MatchResult,
    evaluation_provenance,
    summarize,
    validate_placement_outcome,
)


ROLES = ("candidate", "baseline")
ENSEMBLE_SEARCH_FIELDS = (
    "ensemble_moves",
    "waves_started",
    "waves_completed",
    "waves_discarded",
    "voting_iterations",
    "discarded_iterations",
    "stopped_for_wave_guard",
    "decision_reason_counts",
)
NUMERIC_SEARCH_FIELDS = (
    "moves",
    "iterations",
    "nodes",
    "depth",
    "max_depth",
    "elapsed_ms",
    "deadline_hits",
    "errors",
    "mcts_choices",
    "insufficient_searches",
    "policy_changes",
    *(field for field in ENSEMBLE_SEARCH_FIELDS if field != "decision_reason_counts"),
)
VOLATILE_DIAGNOSTIC_FIELDS = {"last_search"}
COMBINED_REPORT_SCHEMA_VERSION = 2


class ReportValidationError(ValueError):
    pass


@dataclass(frozen=True)
class _LoadedReport:
    path: Path
    sha256: str
    raw: dict
    identities: dict[str, dict]
    identity_fingerprints: dict[str, dict[str, str]]
    matches: list[MatchResult]
    pair_seeds: list[int]
    ensemble_metrics_captured: bool
    ensemble_metrics_matches: int
    ensemble_metrics_pairs: int
    evaluation_provenance: dict | None
    legacy_provenance: bool
    hisss_version: str | None


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ReportValidationError(f"value is not canonical JSON: {exc}") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _object_fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_dict(value: object, context: str) -> dict:
    if not isinstance(value, dict):
        raise ReportValidationError(f"{context} must be an object")
    return value


def _require_int(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReportValidationError(f"{context} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, context: str, *, minimum: float | None = None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
    ):
        qualifier = "finite"
        if minimum is not None:
            qualifier += f" and >= {minimum}"
        raise ReportValidationError(f"{context} must be {qualifier}")
    return value


def _require_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ReportValidationError(
            f"{context} must be an available 64-character SHA-256 fingerprint"
        )
    return value.lower()


def _component_fingerprint(components: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, fingerprint in sorted(components.items()):
        label = name.encode("utf-8", errors="surrogatepass")
        digest.update(len(label).to_bytes(4, byteorder="big", signed=False))
        digest.update(label)
        digest.update(bytes.fromhex(fingerprint))
    return digest.hexdigest()


def _validated_evaluation_provenance(value: object, context: str) -> dict:
    provenance = _require_dict(value, context)
    schema = _require_dict(provenance.get("report_schema"), f"{context}.report_schema")
    if schema != {"name": REPORT_TYPE, "version": REPORT_SCHEMA_VERSION}:
        raise ReportValidationError(
            f"{context}.report_schema is incompatible with this combiner"
        )

    evaluator = _require_dict(provenance.get("evaluator"), f"{context}.evaluator")
    evaluator_sha = _require_sha256(
        evaluator.get("code_sha256"),
        f"{context}.evaluator.code_sha256",
    )
    components = _require_dict(
        evaluator.get("component_sha256"),
        f"{context}.evaluator.component_sha256",
    )
    if not components:
        raise ReportValidationError(f"{context}.evaluator components are missing")
    normalized_components = {}
    for name, fingerprint in components.items():
        if not isinstance(name, str) or not name:
            raise ReportValidationError(
                f"{context}.evaluator has an invalid component name"
            )
        normalized_components[name] = _require_sha256(
            fingerprint,
            f"{context}.evaluator.component_sha256.{name}",
        )
    if _component_fingerprint(normalized_components) != evaluator_sha:
        raise ReportValidationError(
            f"{context}.evaluator code fingerprint does not match components"
        )

    metrics = _require_dict(
        provenance.get("metric_semantics"),
        f"{context}.metric_semantics",
    )
    metric_sha = _require_sha256(
        metrics.get("sha256"),
        f"{context}.metric_semantics.sha256",
    )
    definitions = _require_dict(
        metrics.get("definitions"),
        f"{context}.metric_semantics.definitions",
    )
    if _object_fingerprint(definitions) != metric_sha:
        raise ReportValidationError(
            f"{context}.metric_semantics fingerprint does not match definitions"
        )
    if metric_sha != METRIC_SEMANTICS_SHA256:
        raise ReportValidationError(
            f"{context}.metric_semantics is incompatible with the current summarizer"
        )
    if metrics.get("version") != METRIC_SEMANTICS_VERSION:
        raise ReportValidationError(
            f"{context}.metric_semantics.version is incompatible"
        )

    hisss = _require_dict(provenance.get("hisss"), f"{context}.hisss")
    version = hisss.get("version")
    if not isinstance(version, str) or version in {"", "unknown", "unavailable"}:
        raise ReportValidationError(f"{context}.hisss.version is unavailable")
    normalized_hisss = {"version": version}
    for field in (
        "python_sources_sha256",
        "native_library_sha256",
        "implementation_sha256",
    ):
        normalized_hisss[field] = _require_sha256(
            hisss.get(field),
            f"{context}.hisss.{field}",
        )
    expected_hisss_sha = _object_fingerprint(
        {
            "version": normalized_hisss["version"],
            "python_sources_sha256": normalized_hisss["python_sources_sha256"],
            "native_library_sha256": normalized_hisss["native_library_sha256"],
        }
    )
    if normalized_hisss["implementation_sha256"] != expected_hisss_sha:
        raise ReportValidationError(
            f"{context}.Hisss runtime fingerprint does not match its components"
        )

    _canonical_json(provenance)
    return dict(provenance)


def _legacy_hisss_version(report: dict) -> str | None:
    runtime = report.get("runtime")
    if not isinstance(runtime, dict):
        return None
    hisss = runtime.get("hisss")
    if not isinstance(hisss, dict):
        return None
    version = hisss.get("version")
    if not isinstance(version, str) or version in {"", "unknown", "unavailable"}:
        return None
    return version


def _provenance_signature(provenance: dict) -> dict[str, str]:
    return {
        "evaluator_code_sha256": provenance["evaluator"]["code_sha256"],
        "metric_semantics_sha256": provenance["metric_semantics"]["sha256"],
        "metric_semantics_version": provenance["metric_semantics"]["version"],
        "hisss_version": provenance["hisss"]["version"],
        "hisss_implementation_sha256": provenance["hisss"][
            "implementation_sha256"
        ],
    }


def _normalized_diagnostics(value: object, context: str) -> dict:
    diagnostics = dict(_require_dict(value, context))
    # last_search is request-local state, not part of the effective agent.
    for field in VOLATILE_DIAGNOSTIC_FIELDS:
        diagnostics.pop(field, None)
    agent_name = diagnostics.get("agent")
    if not isinstance(agent_name, str) or not agent_name:
        raise ReportValidationError(f"{context}.agent must be a non-empty string")
    for field in ("model_sha256", "code_sha256"):
        fingerprint = diagnostics.get(field)
        if (
            not isinstance(fingerprint, str)
            or not 12 <= len(fingerprint) <= 64
            or len(fingerprint) % 2
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in fingerprint
            )
        ):
            raise ReportValidationError(
                f"{context}.{field} must be an available hexadecimal fingerprint"
            )
    _canonical_json(diagnostics)
    return diagnostics


def _identity_fingerprints(identity: dict) -> dict[str, str]:
    diagnostics = identity["diagnostics"]
    effective_config = {
        key: value
        for key, value in diagnostics.items()
        if key not in {"model_sha256", "code_sha256"}
    }
    return {
        "model_sha256": diagnostics["model_sha256"],
        "code_sha256": diagnostics["code_sha256"],
        "config_sha256": _object_fingerprint(
            {"kind": identity["kind"], "effective_config": effective_config}
        ),
        "identity_sha256": _object_fingerprint(identity),
    }


def _agent_identity(report: dict, role: str, context: str) -> dict:
    spec = _require_dict(report.get(role), f"{context}.{role}")
    _canonical_json(spec)
    kind = spec.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ReportValidationError(f"{context}.{role}.kind is missing")
    model = spec.get("model")
    if not isinstance(model, str) or not model:
        raise ReportValidationError(f"{context}.{role}.model is missing")
    effective = _require_dict(
        report.get("effective_agent_diagnostics"),
        f"{context}.effective_agent_diagnostics",
    )
    diagnostics = _normalized_diagnostics(
        effective.get(role), f"{context}.effective_agent_diagnostics.{role}"
    )

    # Explicit CLI overrides must agree with the effective runtime config.
    for field in ("time_budget_ms", "max_iterations", "min_iterations"):
        explicit = spec.get(field)
        if explicit is not None:
            if field == "time_budget_ms":
                _finite_number(
                    explicit,
                    f"{context}.{role}.{field}",
                    minimum=0.0,
                )
            else:
                _require_int(explicit, f"{context}.{role}.{field}")
        if explicit is not None and diagnostics.get(field) != explicit:
            raise ReportValidationError(
                f"{context}.{role}.{field}={explicit!r} disagrees with "
                f"effective value {diagnostics.get(field)!r}"
            )
    return {"kind": kind, "diagnostics": diagnostics}


def _normalize_search_stats(value: object, context: str) -> tuple[dict, bool]:
    all_stats = dict(_require_dict(value, context))
    captured = True
    for role in ROLES:
        stats = dict(_require_dict(all_stats.get(role), f"{context}.{role}"))
        captured = captured and all(field in stats for field in ENSEMBLE_SEARCH_FIELDS)
        for field in ENSEMBLE_SEARCH_FIELDS:
            if field == "decision_reason_counts":
                stats.setdefault(field, {})
            else:
                stats.setdefault(field, 0.0)
        for field in NUMERIC_SEARCH_FIELDS:
            if field in stats:
                _finite_number(
                    stats[field],
                    f"{context}.{role}.{field}",
                    minimum=0.0,
                )
        reason_counts = stats["decision_reason_counts"]
        if not isinstance(reason_counts, dict):
            raise ReportValidationError(
                f"{context}.{role}.decision_reason_counts must be an object"
            )
        for reason, count in reason_counts.items():
            if not isinstance(reason, str) or not reason:
                raise ReportValidationError(
                    f"{context}.{role} has an invalid decision reason"
                )
            _finite_number(
                count,
                f"{context}.{role}.decision_reason_counts.{reason}",
                minimum=0.0,
            )
        all_stats[role] = stats
    _canonical_json(all_stats)
    return all_stats, captured


def _parse_match(value: object, context: str) -> tuple[MatchResult, bool]:
    match = _require_dict(value, context)
    pair = _require_int(match.get("pair"), f"{context}.pair")
    game_in_pair = _require_int(
        match.get("game_in_pair"), f"{context}.game_in_pair"
    )
    seed = _require_int(match.get("seed"), f"{context}.seed")
    turns = _require_int(match.get("turns"), f"{context}.turns")

    seat_kinds = match.get("seat_kinds")
    points = match.get("points")
    if not isinstance(seat_kinds, list) or len(seat_kinds) != 4:
        raise ReportValidationError(f"{context}.seat_kinds must have four seats")
    if not isinstance(points, list) or len(points) != 4:
        raise ReportValidationError(f"{context}.points must have four values")
    if any(kind not in ROLES for kind in seat_kinds):
        raise ReportValidationError(f"{context}.seat_kinds contains an unknown role")
    parsed_points = [
        float(_finite_number(point, f"{context}.points[{index}]", minimum=0.0))
        for index, point in enumerate(points)
    ]

    latency = _require_dict(match.get("move_latencies_ms"), f"{context}.latency")
    parsed_latency: dict[str, list[float]] = {}
    for role in ROLES:
        values = latency.get(role)
        if not isinstance(values, list):
            raise ReportValidationError(f"{context}.latency.{role} must be a list")
        parsed_latency[role] = [
            float(
                _finite_number(
                    item,
                    f"{context}.latency.{role}[{index}]",
                    minimum=0.0,
                )
            )
            for index, item in enumerate(values)
        ]

    causes = _require_dict(
        match.get("elimination_causes"), f"{context}.elimination_causes"
    )
    try:
        parsed_causes = {int(seat): cause for seat, cause in causes.items()}
    except (TypeError, ValueError) as exc:
        raise ReportValidationError(
            f"{context}.elimination_causes has an invalid seat"
        ) from exc
    if any(seat not in range(4) for seat in parsed_causes):
        raise ReportValidationError(
            f"{context}.elimination_causes contains an unknown seat"
        )
    if len(parsed_causes) != len(causes):
        raise ReportValidationError(
            f"{context}.elimination_causes contains duplicate seat keys"
        )
    if any(not isinstance(cause, str) or not cause for cause in parsed_causes.values()):
        raise ReportValidationError(
            f"{context}.elimination_causes values must be non-empty strings"
        )

    fatal = _require_dict(match.get("fatal_moves"), f"{context}.fatal_moves")
    avoidable = _require_dict(
        match.get("avoidable_fatal_moves"), f"{context}.avoidable_fatal_moves"
    )
    parsed_fatal = {
        role: _require_int(fatal.get(role), f"{context}.fatal_moves.{role}")
        for role in ROLES
    }
    parsed_avoidable = {
        role: _require_int(
            avoidable.get(role), f"{context}.avoidable_fatal_moves.{role}"
        )
        for role in ROLES
    }

    search_stats, captured = _normalize_search_stats(
        match.get("search_stats"), f"{context}.search_stats"
    )
    diagnostics = _require_dict(
        match.get("agent_diagnostics"), f"{context}.agent_diagnostics"
    )
    _canonical_json(diagnostics)
    if any(role not in diagnostics for role in ROLES):
        raise ReportValidationError(
            f"{context}.agent_diagnostics must contain candidate and baseline"
        )

    timed_out = match.get("timed_out")
    if not isinstance(timed_out, bool):
        raise ReportValidationError(f"{context}.timed_out must be boolean")
    winner = match.get("winner")
    if winner is not None:
        winner = _require_int(winner, f"{context}.winner")
        if winner not in range(4):
            raise ReportValidationError(f"{context}.winner is not a valid seat")
    try:
        validate_placement_outcome(
            parsed_points,
            winner,
            parsed_causes,
            timed_out,
            context=context,
        )
    except ValueError as exc:
        raise ReportValidationError(str(exc)) from exc

    return (
        MatchResult(
            pair=pair,
            game_in_pair=game_in_pair,
            seed=seed,
            seat_kinds=list(seat_kinds),
            points=parsed_points,
            turns=turns,
            timed_out=timed_out,
            winner=winner,
            elimination_causes=parsed_causes,
            move_latencies_ms=parsed_latency,
            fatal_moves=parsed_fatal,
            avoidable_fatal_moves=parsed_avoidable,
            search_stats=search_stats,
            agent_diagnostics=dict(diagnostics),
        ),
        captured,
    )


def _expected_seats(layout: str, game_in_pair: int) -> list[str]:
    if layout == "copies":
        return (
            ["candidate", "baseline", "candidate", "baseline"]
            if game_in_pair == 0
            else ["baseline", "candidate", "baseline", "candidate"]
        )
    if layout == "solo-pair":
        return (
            ["candidate", "baseline", "baseline", "baseline"]
            if game_in_pair == 0
            else ["baseline", "candidate", "candidate", "candidate"]
        )
    raise ReportValidationError(f"unsupported layout: {layout!r}")


def _load_report(
    path: Path,
    *,
    allow_legacy_provenance: bool = False,
) -> _LoadedReport:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ReportValidationError(f"input report does not exist: {resolved}")
    try:
        payload = resolved.read_bytes()
        raw = json.loads(
            payload,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReportValidationError(f"cannot read {resolved}: {exc}") from exc
    raw = _require_dict(raw, str(resolved))
    context = resolved.name

    schema_version = raw.get("schema_version")
    report_type = raw.get("report_type")
    provenance_value = raw.get("evaluation_provenance")
    legacy_provenance = (
        schema_version is None
        and report_type is None
        and provenance_value is None
    )
    if legacy_provenance:
        if not allow_legacy_provenance:
            raise ReportValidationError(
                f"{context} is a legacy report without evaluator/metric/Hisss "
                "provenance; pass --allow-legacy-provenance to combine it "
                "explicitly"
            )
        report_provenance = None
        hisss_version = _legacy_hisss_version(raw)
    else:
        if schema_version != REPORT_SCHEMA_VERSION or report_type != REPORT_TYPE:
            raise ReportValidationError(
                f"{context} has unsupported report schema/type "
                f"{schema_version!r}/{report_type!r}"
            )
        report_provenance = _validated_evaluation_provenance(
            provenance_value,
            f"{context}.evaluation_provenance",
        )
        hisss_version = report_provenance["hisss"]["version"]

    layout = raw.get("layout")
    if not isinstance(layout, str):
        raise ReportValidationError(f"{context}.layout is missing")
    _expected_seats(layout, 0)
    _require_int(raw.get("max_turns"), f"{context}.max_turns", minimum=1)
    if not isinstance(raw.get("match_rng"), str) or not raw["match_rng"]:
        raise ReportValidationError(f"{context}.match_rng is missing")
    report_seed = _require_int(raw.get("seed"), f"{context}.seed")

    identities = {
        role: _agent_identity(raw, role, context) for role in ROLES
    }
    identity_fingerprints = {
        role: _identity_fingerprints(identity) for role, identity in identities.items()
    }

    workers = raw.get("workers")
    if workers is not None:
        _require_int(workers, f"{context}.workers", minimum=1)
    wall_seconds = raw.get("wall_seconds")
    if wall_seconds is not None:
        _finite_number(wall_seconds, f"{context}.wall_seconds", minimum=0.0)
    runtime = raw.get("runtime")
    _canonical_json(runtime)
    if report_provenance is not None:
        runtime_dict = _require_dict(runtime, f"{context}.runtime")
        runtime_hisss = _require_dict(
            runtime_dict.get("hisss"),
            f"{context}.runtime.hisss",
        )
        if _canonical_json(runtime_hisss) != _canonical_json(
            report_provenance["hisss"]
        ):
            raise ReportValidationError(
                f"{context}.runtime.Hisss disagrees with evaluation provenance"
            )

    summary = _require_dict(raw.get("summary"), f"{context}.summary")
    declared_pairs = _require_int(
        summary.get("pairs"), f"{context}.summary.pairs", minimum=1
    )
    declared_games = _require_int(
        summary.get("games"), f"{context}.summary.games", minimum=1
    )
    match_values = raw.get("matches")
    if not isinstance(match_values, list):
        raise ReportValidationError(f"{context}.matches must be a list")
    if declared_games != len(match_values) or declared_games != 2 * declared_pairs:
        raise ReportValidationError(
            f"{context} is partial: summary declares {declared_pairs} pairs/"
            f"{declared_games} games, file contains {len(match_values)} matches"
        )

    matches: list[MatchResult] = []
    captured_flags: list[bool] = []
    capture_by_game: dict[tuple[int, int], bool] = {}
    for index, value in enumerate(match_values):
        match, captured = _parse_match(value, f"{context}.matches[{index}]")
        matches.append(match)
        captured_flags.append(captured)
        capture_by_game[(match.pair, match.game_in_pair)] = captured

    grouped: dict[int, list[MatchResult]] = {}
    for match in matches:
        grouped.setdefault(match.pair, []).append(match)
    if sorted(grouped) != list(range(declared_pairs)):
        raise ReportValidationError(
            f"{context} pair IDs must be contiguous 0..{declared_pairs - 1}"
        )

    pair_seeds: list[int] = []
    ensemble_pairs = 0
    for pair in range(declared_pairs):
        pair_matches = grouped[pair]
        if len(pair_matches) != 2 or {
            match.game_in_pair for match in pair_matches
        } != {0, 1}:
            raise ReportValidationError(
                f"{context} pair {pair} is incomplete or has duplicate games"
            )
        seeds = {match.seed for match in pair_matches}
        if seeds != {report_seed + pair}:
            raise ReportValidationError(
                f"{context} pair {pair} seed mismatch: {sorted(seeds)}"
            )
        pair_seeds.append(report_seed + pair)
        pair_capture_flags = [
            capture_by_game[(match.pair, match.game_in_pair)]
            for match in pair_matches
        ]
        ensemble_pairs += int(all(pair_capture_flags))
        for match in pair_matches:
            expected_seats = _expected_seats(layout, match.game_in_pair)
            if match.seat_kinds != expected_seats:
                raise ReportValidationError(
                    f"{context} pair {pair}/game {match.game_in_pair} seat layout "
                    "does not match report layout"
                )
            for role in ROLES:
                match_identity = _normalized_diagnostics(
                    match.agent_diagnostics.get(role),
                    f"{context} pair {pair} diagnostics.{role}",
                )
                if _canonical_json(match_identity) != _canonical_json(
                    identities[role]["diagnostics"]
                ):
                    raise ReportValidationError(
                        f"{context} pair {pair} {role} diagnostics disagree with "
                        "the effective report identity"
                    )

    return _LoadedReport(
        path=resolved,
        sha256=hashlib.sha256(payload).hexdigest(),
        raw=raw,
        identities=identities,
        identity_fingerprints=identity_fingerprints,
        matches=matches,
        pair_seeds=pair_seeds,
        ensemble_metrics_captured=all(captured_flags),
        ensemble_metrics_matches=sum(captured_flags),
        ensemble_metrics_pairs=ensemble_pairs,
        evaluation_provenance=report_provenance,
        legacy_provenance=legacy_provenance,
        hisss_version=hisss_version,
    )


def _default_bootstrap_seed(reports: Sequence[_LoadedReport]) -> int:
    digest = hashlib.sha256()
    for report in reports:
        digest.update(bytes.fromhex(report.sha256))
    return int.from_bytes(digest.digest()[:8], "big") & 0x7FFFFFFF


def _validate_combined_summary(summary: dict, *, pairs: int, games: int) -> None:
    if summary.get("pairs") != pairs or summary.get("games") != games:
        raise RuntimeError("summarize returned inconsistent pair/game counts")
    _finite_number(
        summary.get("paired_normalized_score_delta"),
        "combined summary paired_normalized_score_delta",
    )
    ci = summary.get("paired_delta_ci95")
    if pairs < 2:
        if ci is not None:
            raise RuntimeError("summarize returned a CI for fewer than two pairs")
        return
    if not isinstance(ci, list) or len(ci) != 2:
        raise RuntimeError("summarize did not return a two-sided paired CI")
    lower = _finite_number(ci[0], "combined summary CI lower")
    upper = _finite_number(ci[1], "combined summary CI upper")
    if lower > upper:
        raise RuntimeError("summarize returned an inverted paired CI")


def combine_reports(
    inputs: Sequence[str | Path],
    *,
    bootstrap_seed: int | None = None,
    allow_legacy_provenance: bool = False,
) -> dict[str, object]:
    if len(inputs) < 2:
        raise ReportValidationError("at least two input reports are required")
    if bootstrap_seed is not None and (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ReportValidationError("bootstrap_seed must be a non-negative integer")
    reports = [
        _load_report(
            Path(path),
            allow_legacy_provenance=allow_legacy_provenance,
        )
        for path in inputs
    ]
    if len({report.sha256 for report in reports}) != len(reports):
        raise ReportValidationError(
            "the same report content was supplied more than once"
        )

    current_reports = [
        report for report in reports if report.evaluation_provenance is not None
    ]
    if current_reports:
        reference_signature = _provenance_signature(
            current_reports[0].evaluation_provenance
        )
        for report in current_reports[1:]:
            signature = _provenance_signature(report.evaluation_provenance)
            if (
                signature["evaluator_code_sha256"]
                != reference_signature["evaluator_code_sha256"]
            ):
                raise ReportValidationError(
                    "input reports have different evaluator code fingerprints"
                )
            if (
                signature["metric_semantics_sha256"]
                != reference_signature["metric_semantics_sha256"]
                or signature["metric_semantics_version"]
                != reference_signature["metric_semantics_version"]
            ):
                raise ReportValidationError(
                    "input reports have different metric semantics fingerprints"
                )
            if (
                signature["hisss_version"] != reference_signature["hisss_version"]
                or signature["hisss_implementation_sha256"]
                != reference_signature["hisss_implementation_sha256"]
            ):
                raise ReportValidationError(
                    "input reports have different Hisss runtime fingerprints"
                )

    known_hisss_versions = {
        report.hisss_version for report in reports if report.hisss_version is not None
    }
    if len(known_hisss_versions) > 1:
        raise ReportValidationError(
            "input reports have different known Hisss versions"
        )

    first = reports[0]
    for report in reports[1:]:
        for field in ("layout", "max_turns", "match_rng"):
            if report.raw.get(field) != first.raw.get(field):
                raise ReportValidationError(
                    f"input reports disagree on {field}: "
                    f"{first.raw.get(field)!r} != {report.raw.get(field)!r}"
                )
        for role in ROLES:
            if (
                report.identity_fingerprints[role]
                != first.identity_fingerprints[role]
            ):
                raise ReportValidationError(
                    f"input reports have different effective {role} "
                    "model/code/config identities"
                )

    seen_seeds: dict[int, Path] = {}
    for report in reports:
        for seed in report.pair_seeds:
            if seed in seen_seeds:
                raise ReportValidationError(
                    f"pair seed {seed} occurs in both {seen_seeds[seed]} "
                    f"and {report.path}"
                )
            seen_seeds[seed] = report.path

    combined_matches: list[MatchResult] = []
    provenance = []
    next_pair = 0
    for report in reports:
        pair_map = {
            original: next_pair + offset
            for offset, original in enumerate(sorted({m.pair for m in report.matches}))
        }
        for match in sorted(
            report.matches,
            key=lambda item: (item.pair, item.game_in_pair),
        ):
            combined_matches.append(replace(match, pair=pair_map[match.pair]))
        combined_ids = sorted(pair_map.values())
        provenance.append(
            {
                "path": str(report.path),
                "sha256": report.sha256,
                "seed": report.raw["seed"],
                "pair_seeds": report.pair_seeds,
                "pairs": len(pair_map),
                "games": len(report.matches),
                "original_pair_ids": sorted(pair_map),
                "combined_pair_ids": combined_ids,
                "workers": report.raw.get("workers"),
                "wall_seconds": report.raw.get("wall_seconds"),
                "runtime": report.raw.get("runtime"),
                "candidate": report.raw["candidate"],
                "baseline": report.raw["baseline"],
                "effective_agent_fingerprints": report.identity_fingerprints,
                "evaluation_provenance": report.evaluation_provenance,
                "legacy_provenance": report.legacy_provenance,
                "observed_hisss_version": report.hisss_version,
                "ensemble_search_metrics_captured": (
                    report.ensemble_metrics_captured
                ),
                "matches_with_ensemble_search_metrics": (
                    report.ensemble_metrics_matches
                ),
                "pairs_with_ensemble_search_metrics": (
                    report.ensemble_metrics_pairs
                ),
            }
        )
        next_pair += len(pair_map)

    if bootstrap_seed is None:
        bootstrap_seed = _default_bootstrap_seed(reports)

    summary = summarize(combined_matches, bootstrap_seed)
    _validate_combined_summary(
        summary,
        pairs=next_pair,
        games=len(combined_matches),
    )
    captured_inputs = sum(report.ensemble_metrics_captured for report in reports)
    captured_matches = sum(report.ensemble_metrics_matches for report in reports)
    captured_pairs = sum(report.ensemble_metrics_pairs for report in reports)
    effective_diagnostics = {
        role: dict(first.identities[role]["diagnostics"]) for role in ROLES
    }
    legacy_reports = [report for report in reports if report.legacy_provenance]
    known_input_provenance = (
        current_reports[0].evaluation_provenance if current_reports else None
    )
    compatibility_limitations = []
    if legacy_reports:
        compatibility_limitations.append(
            "Legacy inputs lack evaluator, metric, and complete Hisss "
            "implementation fingerprints. Their placement scores passed "
            "current structural invariants, but metric semantics (especially "
            "known-fatal enemy-tail handling) and native simulator identity "
            "cannot be proven identical; aggregated rates are descriptive only."
        )
    return {
        "schema_version": COMBINED_REPORT_SCHEMA_VERSION,
        "report_type": "combined-evaluate-inference",
        "combiner_provenance": {
            "report_schema": {
                "name": "combined-evaluate-inference",
                "version": COMBINED_REPORT_SCHEMA_VERSION,
            },
            "summarizer": evaluation_provenance(),
        },
        "input_evaluation_provenance": {
            "complete": not legacy_reports,
            "known_provenance": known_input_provenance,
            "known_provenance_sha256": (
                _object_fingerprint(known_input_provenance)
                if known_input_provenance is not None
                else None
            ),
            "legacy_input_sha256": [report.sha256 for report in legacy_reports],
        },
        "compatibility": {
            "strict": not legacy_reports,
            "legacy_override_used": bool(legacy_reports),
            "limitations": compatibility_limitations,
        },
        "candidate": first.raw["candidate"],
        "baseline": first.raw["baseline"],
        "layout": first.raw["layout"],
        "max_turns": first.raw["max_turns"],
        "match_rng": first.raw["match_rng"],
        "bootstrap_seed": bootstrap_seed,
        "bootstrap": {
            "seed": bootstrap_seed,
            "unit": "complete-pair",
            "implementation": "evaluate_inference.summarize",
            "ci95_available": summary["paired_delta_ci95"] is not None,
        },
        "effective_agent_diagnostics": effective_diagnostics,
        "effective_agent_fingerprints": first.identity_fingerprints,
        "effective_identity_sha256": {
            role: fingerprints["identity_sha256"]
            for role, fingerprints in first.identity_fingerprints.items()
        },
        "input_seeds": [report.raw["seed"] for report in reports],
        "pair_seeds": sorted(seen_seeds),
        "provenance": provenance,
        "search_metric_coverage": {
            "ensemble_wave_fields": {
                "inputs_with_metrics": captured_inputs,
                "inputs_total": len(reports),
                "pairs_with_metrics": captured_pairs,
                "pairs_total": next_pair,
                "matches_with_metrics": captured_matches,
                "matches_total": len(combined_matches),
                "complete": captured_inputs == len(reports),
                "note": (
                    "Wave-metric numerators retain fields captured by each "
                    "source evaluator. Missing legacy fields contribute "
                    "compatibility zeros, not reconstructed measurements; combined "
                    "per-move rates therefore use all moves as their denominator "
                    "and are lower bounds when coverage is incomplete."
                ),
            }
        },
        "summary": summary,
        "matches": [asdict(match) for match in combined_matches],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine compatible complete evaluate_inference JSON reports."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument(
        "--allow-legacy-provenance",
        action="store_true",
        help=(
            "Explicitly accept old reports without evaluator/metric/Hisss "
            "fingerprints; the combined report records the resulting caveat."
        ),
    )
    args = parser.parse_args(argv)
    if len(args.inputs) < 2:
        parser.error("at least two input reports are required")
    input_paths = {path.resolve() for path in args.inputs}
    if args.output.resolve() in input_paths:
        parser.error("--output must not overwrite an input report")
    if args.bootstrap_seed is not None and args.bootstrap_seed < 0:
        parser.error("--bootstrap-seed must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    combined = combine_reports(
        args.inputs,
        bootstrap_seed=args.bootstrap_seed,
        allow_legacy_provenance=args.allow_legacy_provenance,
    )
    rendered = json.dumps(combined, allow_nan=False, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps(combined["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
