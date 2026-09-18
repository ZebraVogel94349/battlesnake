from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import combine_inference_reports as combiner
import evaluate_inference as evaluation
import ppo3


ENSEMBLE_FIELDS = {
    "ensemble_moves",
    "waves_started",
    "waves_completed",
    "waves_discarded",
    "voting_iterations",
    "discarded_iterations",
    "stopped_for_wave_guard",
    "decision_reason_counts",
}


def _diagnostics(
    *,
    candidate_model_sha: str = "a" * 12,
    candidate_code_sha: str = "b" * 12,
    candidate_particles: int = 24,
    baseline_code_sha: str = "d" * 12,
) -> dict[str, dict[str, object]]:
    return {
        "candidate": {
            "agent": "PPOMCTSAgent",
            "model_sha256": candidate_model_sha,
            "code_sha256": candidate_code_sha,
            "time_budget_ms": 440.0,
            "max_iterations": 32,
            "min_iterations": 24,
            "particles": candidate_particles,
            "horizon": 12,
            "last_search": None,
        },
        "baseline": {
            "agent": "PPOAgent4",
            "model_sha256": "c" * 12,
            "code_sha256": baseline_code_sha,
            "symmetries": 1,
            "safety_search": True,
        },
    }


def _search_stats(*, ensemble: bool, candidate: bool) -> dict[str, object]:
    moves = 4.0 if candidate else 0.0
    result: dict[str, object] = {
        "moves": moves,
        "iterations": 32.0 * moves,
        "nodes": 20.0 * moves,
        "depth": 8.0 * moves,
        "max_depth": 12.0 if candidate else 0.0,
        "elapsed_ms": 100.0 * moves,
        "deadline_hits": 0.0,
        "errors": 0.0,
        "mcts_choices": 2.0 if candidate else 0.0,
        "insufficient_searches": 0.0,
        "policy_changes": 1.0 if candidate else 0.0,
    }
    if ensemble:
        result.update(
            {
                "ensemble_moves": 3.0 if candidate else 0.0,
                "waves_started": 7.0 if candidate else 0.0,
                "waves_completed": 6.0 if candidate else 0.0,
                "waves_discarded": 1.0 if candidate else 0.0,
                "voting_iterations": 192.0 if candidate else 0.0,
                "discarded_iterations": 16.0 if candidate else 0.0,
                "stopped_for_wave_guard": 1.0 if candidate else 0.0,
                "decision_reason_counts": (
                    {"alternative-wave-majority": 3.0} if candidate else {}
                ),
            }
        )
    return result


def _seat_kinds(layout: str, game: int) -> list[str]:
    if layout == "copies":
        return (
            ["candidate", "baseline", "candidate", "baseline"]
            if game == 0
            else ["baseline", "candidate", "baseline", "candidate"]
        )
    return (
        ["candidate", "baseline", "baseline", "baseline"]
        if game == 0
        else ["baseline", "candidate", "candidate", "candidate"]
    )


def _report(
    *,
    seed: int,
    ensemble: bool,
    candidate_model_path: str = "/models/candidate.zip",
    layout: str = "copies",
    max_turns: int = 2000,
    match_rng: str = "turn-reseeded-common-random-numbers-v1",
    candidate_kind: str = "ppo-mcts",
    candidate_model_sha: str = "a" * 12,
    candidate_code_sha: str = "b" * 12,
    candidate_particles: int = 24,
    baseline_code_sha: str = "d" * 12,
    workers: int = 2,
    favorable: bool = True,
    legacy_provenance: bool = False,
    evaluator_code_sha: str | None = None,
    evaluator_component_sha: str | None = None,
    hisss_implementation_sha: str | None = None,
    hisss_native_sha: str | None = None,
    hisss_version: str | None = None,
) -> dict[str, object]:
    diagnostics = _diagnostics(
        candidate_model_sha=candidate_model_sha,
        candidate_code_sha=candidate_code_sha,
        candidate_particles=candidate_particles,
        baseline_code_sha=baseline_code_sha,
    )
    if favorable:
        points = ([2.0, 1.0, 0.0, 0.0], [0.0, 2.0, 1.0, 0.0])
        winners = (0, 1)
    else:
        points = ([1.0, 2.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0])
        winners = (1, 0)

    matches = []
    for game in range(2):
        winner = winners[game]
        matches.append(
            {
                "pair": 0,
                "game_in_pair": game,
                "seed": seed,
                "seat_kinds": _seat_kinds(layout, game),
                "points": list(points[game]),
                "turns": 20 + game,
                "timed_out": False,
                "winner": winner,
                "elimination_causes": {
                    str(seat): "test-elimination"
                    for seat in range(4)
                    if seat != winner
                },
                "move_latencies_ms": {
                    "candidate": [10.0, 20.0],
                    "baseline": [1.0, 2.0],
                },
                "fatal_moves": {"candidate": 1, "baseline": 0},
                "avoidable_fatal_moves": {"candidate": 0, "baseline": 0},
                "search_stats": {
                    "candidate": _search_stats(
                        ensemble=ensemble,
                        candidate=True,
                    ),
                    "baseline": _search_stats(
                        ensemble=ensemble,
                        candidate=False,
                    ),
                },
                "agent_diagnostics": copy.deepcopy(diagnostics),
            }
        )

    provenance = evaluation.evaluation_provenance()
    if evaluator_code_sha is not None:
        provenance["evaluator"]["code_sha256"] = evaluator_code_sha
    if evaluator_component_sha is not None:
        components = provenance["evaluator"]["component_sha256"]
        components["evaluate_inference.py"] = evaluator_component_sha
        provenance["evaluator"]["code_sha256"] = (
            combiner._component_fingerprint(components)
        )
    if hisss_implementation_sha is not None:
        provenance["hisss"][
            "implementation_sha256"
        ] = hisss_implementation_sha
    if hisss_native_sha is not None:
        provenance["hisss"]["native_library_sha256"] = hisss_native_sha
        provenance["hisss"]["implementation_sha256"] = (
            combiner._object_fingerprint(
                {
                    "version": provenance["hisss"]["version"],
                    "python_sources_sha256": provenance["hisss"][
                        "python_sources_sha256"
                    ],
                    "native_library_sha256": hisss_native_sha,
                }
            )
        )
    if hisss_version is not None:
        provenance["hisss"]["version"] = hisss_version

    report = {
        "candidate": {
            "kind": candidate_kind,
            "model": candidate_model_path,
            "time_budget_ms": 440.0,
            "max_iterations": 32,
            "min_iterations": 24,
        },
        "baseline": {
            "kind": "ppo4",
            "model": "/models/baseline.zip",
            "time_budget_ms": None,
            "max_iterations": None,
            "min_iterations": None,
        },
        "layout": layout,
        "seed": seed,
        "max_turns": max_turns,
        "workers": workers,
        "runtime": {
            "python": {"version": f"3.1{workers}"},
            "hardware": {"logical_cpu_count": workers},
            "hisss": copy.deepcopy(provenance["hisss"]),
        },
        "match_rng": match_rng,
        "effective_agent_diagnostics": copy.deepcopy(diagnostics),
        "wall_seconds": float(workers),
        "summary": {"pairs": 1, "games": 2},
        "matches": matches,
    }
    if not legacy_provenance:
        report.update(
            {
                "schema_version": evaluation.REPORT_SCHEMA_VERSION,
                "report_type": evaluation.REPORT_TYPE,
                "evaluation_provenance": provenance,
            }
        )
    return report


def _write(path: Path, report: dict[str, object]) -> Path:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


def _as_ppo3_report(
    report: dict[str, object],
    *,
    kind: str,
    model_path: Path,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    report["candidate"] = {
        "kind": kind,
        "model": str(model_path.resolve()),
        "time_budget_ms": None,
        "max_iterations": None,
        "min_iterations": None,
    }
    report["effective_agent_diagnostics"]["candidate"] = copy.deepcopy(
        diagnostics
    )
    for match in report["matches"]:
        match["agent_diagnostics"]["candidate"] = copy.deepcopy(diagnostics)
        match["search_stats"]["candidate"] = _search_stats(
            ensemble=False,
            candidate=False,
        )
    return report


def test_combines_complete_reports_with_current_summary_and_provenance(tmp_path):
    old_path = _write(
        tmp_path / "old.json",
        _report(
            seed=100,
            ensemble=False,
            candidate_model_path="/old/location/model.zip",
            workers=2,
        ),
    )
    new_path = _write(
        tmp_path / "new.json",
        _report(
            seed=200,
            ensemble=True,
            candidate_model_path="/new/location/same-model.zip",
            workers=7,
            favorable=False,
        ),
    )

    combined = combiner.combine_reports(
        [old_path, new_path],
        bootstrap_seed=73,
    )

    assert combined["summary"]["pairs"] == 2
    assert combined["summary"]["games"] == 4
    assert combined["summary"]["paired_delta_ci95"] is not None
    assert combined["summary"]["paired_normalized_score_delta"] == pytest.approx(0.0)
    assert [match["pair"] for match in combined["matches"]] == [0, 0, 1, 1]
    assert [match["seed"] for match in combined["matches"]] == [100, 100, 200, 200]
    assert combined["input_seeds"] == [100, 200]
    assert combined["pair_seeds"] == [100, 200]
    assert combined["bootstrap"] == {
        "seed": 73,
        "unit": "complete-pair",
        "implementation": "evaluate_inference.summarize",
        "ci95_available": True,
    }
    assert combined["schema_version"] == combiner.COMBINED_REPORT_SCHEMA_VERSION
    assert combined["compatibility"] == {
        "strict": True,
        "legacy_override_used": False,
        "limitations": [],
    }
    assert combined["input_evaluation_provenance"]["complete"] is True
    assert combined["input_evaluation_provenance"]["legacy_input_sha256"] == []

    provenance = combined["provenance"]
    assert [item["workers"] for item in provenance] == [2, 7]
    assert provenance[0]["combined_pair_ids"] == [0]
    assert provenance[1]["combined_pair_ids"] == [1]
    assert provenance[0]["sha256"] == hashlib.sha256(old_path.read_bytes()).hexdigest()
    assert provenance[1]["sha256"] == hashlib.sha256(new_path.read_bytes()).hexdigest()
    assert provenance[0]["candidate"]["model"] != provenance[1]["candidate"]["model"]
    assert (
        provenance[0]["effective_agent_fingerprints"]
        == provenance[1]["effective_agent_fingerprints"]
    )

    coverage = combined["search_metric_coverage"]["ensemble_wave_fields"]
    assert coverage == {
        "inputs_with_metrics": 1,
        "inputs_total": 2,
        "pairs_with_metrics": 1,
        "pairs_total": 2,
        "matches_with_metrics": 2,
        "matches_total": 4,
        "complete": False,
        "note": coverage["note"],
    }
    assert "lower bounds" in coverage["note"]
    for role in ("candidate", "baseline"):
        assert ENSEMBLE_FIELDS <= set(combined["matches"][0]["search_stats"][role])
        assert combined["matches"][0]["search_stats"][role]["waves_started"] == 0.0

    search = combined["summary"]["agents"]["candidate"]["search"]
    assert search["moves"] == 16
    assert search["ensemble_moves"] == 6
    assert search["ensemble_move_rate"] == pytest.approx(6.0 / 16.0)
    assert search["waves_started"] == 14
    assert search["decision_reason_counts"] == {
        "alternative-wave-majority": 6
    }
    assert combined["summary"]["agents"]["candidate"][
        "elimination_causes"
    ]["test-elimination"] == 6
    json.dumps(combined, allow_nan=False)


@pytest.mark.parametrize(
    ("kind", "space_mask"),
    [("ppo3", True), ("ppo3-no-space", False)],
)
def test_new_ppo3_schema_reports_combine_and_reject_identity_drift(
    tmp_path,
    monkeypatch,
    kind,
    space_mask,
):
    class FakePolicy:
        def set_training_mode(self, training):
            assert training is False

    class FakeModel:
        def __init__(self):
            self.policy = FakePolicy()

        def set_parameters(self, path, *, device):
            assert Path(path).is_file()
            assert device == "cpu"

    monkeypatch.setattr(
        ppo3,
        "build_model",
        lambda *, device: FakeModel(),
    )
    first_model = tmp_path / "model-a.zip"
    relocated_model = tmp_path / "model-a-relocated.zip"
    drifted_model = tmp_path / "model-b.zip"
    first_model.write_bytes(b"identical-ppo3-model")
    relocated_model.write_bytes(first_model.read_bytes())
    drifted_model.write_bytes(b"different-ppo3-model")

    first_diagnostics = ppo3.PPOAgent3(
        first_model,
        space_mask=space_mask,
    ).get_diagnostics()
    relocated_diagnostics = ppo3.PPOAgent3(
        relocated_model,
        space_mask=space_mask,
    ).get_diagnostics()
    drifted_diagnostics = ppo3.PPOAgent3(
        drifted_model,
        space_mask=space_mask,
    ).get_diagnostics()
    assert first_diagnostics == relocated_diagnostics

    first = _write(
        tmp_path / "first.json",
        _as_ppo3_report(
            _report(seed=10, ensemble=False),
            kind=kind,
            model_path=first_model,
            diagnostics=first_diagnostics,
        ),
    )
    relocated = _write(
        tmp_path / "relocated.json",
        _as_ppo3_report(
            _report(seed=20, ensemble=False),
            kind=kind,
            model_path=relocated_model,
            diagnostics=relocated_diagnostics,
        ),
    )
    drifted = _write(
        tmp_path / "drifted.json",
        _as_ppo3_report(
            _report(seed=30, ensemble=False),
            kind=kind,
            model_path=drifted_model,
            diagnostics=drifted_diagnostics,
        ),
    )

    combined = combiner.combine_reports([first, relocated])
    assert combined["compatibility"]["strict"] is True
    assert combined["summary"]["pairs"] == 2
    assert combined["effective_agent_diagnostics"]["candidate"] == (
        first_diagnostics
    )
    assert len(
        combined["effective_agent_fingerprints"]["candidate"][
            "config_sha256"
        ]
    ) == 64

    with pytest.raises(
        combiner.ReportValidationError,
        match="different effective candidate",
    ):
        combiner.combine_reports([first, drifted])

    for index, changed_diagnostics in enumerate(
        (
            {
                **first_diagnostics,
                "code_sha256": "f" * 64,
            },
            {
                **first_diagnostics,
                "space_mask": not space_mask,
            },
        ),
        start=1,
    ):
        changed = _write(
            tmp_path / f"identity-drift-{index}.json",
            _as_ppo3_report(
                _report(seed=40 + index, ensemble=False),
                kind=kind,
                model_path=first_model,
                diagnostics=changed_diagnostics,
            ),
        )
        with pytest.raises(
            combiner.ReportValidationError,
            match="different effective candidate",
        ):
            combiner.combine_reports([first, changed])


def test_summary_exactly_matches_evaluator_and_default_bootstrap_is_stable(tmp_path):
    paths = [
        _write(tmp_path / "a.json", _report(seed=10, ensemble=False)),
        _write(
            tmp_path / "b.json",
            _report(seed=20, ensemble=True, favorable=False),
        ),
    ]

    first = combiner.combine_reports(paths)
    second = combiner.combine_reports(paths)
    loaded = [combiner._load_report(path) for path in paths]
    expected_matches = []
    for pair, report in enumerate(loaded):
        for match in report.matches:
            expected_matches.append(
                evaluation.MatchResult(**{**vars(match), "pair": pair})
            )
    expected = evaluation.summarize(expected_matches, first["bootstrap_seed"])

    assert first["bootstrap_seed"] == second["bootstrap_seed"]
    assert first["summary"] == second["summary"] == expected
    assert first["effective_agent_fingerprints"]["candidate"].keys() == {
        "model_sha256",
        "code_sha256",
        "config_sha256",
        "identity_sha256",
    }


@pytest.mark.parametrize("bootstrap_seed", [-1, True, 1.5, "7"])
def test_rejects_invalid_bootstrap_seed(tmp_path, bootstrap_seed):
    paths = [
        _write(tmp_path / "a.json", _report(seed=10, ensemble=False)),
        _write(tmp_path / "b.json", _report(seed=20, ensemble=False)),
    ]

    with pytest.raises(combiner.ReportValidationError, match="bootstrap_seed"):
        combiner.combine_reports(paths, bootstrap_seed=bootstrap_seed)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"layout": "solo-pair"}, "layout"),
        ({"max_turns": 999}, "max_turns"),
        ({"match_rng": "different-rng-v2"}, "match_rng"),
        ({"candidate_kind": "ppo4"}, "different effective candidate"),
        ({"candidate_model_sha": "e" * 12}, "different effective candidate"),
        ({"candidate_code_sha": "f" * 12}, "different effective candidate"),
        ({"candidate_particles": 99}, "different effective candidate"),
        ({"candidate_particles": 24.0}, "different effective candidate"),
        ({"baseline_code_sha": "1" * 12}, "different effective baseline"),
    ],
)
def test_rejects_incompatible_reports(tmp_path, change, error):
    first = _write(tmp_path / "first.json", _report(seed=10, ensemble=False))
    second = _write(
        tmp_path / "second.json",
        _report(seed=20, ensemble=True, **change),
    )

    with pytest.raises(combiner.ReportValidationError, match=error):
        combiner.combine_reports([first, second])


@pytest.mark.parametrize("failure", ["partial", "duplicate-game", "pair-gap"])
def test_rejects_partial_or_unpaired_input(tmp_path, failure):
    first_report = _report(seed=10, ensemble=False)
    if failure == "partial":
        first_report["matches"].pop()
    elif failure == "duplicate-game":
        first_report["matches"][1]["game_in_pair"] = 0
    else:
        for match in first_report["matches"]:
            match["pair"] = 1
            match["seed"] = 11
    paths = [
        _write(tmp_path / "first.json", first_report),
        _write(tmp_path / "second.json", _report(seed=20, ensemble=False)),
    ]

    with pytest.raises(combiner.ReportValidationError):
        combiner.combine_reports(paths)


def test_rejects_internal_diagnostic_drift_duplicate_seeds_and_missing_fingerprint(
    tmp_path,
):
    drifted = _report(seed=10, ensemble=False)
    drifted["matches"][0]["agent_diagnostics"]["candidate"]["particles"] = 48
    paths = [
        _write(tmp_path / "drifted.json", drifted),
        _write(tmp_path / "other.json", _report(seed=20, ensemble=False)),
    ]
    with pytest.raises(combiner.ReportValidationError, match="diagnostics disagree"):
        combiner.combine_reports(paths)

    valid_seed_ten = _write(
        tmp_path / "valid-seed-ten.json",
        _report(seed=10, ensemble=False),
    )
    duplicate_seed = _report(seed=10, ensemble=True, workers=9)
    duplicate_path = _write(tmp_path / "duplicate-seed.json", duplicate_seed)
    with pytest.raises(combiner.ReportValidationError, match="pair seed 10"):
        combiner.combine_reports([valid_seed_ten, duplicate_path])

    missing = _report(seed=30, ensemble=False)
    del missing["effective_agent_diagnostics"]["candidate"]["code_sha256"]
    missing_path = _write(tmp_path / "missing.json", missing)
    with pytest.raises(combiner.ReportValidationError, match="code_sha256"):
        combiner.combine_reports([paths[1], missing_path])


def test_rejects_nonfinite_search_values(tmp_path):
    invalid = _report(seed=10, ensemble=True)
    invalid["matches"][0]["search_stats"]["candidate"]["waves_started"] = float(
        "nan"
    )
    paths = [
        _write(tmp_path / "invalid.json", invalid),
        _write(tmp_path / "valid.json", _report(seed=20, ensemble=True)),
    ]

    with pytest.raises(combiner.ReportValidationError, match="non-finite JSON"):
        combiner.combine_reports(paths)


def test_rejects_duplicate_input_content(tmp_path):
    path = _write(tmp_path / "report.json", _report(seed=10, ensemble=False))

    with pytest.raises(combiner.ReportValidationError, match="same report content"):
        combiner.combine_reports([path, path])


def test_legacy_provenance_requires_explicit_transparent_override(tmp_path):
    legacy = _write(
        tmp_path / "legacy.json",
        _report(seed=10, ensemble=False, legacy_provenance=True),
    )
    current = _write(tmp_path / "current.json", _report(seed=20, ensemble=True))

    with pytest.raises(combiner.ReportValidationError, match="allow-legacy"):
        combiner.combine_reports([legacy, current])

    combined = combiner.combine_reports(
        [legacy, current],
        allow_legacy_provenance=True,
    )

    assert combined["compatibility"]["strict"] is False
    assert combined["compatibility"]["legacy_override_used"] is True
    assert "known-fatal" in combined["compatibility"]["limitations"][0]
    assert "native simulator identity" in (
        combined["compatibility"]["limitations"][0]
    )
    assert combined["input_evaluation_provenance"]["complete"] is False
    assert combined["input_evaluation_provenance"]["legacy_input_sha256"] == [
        hashlib.sha256(legacy.read_bytes()).hexdigest()
    ]
    assert [item["legacy_provenance"] for item in combined["provenance"]] == [
        True,
        False,
    ]


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"evaluator_component_sha": "1" * 64}, "evaluator code"),
        ({"hisss_native_sha": "2" * 64}, "Hisss runtime"),
    ],
)
def test_rejects_known_evaluator_or_hisss_drift_even_with_legacy_flag(
    tmp_path,
    change,
    error,
):
    paths = [
        _write(tmp_path / "a.json", _report(seed=10, ensemble=False)),
        _write(tmp_path / "b.json", _report(seed=20, ensemble=True, **change)),
    ]

    with pytest.raises(combiner.ReportValidationError, match=error):
        combiner.combine_reports(paths, allow_legacy_provenance=True)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"evaluator_code_sha": "1" * 64}, "does not match components"),
        (
            {"hisss_implementation_sha": "2" * 64},
            "does not match its components",
        ),
    ],
)
def test_rejects_internally_inconsistent_provenance(tmp_path, change, error):
    paths = [
        _write(tmp_path / "a.json", _report(seed=10, ensemble=False)),
        _write(tmp_path / "b.json", _report(seed=20, ensemble=True, **change)),
    ]

    with pytest.raises(combiner.ReportValidationError, match=error):
        combiner.combine_reports(paths)


def test_rejects_metric_semantics_drift(tmp_path):
    first = _write(tmp_path / "first.json", _report(seed=10, ensemble=False))
    drifted = _report(seed=20, ensemble=True)
    metrics = drifted["evaluation_provenance"]["metric_semantics"]
    metrics["definitions"]["known_fatal"] = "different semantics"
    metrics["sha256"] = combiner._object_fingerprint(metrics["definitions"])
    second = _write(tmp_path / "second.json", drifted)

    with pytest.raises(combiner.ReportValidationError, match="metric_semantics"):
        combiner.combine_reports([first, second])


def test_legacy_override_still_rejects_known_hisss_version_drift(tmp_path):
    current = _write(tmp_path / "current.json", _report(seed=10, ensemble=False))
    legacy = _write(
        tmp_path / "legacy.json",
        _report(
            seed=20,
            ensemble=False,
            legacy_provenance=True,
            hisss_version="999.0",
        ),
    )

    with pytest.raises(combiner.ReportValidationError, match="Hisss versions"):
        combiner.combine_reports(
            [current, legacy],
            allow_legacy_provenance=True,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda match: match.update(points=[2.0, 2.0, 0.0, 0.0]),
        lambda match: match.update(winner=None),
        lambda match: match["elimination_causes"].update(
            {str(match["winner"]): "winner-marked-dead"}
        ),
        lambda match: match.update(timed_out=True),
    ],
)
def test_rejects_invalid_placement_points_winner_or_survivors(tmp_path, mutation):
    invalid = _report(seed=10, ensemble=False)
    mutation(invalid["matches"][0])
    paths = [
        _write(tmp_path / "invalid.json", invalid),
        _write(tmp_path / "valid.json", _report(seed=20, ensemble=False)),
    ]

    with pytest.raises(combiner.ReportValidationError):
        combiner.combine_reports(paths)


def test_cli_writes_full_combined_json(tmp_path, capsys):
    paths = [
        _write(tmp_path / "a.json", _report(seed=10, ensemble=False)),
        _write(tmp_path / "b.json", _report(seed=20, ensemble=True)),
    ]
    output = tmp_path / "nested" / "combined.json"

    combiner.main(
        [
            *(str(path) for path in paths),
            "--output",
            str(output),
            "--bootstrap-seed",
            "123",
        ]
    )

    written = json.loads(output.read_text())
    assert written["bootstrap_seed"] == 123
    assert written["summary"]["pairs"] == 2
    assert len(written["matches"]) == 4
    assert "paired_normalized_score_delta" in capsys.readouterr().out


def test_cli_legacy_override_is_recorded(tmp_path):
    paths = [
        _write(
            tmp_path / "legacy.json",
            _report(seed=10, ensemble=False, legacy_provenance=True),
        ),
        _write(tmp_path / "current.json", _report(seed=20, ensemble=True)),
    ]
    output = tmp_path / "combined-legacy.json"

    combiner.main(
        [
            *(str(path) for path in paths),
            "--output",
            str(output),
            "--allow-legacy-provenance",
        ]
    )

    written = json.loads(output.read_text())
    assert written["compatibility"]["legacy_override_used"] is True
    assert written["input_evaluation_provenance"]["complete"] is False
