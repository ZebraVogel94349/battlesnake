from __future__ import annotations

import hashlib
import math
import sys

import pytest

import evaluate_inference as evaluation
import ppo3
from battlesnake_types import Direction, GameState
from evaluate_inference import AgentSpec, MatchResult


def _search_stats(
    *,
    moves: float = 0.0,
    deadline_hits: float = 0.0,
    mcts_choices: float = 0.0,
    insufficient_searches: float = 0.0,
    max_depth: float = 0.0,
    ensemble_moves: float = 0.0,
    waves_started: float = 0.0,
    waves_completed: float = 0.0,
    waves_discarded: float = 0.0,
    voting_iterations: float = 0.0,
    discarded_iterations: float = 0.0,
    stopped_for_wave_guard: float = 0.0,
    decision_reason_counts: dict[str, float] | None = None,
) -> dict[str, object]:
    return {
        "moves": moves,
        "iterations": 5.0 * moves,
        "nodes": 3.0 * moves,
        "depth": 2.0 * moves,
        "max_depth": max_depth,
        "elapsed_ms": 10.0 * moves,
        "deadline_hits": deadline_hits,
        "errors": 0.0,
        "mcts_choices": mcts_choices,
        "insufficient_searches": insufficient_searches,
        "policy_changes": 0.0,
        "ensemble_moves": ensemble_moves,
        "waves_started": waves_started,
        "waves_completed": waves_completed,
        "waves_discarded": waves_discarded,
        "voting_iterations": voting_iterations,
        "discarded_iterations": discarded_iterations,
        "stopped_for_wave_guard": stopped_for_wave_guard,
        "decision_reason_counts": decision_reason_counts or {},
    }


def _match(
    *,
    pair: int,
    game: int,
    points: list[float],
    candidate_fatal: int = 0,
    candidate_avoidable: int = 0,
    candidate_search: dict[str, object] | None = None,
) -> MatchResult:
    seat_kinds = (
        ["candidate", "baseline", "candidate", "baseline"]
        if game == 0
        else ["baseline", "candidate", "baseline", "candidate"]
    )
    highest = max(points)
    leaders = [seat for seat, value in enumerate(points) if value == highest]
    timed_out = len(leaders) > 1
    winner = leaders[0] if len(leaders) == 1 else None
    elimination_causes = (
        {}
        if timed_out
        else {
            seat: "test-elimination"
            for seat in range(4)
            if seat != winner
        }
    )
    return MatchResult(
        pair=pair,
        game_in_pair=game,
        seed=100 + pair,
        seat_kinds=seat_kinds,
        points=points,
        turns=20,
        timed_out=timed_out,
        winner=winner,
        elimination_causes=elimination_causes,
        move_latencies_ms={"candidate": [10.0, 20.0], "baseline": [1.0, 2.0]},
        fatal_moves={"candidate": candidate_fatal, "baseline": 0},
        avoidable_fatal_moves={
            "candidate": candidate_avoidable,
            "baseline": 0,
        },
        search_stats={
            "candidate": candidate_search or _search_stats(),
            "baseline": _search_stats(),
        },
        agent_diagnostics={},
    )


def test_top2_credit_is_fractional_at_tie_boundary():
    assert evaluation._top_k_credits([2.0, 0.5, 0.5, 0.0], 2) == [
        1.0,
        0.5,
        0.5,
        0.0,
    ]
    credits = evaluation._top_k_credits([1.0, 1.0, 1.0, 0.0], 2)
    assert credits[:3] == pytest.approx([2.0 / 3.0] * 3)
    assert credits[3] == 0.0


def test_single_pair_ci_is_unavailable_and_rates_are_per_move():
    results = [
        _match(
            pair=0,
            game=0,
            points=[0.75] * 4,
            candidate_fatal=1,
            candidate_avoidable=1,
            candidate_search=_search_stats(
                moves=2,
                deadline_hits=1,
                mcts_choices=1,
                insufficient_searches=1,
                max_depth=4,
            ),
        ),
        _match(
            pair=0,
            game=1,
            points=[0.75] * 4,
            candidate_fatal=1,
            candidate_search=_search_stats(
                moves=2,
                deadline_hits=1,
                mcts_choices=1,
                insufficient_searches=1,
                max_depth=7,
            ),
        ),
    ]

    summary = evaluation.summarize(results, seed=9)

    assert summary["paired_delta_ci95"] is None
    candidate = summary["agents"]["candidate"]
    assert candidate["top2_rate"] == pytest.approx(0.5)
    assert candidate["moves"] == 4
    assert candidate["known_fatal_moves"] == 2
    assert candidate["known_fatal_rate_per_move"] == pytest.approx(0.5)
    assert candidate["avoidable_known_fatal_rate_per_move"] == pytest.approx(0.25)
    assert candidate["search"]["deadline_hit_rate"] == pytest.approx(0.5)
    assert candidate["search"]["mcts_choice_rate"] == pytest.approx(0.5)
    assert candidate["search"]["insufficient_searches"] == 2
    assert candidate["search"]["insufficient_search_rate"] == pytest.approx(0.5)
    assert candidate["search"]["max_depth"] == 7


def test_ensemble_search_diagnostics_aggregate_counts_and_rates():
    results = [
        _match(
            pair=0,
            game=0,
            points=[0.75] * 4,
            candidate_search=_search_stats(
                moves=2,
                ensemble_moves=2,
                waves_started=6,
                waves_completed=4,
                waves_discarded=2,
                voting_iterations=128,
                discarded_iterations=10,
                stopped_for_wave_guard=1,
                decision_reason_counts={
                    "alternative-wave-majority": 1,
                    "no-strict-wave-majority": 1,
                },
            ),
        ),
        _match(
            pair=0,
            game=1,
            points=[0.75] * 4,
            candidate_search=_search_stats(
                moves=2,
                ensemble_moves=1,
                waves_started=2,
                waves_completed=2,
                voting_iterations=64,
                decision_reason_counts={"policy-wave-majority": 1},
            ),
        ),
    ]

    search = evaluation.summarize(results, seed=7)["agents"]["candidate"][
        "search"
    ]

    assert search["moves"] == 4
    assert search["ensemble_moves"] == 3
    assert search["ensemble_move_rate"] == pytest.approx(0.75)
    assert search["waves_started"] == 8
    assert search["waves_started_per_move"] == pytest.approx(2.0)
    assert search["waves_completed"] == 6
    assert search["waves_completed_per_move"] == pytest.approx(1.5)
    assert search["waves_discarded"] == 2
    assert search["waves_discarded_per_move"] == pytest.approx(0.5)
    assert search["voting_iterations"] == 192
    assert search["voting_iterations_per_move"] == pytest.approx(48.0)
    assert search["discarded_iterations"] == 10
    assert search["discarded_iterations_per_move"] == pytest.approx(2.5)
    assert search["stopped_for_wave_guard"] == 1
    assert search["stopped_for_wave_guard_rate"] == pytest.approx(1.0 / 3.0)
    assert search["decision_reason_counts"] == {
        "alternative-wave-majority": 1,
        "no-strict-wave-majority": 1,
        "policy-wave-majority": 1,
    }
    assert search["decision_reason_rates"] == pytest.approx(
        {
            "alternative-wave-majority": 1.0 / 3.0,
            "no-strict-wave-majority": 1.0 / 3.0,
            "policy-wave-majority": 1.0 / 3.0,
        }
    )


def test_old_single_tree_search_stats_default_ensemble_fields_to_zero():
    old_stats = _search_stats(moves=2)
    for key in (
        "ensemble_moves",
        "waves_started",
        "waves_completed",
        "waves_discarded",
        "voting_iterations",
        "discarded_iterations",
        "stopped_for_wave_guard",
        "decision_reason_counts",
    ):
        old_stats.pop(key)
    results = [
        _match(pair=0, game=game, points=[0.75] * 4, candidate_search=old_stats)
        for game in range(2)
    ]

    search = evaluation.summarize(results, seed=7)["agents"]["candidate"][
        "search"
    ]

    assert search["ensemble_moves"] == 0
    assert search["waves_started"] == 0
    assert search["waves_completed"] == 0
    assert search["waves_discarded"] == 0
    assert search["voting_iterations"] == 0
    assert search["discarded_iterations"] == 0
    assert search["stopped_for_wave_guard"] == 0
    assert search["stopped_for_wave_guard_rate"] == 0.0
    assert search["decision_reason_counts"] == {}
    assert search["decision_reason_rates"] == {}


def test_two_pairs_produce_finite_bootstrap_interval():
    results = [
        _match(pair=pair, game=game, points=[2.0, 1.0, 0.0, 0.0])
        for pair in range(2)
        for game in range(2)
    ]
    interval = evaluation.summarize(results, seed=3)["paired_delta_ci95"]
    assert isinstance(interval, list)
    assert len(interval) == 2
    assert all(math.isfinite(value) for value in interval)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -0.01])
def test_agent_spec_rejects_invalid_time_budget(value: float):
    with pytest.raises(ValueError, match="finite and non-negative"):
        evaluation._validate_agent_spec(
            AgentSpec(kind="ppo-mcts", model="model.zip", time_budget_ms=value)
        )


@pytest.mark.parametrize("field", ["max_iterations", "min_iterations"])
def test_agent_spec_rejects_negative_iteration_limits(field: str):
    kwargs = {field: -1}
    with pytest.raises(ValueError, match="must be non-negative"):
        evaluation._validate_agent_spec(
            AgentSpec(kind="ppo-mcts", model="model.zip", **kwargs)
        )


def test_make_agent_forwards_min_iterations(monkeypatch):
    captured = {}

    def fake_agent(model, **kwargs):
        captured.update(model=model, **kwargs)
        return object()

    monkeypatch.setattr(evaluation, "PPOMCTSAgent", fake_agent)
    result = evaluation._make_agent(
        AgentSpec(
            kind="ppo-mcts",
            model="model.zip",
            time_budget_ms=440.0,
            max_iterations=80,
            min_iterations=24,
        )
    )

    assert result is not None
    assert captured == {
        "model": "model.zip",
        "time_budget_ms": 440.0,
        "max_iterations": 80,
        "min_iterations": 24,
    }


def test_ppo3_diagnostics_fingerprint_effective_model_code_and_config(
    monkeypatch,
    tmp_path,
):
    class FakePolicy:
        def set_training_mode(self, training):
            assert training is False

    class FakeModel:
        def __init__(self):
            self.policy = FakePolicy()
            self.loaded = None

        def set_parameters(self, path, *, device):
            self.loaded = (path, device)

    models = []

    def fake_build_model(*, device):
        assert device == "cpu"
        model = FakeModel()
        models.append(model)
        return model

    monkeypatch.setattr(ppo3, "build_model", fake_build_model)
    first_path = tmp_path / "first.zip"
    second_path = tmp_path / "relocated.zip"
    first_path.write_bytes(b"same-model-weights")
    second_path.write_bytes(first_path.read_bytes())

    first = ppo3.PPOAgent3(first_path, space_mask=True)
    relocated = ppo3.PPOAgent3(second_path, space_mask=True)
    no_space = ppo3.PPOAgent3(first_path, space_mask=False)

    first_diagnostics = first.get_diagnostics()
    relocated_diagnostics = relocated.get_diagnostics()
    assert first_diagnostics == relocated_diagnostics
    assert first_diagnostics["model_sha256"] == hashlib.sha256(
        first_path.read_bytes()
    ).hexdigest()
    assert len(first_diagnostics["code_sha256"]) == 64
    assert first_diagnostics["space_mask"] is True
    assert first_diagnostics["device"] == "cpu"
    assert first_diagnostics["observation_shape"] == list(ppo3.OBS_SHAPE)
    assert no_space.get_diagnostics()["space_mask"] is False
    assert models[0].loaded == (str(first_path.resolve()), "cpu")


def test_parse_args_accepts_min_iterations(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_inference.py",
            "--candidate-min-iterations",
            "12",
            "--baseline-min-iterations",
            "8",
        ],
    )
    args = evaluation.parse_args()
    assert args.candidate_min_iterations == 12
    assert args.baseline_min_iterations == 8


def test_parse_args_defaults_to_one_hour_ppo5_vs_ppo4(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate_inference.py"])

    args = evaluation.parse_args()

    assert args.candidate == "ppo5"
    assert args.baseline == "ppo4"
    assert args.pairs is None
    assert args.duration_seconds == 3_600.0
    assert args.layout == "copies"


def test_pairs_switches_to_fixed_count_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_inference.py", "--pairs", "7"],
    )

    args = evaluation.parse_args()

    assert args.pairs == 7


def test_runtime_metadata_contains_reproducibility_versions(monkeypatch):
    monkeypatch.setattr(evaluation.importlib.metadata, "version", lambda name: "1.3.0")
    monkeypatch.setattr(evaluation.torch.cuda, "is_available", lambda: False)

    metadata = evaluation._runtime_metadata()

    assert metadata["python"]["version"]
    assert metadata["torch"]["version"]
    assert metadata["hisss"]["version"] == "1.3.0"
    assert len(metadata["hisss"]["implementation_sha256"]) == 64
    assert len(metadata["hisss"]["native_library_sha256"]) == 64
    assert metadata["hardware"]["logical_cpu_count"]
    assert metadata["hardware"]["cuda_available"] is False
    assert metadata["hardware"]["cuda_devices"] == []


def _tail_state(*, blackout: bool) -> GameState:
    def snake(snake_id, body):
        return {
            "id": snake_id,
            "name": snake_id,
            "length": len(body),
            "latency": "0",
            "squad": None,
            "health": 90,
            "head": {"x": body[0][0], "y": body[0][1]},
            "body": [{"x": x, "y": y} for x, y in body],
            "customizations": {
                "color": [1, 2, 3],
                "head": "default",
                "tail": "default",
            },
        }

    you = snake("you", [(2, 1), (1, 1), (1, 0)])
    enemy = snake("enemy", [(3, 2), (3, 1)])
    return GameState.model_validate(
        {
            "turn": 10,
            "game": {
                "id": "tail-test",
                "source": "test",
                "timeout": 500,
                "ruleset": {
                    "name": "blackout" if blackout else "standard",
                    "version": "v1",
                    "settings": {
                        "foodSpawnChance": 15,
                        "hazardDamagePerTurn": 14,
                        "minimumFood": 1,
                        "viewRadius": 5 if blackout else None,
                        "royale": {"shrinkEveryNTurns": 25},
                        "squad": {
                            "allowBodyCollisions": False,
                            "sharedElimination": False,
                            "sharedHealth": False,
                            "sharedLength": False,
                        },
                    },
                },
            },
            "board": {
                "height": 7,
                "width": 7,
                "food": [],
                "hazards": [],
                "snakes": [you, enemy],
            },
            "you": you,
        }
    )


def test_blackout_enemy_tail_is_never_known_vacating_from_redacted_length():
    assert (
        evaluation._known_fatal_reason(
            _tail_state(blackout=True),
            Direction.RIGHT,
        )
        == "enemy-body"
    )
    assert (
        evaluation._known_fatal_reason(
            _tail_state(blackout=False),
            Direction.RIGHT,
        )
        is None
    )


def test_evaluation_provenance_is_versioned_and_self_consistent():
    provenance = evaluation.evaluation_provenance()

    assert provenance["report_schema"] == {
        "name": evaluation.REPORT_TYPE,
        "version": evaluation.REPORT_SCHEMA_VERSION,
    }
    assert len(provenance["evaluator"]["code_sha256"]) == 64
    semantics = provenance["metric_semantics"]
    assert semantics["version"] == evaluation.METRIC_SEMANTICS_VERSION
    assert semantics["sha256"] == hashlib.sha256(
        evaluation._canonical_json_bytes(semantics["definitions"])
    ).hexdigest()
    assert len(provenance["hisss"]["implementation_sha256"]) == 64


@pytest.mark.parametrize(
    ("points", "winner", "causes", "timed_out"),
    [
        ([2.0, 0.5, 0.5, 0.0], 0, {1: "x", 2: "x", 3: "x"}, False),
        ([1.5, 1.5, 0.0, 0.0], None, {2: "x", 3: "x"}, True),
        ([0.75, 0.75, 0.75, 0.75], None, {}, True),
        ([2.0, 1.0, 0.0, 0.0], None, {0: "x", 1: "x", 2: "x", 3: "x"}, False),
    ],
)
def test_valid_placement_and_winner_invariants(points, winner, causes, timed_out):
    evaluation.validate_placement_outcome(points, winner, causes, timed_out)


@pytest.mark.parametrize(
    ("points", "winner", "causes", "timed_out", "error"),
    [
        ([2.0, 2.0, 0.0, 0.0], 0, {1: "x", 2: "x", 3: "x"}, False, "valid tied"),
        ([2.0, 1.0, 0.0, 0.0], None, {1: "x", 2: "x", 3: "x"}, False, "winner"),
        ([2.0, 1.0, 0.0, 0.0], 0, {1: "x", 2: "x", 3: "x"}, True, "timed_out"),
        ([1.5, 1.5, 0.0, 0.0], None, {1: "x", 3: "x"}, True, "top-scoring"),
    ],
)
def test_rejects_invalid_placement_or_winner_invariants(
    points,
    winner,
    causes,
    timed_out,
    error,
):
    with pytest.raises(ValueError, match=error):
        evaluation.validate_placement_outcome(
            points,
            winner,
            causes,
            timed_out,
        )
