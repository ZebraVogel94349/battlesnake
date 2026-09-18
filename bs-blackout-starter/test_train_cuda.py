import copy
import tempfile
import unittest
from types import SimpleNamespace

import torch

import train_cuda as tc


class TestCudaTrainingHelpers(unittest.TestCase):
    def test_resume_does_not_repeat_bootstrap_evaluations(self):
        self.assertTrue(tc.should_run_evaluation(1, 0, 250))
        self.assertTrue(tc.should_run_evaluation(10, 0, 250))
        self.assertTrue(tc.should_run_evaluation(20, 0, 250))
        self.assertFalse(tc.should_run_evaluation(1, 5250, 250))
        self.assertFalse(tc.should_run_evaluation(10, 5250, 250))
        self.assertFalse(tc.should_run_evaluation(20, 5250, 250))
        self.assertTrue(tc.should_run_evaluation(250, 5250, 250))
        self.assertFalse(tc.should_run_evaluation(250, 5250, 0))

    def test_cuda_all_seat_reset_validation_rejects_unwritten_observations(self):
        obs = torch.zeros((2, 4, *tc.OBS_SHAPE))
        legal = torch.ones((2, 4, 4), dtype=torch.bool)
        obs[:, :, 2, tc.OBS_CENTER, tc.OBS_CENTER] = 0.3
        obs[:, :, 4, tc.OBS_CENTER, tc.OBS_CENTER] = 1.0
        tc.validate_cuda_all_seat_reset(obs, legal, context="unit test")

        obs[0, 2] = 0.0
        with self.assertRaisesRegex(RuntimeError, "native Hisss extension"):
            tc.validate_cuda_all_seat_reset(obs, legal, context="unit test")

        obs[0, 2] = 0.0
        obs[0, 3] = 0.0
        legal[0, 2:] = False
        tc.validate_cuda_all_seat_reset(
            obs,
            legal,
            context="duel unit test",
            allow_duels=True,
        )

    def test_zero_sum_nash_finds_hard_opponent_and_rps_mix(self):
        _, hard_mix, value = tc.solve_zero_sum_nash_distribution(
            torch.tensor([[0.90, 0.20], [0.80, 0.10]]),
            iterations=4_000,
        )
        self.assertGreater(hard_mix[1].item(), 0.95)
        self.assertLess(value, 0.25)

        rps = torch.tensor(
            [
                [0.5, 0.0, 1.0],
                [1.0, 0.5, 0.0],
                [0.0, 1.0, 0.5],
            ]
        )
        rows, columns, value = tc.solve_zero_sum_nash_distribution(
            rps, iterations=4_000
        )
        torch.testing.assert_close(rows, torch.full((3,), 1 / 3), atol=0.02, rtol=0)
        torch.testing.assert_close(
            columns, torch.full((3,), 1 / 3), atol=0.02, rtol=0
        )
        self.assertAlmostEqual(0.5, value, places=2)

    def test_monotonic_cosine_lr_has_no_restart_jumps(self):
        args = SimpleNamespace(
            lr=2.5e-4,
            lr_floor=1e-5,
            lr_schedule="cosine",
            updates=23_000,
        )
        values = [tc.lr_for_update(args, update) for update in range(1, 23_001)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(args.lr, values[0])
        self.assertAlmostEqual(args.lr_floor, values[-1])

        continued = [
            tc.lr_for_update(args, update, start_lr=1e-5)
            for update in (1, 1_000, 23_000)
        ]
        self.assertEqual([1e-5, 1e-5, 1e-5], continued)

        args.lr_decay_updates = 5_000
        self.assertGreater(tc.lr_for_update(args, 4_999), args.lr_floor)
        self.assertAlmostEqual(args.lr_floor, tc.lr_for_update(args, 5_000))
        self.assertAlmostEqual(args.lr_floor, tc.lr_for_update(args, 23_000))
        saved_lr = tc.lr_for_update(args, 2_000)
        self.assertAlmostEqual(
            tc.lr_for_update(args, 2_001),
            tc.lr_for_update(
                args,
                1,
                start_lr=saved_lr,
                update_offset=2_000,
            ),
        )

    def test_v16_resume_keeps_v15_floor_learning_rate(self):
        args = SimpleNamespace(
            lr=5e-6,
            lr_floor=5e-6,
            lr_schedule="constant",
            updates=23_000,
        )
        for local_update in (1, 250, 23_000):
            self.assertAlmostEqual(
                5e-6,
                tc.lr_for_update(
                    args,
                    local_update,
                    start_lr=5e-6,
                    update_offset=23_000,
                ),
            )

    def test_phase_schedule_origin_survives_warm_start_and_resume(self):
        self.assertEqual(
            0,
            tc.phase_schedule_offset(
                51_250,
                51_250,
                resumed=False,
            ),
        )
        self.assertEqual(
            6_250,
            tc.phase_schedule_offset(
                57_500,
                51_250,
                resumed=True,
            ),
        )
        self.assertEqual(
            57_500,
            tc.phase_schedule_offset(
                57_500,
                None,
                resumed=True,
            ),
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            tc.phase_schedule_offset(50, 51, resumed=True)

    def test_entropy_coefficient_decays_without_changing_legacy_runs(self):
        args = SimpleNamespace(
            ent_coef=0.008,
            ent_coef_final=0.002,
            ent_decay_updates=101,
            updates=1_000,
        )
        values = [tc.entropy_coef_for_update(args, update) for update in range(1, 102)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(0.008, values[0])
        self.assertAlmostEqual(0.002, values[-1])

        args.ent_coef_final = None
        self.assertEqual(0.008, tc.entropy_coef_for_update(args, 10_000))

    def test_anchor_specs_are_named_unique_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            first = f"{tmp_dir}/first.zip"
            second = f"{tmp_dir}/second.pt"
            open(first, "wb").close()
            open(second, "wb").close()
            parsed = tc.parse_anchor_specs(
                [f"old_655m={first}", f"v7-917m={second}"]
            )
            self.assertEqual(["old_655m", "v7-917m"], [item[0] for item in parsed])
            with self.assertRaises(ValueError):
                tc.parse_anchor_specs([f"duplicate={first}", f"duplicate={second}"])
            with self.assertRaises(ValueError):
                tc.parse_anchor_specs([f"invalid label={first}"])

    def test_labeled_scores_are_bounded_unique_and_named(self):
        self.assertEqual(
            {"old_655m": 0.77, "v13": 0.58},
            tc.parse_labeled_scores(
                ["old_655m=0.77", "v13=.58"],
                option_name="test score",
            ),
        )
        for invalid in (["missing"], ["a=1.01"], ["a=nan"], ["bad label=.5"]):
            with self.assertRaises(ValueError):
                tc.parse_labeled_scores(invalid, option_name="test score")
        with self.assertRaises(ValueError):
            tc.parse_labeled_scores(
                ["duplicate=.4", "duplicate=.5"], option_name="test score"
            )

    def test_league_promotion_requires_direct_and_guard_scores(self):
        comparisons = [
            {"games": 100.0, "a_wins": 52.0, "draws": 2.0},
            {"games": 200.0, "a_wins": 104.0, "draws": 4.0},
        ]
        evaluations = [
            {
                "kind": "anchor",
                "label": "old_655m",
                "current_win_rate": 0.78,
                "draw_rate": 0.02,
            },
            {
                "kind": "anchor",
                "label": "v7_917m",
                "current_win_rate": 0.75,
                "draw_rate": 0.02,
            },
        ]
        promote, direct, guard, reason = tc.league_promotion_decision(
            comparisons,
            evaluations,
            threshold=0.525,
            guard_labels=["old_655m", "v7_917m"],
            min_guard_score=0.75,
        )
        self.assertTrue(promote, reason)
        self.assertAlmostEqual(0.53, direct)
        self.assertAlmostEqual(0.76, guard)

        evaluations[1]["current_win_rate"] = 0.72
        promote, _, guard, reason = tc.league_promotion_decision(
            comparisons,
            evaluations,
            threshold=0.525,
            guard_labels=["old_655m", "v7_917m"],
            min_guard_score=0.75,
        )
        self.assertFalse(promote)
        self.assertAlmostEqual(0.73, guard)
        self.assertIn("guard score", reason)

        evaluations[1]["current_win_rate"] = 0.69
        promote, _, _, reason = tc.league_promotion_decision(
            comparisons,
            evaluations,
            threshold=0.525,
            guard_labels=[],
            min_guard_score=0.0,
            guard_thresholds={"old_655m": 0.78, "v7_917m": 0.71},
        )
        self.assertFalse(promote)
        self.assertIn("v7_917m", reason)

        evaluations.append(
            {
                "kind": "heuristic",
                "label": "snake25_duelist",
                "current_win_rate": 0.34,
                "draw_rate": 0.0,
            }
        )
        promote, _, _, reason = tc.league_promotion_decision(
            comparisons,
            evaluations,
            threshold=0.525,
            guard_labels=[],
            min_guard_score=0.0,
            guard_thresholds={"snake25_duelist": 0.35},
        )
        self.assertFalse(promote)
        self.assertIn("snake25_duelist", reason)

    def test_generalization_gate_rejects_cyclic_anchor_tradeoffs(self):
        baseline = {"v18": 0.50, "v20": 0.40}
        balanced = [
            {
                "kind": "anchor",
                "label": "v18",
                "current_win_rate": 0.51,
                "draw_rate": 0.0,
            },
            {
                "kind": "anchor",
                "label": "v20",
                "current_win_rate": 0.405,
                "draw_rate": 0.0,
            },
        ]
        promote, mean_delta, worst_delta, reason = (
            tc.league_generalization_decision(
                balanced,
                baseline,
                min_mean_improvement=0.005,
                max_anchor_regression=0.025,
            )
        )
        self.assertTrue(promote, reason)
        self.assertAlmostEqual(0.0075, mean_delta)
        self.assertAlmostEqual(0.005, worst_delta)

        cyclic = [dict(result) for result in balanced]
        cyclic[0]["current_win_rate"] = 0.56
        cyclic[1]["current_win_rate"] = 0.35
        promote, mean_delta, worst_delta, reason = (
            tc.league_generalization_decision(
                cyclic,
                baseline,
                min_mean_improvement=0.005,
                max_anchor_regression=0.025,
            )
        )
        self.assertFalse(promote)
        self.assertAlmostEqual(0.005, mean_delta)
        self.assertAlmostEqual(-0.05, worst_delta)
        self.assertIn("v20", reason)

    def test_anchor_sampling_prioritizes_the_regressed_policy(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=32,
                hidden_size=256,
                device=device,
                pool_dir=tmp_dir,
                max_checkpoints=4,
                checkpoint_weight=1.0,
                best_weight=0.0,
                hungry_weight=0.0,
                random_weight=0.0,
                latest_probability=0.0,
                seed=91,
                load_existing=False,
                active_checkpoint_limit=4,
                anchor_probability=1.0,
                anchor_targets={"healthy": 0.75, "regressed": 0.75},
                anchor_priority_exponent=2.0,
                anchor_min_weight=0.05,
                anchor_score_ema=0.5,
            )
            model = tc.ActorCritic()
            pool.add_anchor(model, 1, 10, label="healthy")
            pool.add_anchor(model, 1, 10, label="regressed")
            pool.enable_checkpoint_sampling()
            pool.begin_rollout()
            priorities = pool.update_anchor_priorities(
                [
                    {
                        "kind": "anchor",
                        "label": "healthy",
                        "current_win_rate": 0.80,
                        "draw_rate": 0.0,
                    },
                    {
                        "kind": "anchor",
                        "label": "regressed",
                        "current_win_rate": 0.20,
                        "draw_rate": 0.0,
                    },
                ]
            )
            self.assertGreater(
                priorities["regressed"][1], priorities["healthy"][1] * 10.0
            )
            regressed_id = pool.pinned_id_for_label("regressed")
            samples = pool._sample(20_000)
            self.assertGreater(
                (samples == regressed_id).float().mean().item(), 0.88
            )

    def test_uniform_anchor_floor_and_deficit_priority_survive_nash(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=32,
                hidden_size=256,
                device=device,
                pool_dir=tmp_dir,
                max_checkpoints=4,
                checkpoint_weight=0.60,
                best_weight=0.40,
                hungry_weight=0.0,
                random_weight=0.0,
                latest_probability=0.0,
                seed=92,
                load_existing=False,
                active_checkpoint_limit=4,
                anchor_targets={"healthy": 0.75, "regressed": 0.75},
                anchor_uniform_floor=0.30,
                champion_min_weight=0.10,
            )
            model = tc.ActorCritic()
            pool.add_anchor(model, 1, 10, label="healthy")
            pool.add_anchor(model, 1, 10, label="regressed")
            pool.add_anchor(model, 1, 10, label=tc.LEAGUE_CHAMPION_LABEL)
            pool.enable_checkpoint_sampling()
            pool.begin_rollout()
            pool.update_anchor_priorities(
                [
                    {
                        "kind": "anchor",
                        "label": "healthy",
                        "current_win_rate": 0.90,
                        "draw_rate": 0.0,
                    },
                    {
                        "kind": "anchor",
                        "label": "regressed",
                        "current_win_rate": 0.20,
                        "draw_rate": 0.0,
                    },
                ]
            )
            pool.nash_weights = {
                pool.opponent_key(opponent_id): 1.0
                for opponent_id in [*pool.active_ids, tc.POOL_BEST]
            }
            effective = pool.effective_training_weights()
            self.assertGreaterEqual(effective["anchor:healthy"], 0.15 - 1e-6)
            self.assertGreaterEqual(effective["anchor:regressed"], 0.15 - 1e-6)
            self.assertGreaterEqual(
                effective[f"anchor:{tc.LEAGUE_CHAMPION_LABEL}"], 0.10 - 1e-6
            )
            self.assertGreater(
                effective["anchor:regressed"], effective["anchor:healthy"]
            )

    def test_true_duels_force_the_configured_native_opponent(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=32,
                hidden_size=256,
                device=device,
                pool_dir=tmp_dir,
                max_checkpoints=4,
                checkpoint_weight=0.0,
                best_weight=0.5,
                hungry_weight=0.0,
                random_weight=0.0,
                heuristic_weights={"snake25_duelist": 0.5},
                latest_probability=0.0,
                seed=93,
                load_existing=False,
                active_checkpoint_limit=4,
                duel_opponent_label="snake25_duelist",
                duel_opponent_probability=1.0,
            )
            duel_rows = torch.arange(32) % 2 == 0
            pool.resample(torch.ones(32, dtype=torch.bool), duel_rows=duel_rows)
            self.assertTrue(
                (
                    pool.assignments[duel_rows, 0]
                    == tc.POOL_SNAKE25_DUELIST
                ).all().item()
            )
            torch.testing.assert_close(pool.duel_rows, duel_rows)

    def test_copy_lineups_share_policy_and_action_mode_across_opponent_seats(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=4_000,
                hidden_size=256,
                device=device,
                pool_dir=tmp_dir,
                max_checkpoints=4,
                checkpoint_weight=1.0,
                best_weight=0.0,
                hungry_weight=0.0,
                random_weight=0.0,
                latest_probability=0.0,
                seed=931,
                load_existing=False,
                active_checkpoint_limit=4,
                deterministic_probability=0.5,
                copies_probability=0.8,
            )
            model = tc.ActorCritic()
            pool.add_anchor(model, 1, 10, label="first")
            pool.add_anchor(model, 1, 10, label="second")
            pool.enable_checkpoint_sampling()
            pool.begin_rollout()
            done = torch.ones(4_000, dtype=torch.bool)
            duel_rows = torch.arange(4_000) % 10 == 0
            pool.resample(done, duel_rows=duel_rows)

            expected_copy_rate = 0.8 * 0.9
            self.assertAlmostEqual(
                float(pool.copy_rows.float().mean()),
                expected_copy_rate,
                delta=0.03,
            )
            self.assertFalse(pool.copy_rows[duel_rows].any().item())
            copied = pool.assignments[pool.copy_rows]
            self.assertTrue((copied == copied[:, :1]).all().item())
            copied_modes = pool.deterministic_assignments[pool.copy_rows]
            self.assertTrue((copied_modes == copied_modes[:, :1]).all().item())
            self.assertTrue(copied_modes.any().item())
            self.assertTrue((~copied_modes).any().item())

    def test_true_duels_mix_native_and_fixed_neural_opponents(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=4_000,
                hidden_size=256,
                device=device,
                pool_dir=tmp_dir,
                max_checkpoints=4,
                checkpoint_weight=0.0,
                best_weight=0.999,
                hungry_weight=0.0,
                random_weight=0.0,
                heuristic_weights={"snake25_duelist": 0.001},
                latest_probability=0.0,
                seed=94,
                load_existing=False,
                active_checkpoint_limit=4,
                duel_opponent_weights={
                    "snake25_duelist": 0.5,
                    "snake25_bc": 0.3,
                },
            )
            pool.add_anchor(tc.ActorCritic(), 1, 10, label="snake25_bc")
            bc_id = pool.pinned_id_for_label("snake25_bc")
            duel_rows = torch.ones(4_000, dtype=torch.bool)
            pool.resample(torch.ones_like(duel_rows), duel_rows=duel_rows)
            native_rate = float(
                (pool.assignments[:, 0] == tc.POOL_SNAKE25_DUELIST).float().mean()
            )
            clone_rate = float((pool.assignments[:, 0] == bc_id).float().mean())
            self.assertAlmostEqual(native_rate, 0.5, delta=0.03)
            self.assertAlmostEqual(clone_rate, 0.3, delta=0.03)

    def test_nash_sampling_includes_heuristics_and_neural_population(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=16,
                hidden_size=256,
                device=device,
                pool_dir=tmp_dir,
                max_checkpoints=4,
                checkpoint_weight=0.70,
                best_weight=0.15,
                hungry_weight=0.10,
                random_weight=0.05,
                latest_probability=0.0,
                seed=92,
                load_existing=False,
                active_checkpoint_limit=4,
                nash_iterations=4_000,
                nash_exploration=0.04,
            )
            pool.add_anchor(tc.ActorCritic(), 1, 10, label="reference")
            reference_id = pool.pinned_id_for_label("reference")
            pool.enable_checkpoint_sampling()
            pool.begin_rollout()
            results = [
                {
                    "kind": "anchor",
                    "label": "reference",
                    "opponent_key": pool.opponent_key(reference_id),
                    "current_win_rate": 0.8,
                    "draw_rate": 0.0,
                },
                *[
                    {
                        "kind": "heuristic",
                        "label": label,
                        "opponent_key": pool.opponent_key(opponent_id),
                        "current_win_rate": score,
                        "draw_rate": 0.0,
                    }
                    for label, opponent_id, score in (
                        ("best", tc.POOL_BEST, 0.1),
                        ("hungry", tc.POOL_HUNGRY, 0.4),
                        ("random", tc.POOL_RANDOM, 0.9),
                    )
                ],
            ]
            weights = pool.update_nash_distribution(results, update=27_000)
            self.assertGreater(
                weights["heuristic:best"], weights["heuristic:hungry"]
            )
            effective = pool.effective_training_weights()
            self.assertAlmostEqual(sum(effective.values()), 1.0, places=6)
            self.assertGreater(effective["heuristic:best"], 0.75)
            self.assertGreaterEqual(effective["heuristic:hungry"], 0.10)
            self.assertGreaterEqual(effective["heuristic:random"], 0.05)
            samples = pool._sample(20_000)
            self.assertGreater((samples == tc.POOL_BEST).float().mean().item(), 0.74)
            self.assertGreater((samples == tc.POOL_HUNGRY).float().mean().item(), 0.08)
            self.assertGreater((samples == tc.POOL_RANDOM).float().mean().item(), 0.035)
            self.assertTrue(pool.nash_state_path.is_file())
            regressed_results = copy.deepcopy(results)
            regressed_results[1]["current_win_rate"] = 0.95
            pool.update_nash_distribution(regressed_results, update=29_250)
            resumed = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=4,
                hidden_size=256,
                device=device,
                pool_dir=tmp_dir,
                max_checkpoints=4,
                checkpoint_weight=0.70,
                best_weight=0.15,
                hungry_weight=0.10,
                random_weight=0.05,
                latest_probability=0.0,
                seed=92,
                load_existing=True,
                active_checkpoint_limit=4,
                nash_iterations=4_000,
                nash_exploration=0.04,
                max_snapshot_update=27_000,
            )
            self.assertEqual(1, len(resumed.nash_rows))
            self.assertEqual(27_000, resumed.nash_state_update)
            self.assertAlmostEqual(
                weights["heuristic:best"],
                resumed.nash_weights["heuristic:best"],
            )

    def test_nash_guarantees_league_champion_floor_and_decays_stale_scores(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=32,
                hidden_size=256,
                device=device,
                pool_dir=tmp_dir,
                max_checkpoints=4,
                checkpoint_weight=0.70,
                best_weight=0.15,
                hungry_weight=0.10,
                random_weight=0.05,
                latest_probability=0.0,
                seed=101,
                load_existing=False,
                active_checkpoint_limit=4,
                nash_iterations=2_000,
                nash_exploration=0.04,
                nash_score_half_life_updates=100,
                champion_min_weight=0.15,
                focus_label=tc.LEAGUE_CHAMPION_LABEL,
            )
            model = tc.ActorCritic()
            pool.add_anchor(
                model, 0, 0, label=tc.LEAGUE_CHAMPION_LABEL
            )
            pool.add_checkpoint(model, 1, 10)
            pool.enable_checkpoint_sampling()
            pool.begin_rollout()
            champion_id = pool.pinned_id_for_label(tc.LEAGUE_CHAMPION_LABEL)
            champion_key = pool.opponent_key(champion_id)
            pool.update_nash_distribution(
                [
                    {
                        "kind": "anchor",
                        "label": tc.LEAGUE_CHAMPION_LABEL,
                        "opponent_key": champion_key,
                        "current_win_rate": 0.0,
                        "draw_rate": 0.0,
                    },
                    {
                        "kind": "heuristic",
                        "label": "best",
                        "opponent_key": "heuristic:best",
                        "current_win_rate": 0.5,
                        "draw_rate": 0.0,
                    },
                ],
                update=0,
            )
            pool.update_nash_distribution(
                [
                    {
                        "kind": "heuristic",
                        "label": "best",
                        "opponent_key": "heuristic:best",
                        "current_win_rate": 0.5,
                        "draw_rate": 0.0,
                    }
                ],
                update=100,
            )

            self.assertAlmostEqual(
                0.25, pool.nash_rows[-1][champion_key], places=6
            )
            effective = pool.effective_training_weights()
            self.assertGreaterEqual(
                effective[champion_key], 0.15 - 1e-6
            )
            self.assertGreaterEqual(effective["heuristic:best"], 0.15 - 1e-6)
            self.assertGreaterEqual(effective["heuristic:hungry"], 0.10 - 1e-6)
            self.assertGreaterEqual(effective["heuristic:random"], 0.05 - 1e-6)
            self.assertAlmostEqual(sum(effective.values()), 1.0, places=6)

    def test_similar_selfplay_checkpoint_is_rejected_behaviorally(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=4,
                hidden_size=256,
                device=device,
                pool_dir=tmp_dir,
                max_checkpoints=4,
                checkpoint_weight=1.0,
                best_weight=0.0,
                hungry_weight=0.0,
                random_weight=0.0,
                latest_probability=0.0,
                seed=93,
                load_existing=False,
                active_checkpoint_limit=4,
                min_action_disagreement=0.20,
                diversity_probe_size=32,
            )
            up = tc.ActorCritic()
            with torch.no_grad():
                up.actor[-1].weight.zero_()
                up.actor[-1].bias.copy_(torch.tensor([100.0, 0.0, 0.0, 0.0]))
            pool.add_anchor(up, 1, 10, label="up")

            probe = torch.zeros((32, *tc.OBS_SHAPE))
            for x, y in ((14, 13), (14, 15), (13, 14), (15, 14)):
                probe[:, 1, x, y] = 1.0
            probe[:, 2, 14, 14] = 0.3
            probe[:, 4, 14, 14] = 1.0

            self.assertIsNone(
                pool.add_checkpoint(up, 2, 20, probe_obs=probe)
            )
            self.assertEqual(2, pool.last_snapshot_attempt_update)
            self.assertEqual(1, pool.checkpoint_count)

            down = tc.ActorCritic()
            down.load_state_dict(up.state_dict())
            with torch.no_grad():
                down.actor[-1].bias.copy_(
                    torch.tensor([0.0, 100.0, 0.0, 0.0])
                )
            self.assertIsNotNone(
                pool.add_checkpoint(down, 3, 30, probe_obs=probe)
            )
            self.assertEqual(2, pool.checkpoint_count)

    def test_replacing_league_champion_preserves_active_episode(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=4,
                hidden_size=256,
                device=device,
                pool_dir=tmp_dir,
                max_checkpoints=4,
                checkpoint_weight=1.0,
                best_weight=0.0,
                hungry_weight=0.0,
                random_weight=0.0,
                latest_probability=0.25,
                seed=21,
                load_existing=False,
                active_checkpoint_limit=4,
            )
            model = tc.ActorCritic()
            pool.add_anchor(model, 1, 10, label="fixed")
            old_path = pool.add_anchor(
                model, 2, 20, label=tc.LEAGUE_CHAMPION_LABEL
            )
            old_id = pool.pinned_id_for_label(tc.LEAGUE_CHAMPION_LABEL)
            pool.assignments[0, 0] = old_id

            pool.replace_anchor(
                tc.LEAGUE_CHAMPION_LABEL, model, update=3, total_steps=30
            )
            new_id = pool.pinned_id_for_label(tc.LEAGUE_CHAMPION_LABEL)
            self.assertNotEqual(old_id, new_id)
            self.assertIn(old_id, pool.retired_ids)
            self.assertIn(old_id, pool.models)
            self.assertFalse(old_path.exists())
            self.assertEqual(
                ["fixed", tc.LEAGUE_CHAMPION_LABEL],
                [
                    label
                    for kind, label, _, _ in pool.evaluation_candidates(0)
                    if kind == "anchor"
                ],
            )

            pool.assignments[pool.assignments == old_id] = tc.POOL_RANDOM
            pool._collect_retired()
            self.assertNotIn(old_id, pool.retired_ids)
            self.assertNotIn(old_id, pool.models)

    def test_replacing_active_champion_refreshes_sampling_before_model_release(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=8,
                hidden_size=256,
                device=device,
                pool_dir=tmp_dir,
                max_checkpoints=4,
                checkpoint_weight=1.0,
                best_weight=0.0,
                hungry_weight=0.0,
                random_weight=0.0,
                latest_probability=0.25,
                seed=22,
                load_existing=False,
                active_checkpoint_limit=4,
                champion_min_weight=0.20,
            )
            model = tc.ActorCritic()
            pool.add_anchor(model, 1, 10, label="fixed")
            pool.add_anchor(
                model, 2, 20, label=tc.LEAGUE_CHAMPION_LABEL
            )
            old_id = pool.pinned_id_for_label(tc.LEAGUE_CHAMPION_LABEL)
            # Simulate a duplicate left by an interrupted promotion as well as
            # the production failure: the old champion is in the active
            # sampling set but no current episode happens to reference it.
            duplicate_id = pool._register(
                tc.FrozenActor(model),
                None,
                update=2,
                total_steps=20,
                pinned=True,
                label=tc.LEAGUE_CHAMPION_LABEL,
                track_snapshot_update=False,
            )
            pool.enable_checkpoint_sampling()
            pool.begin_rollout(2)
            pool.assignments.fill_(tc.POOL_RANDOM)
            self.assertTrue({old_id, duplicate_id} & set(pool.active_ids))

            pool.replace_anchor(
                tc.LEAGUE_CHAMPION_LABEL, model, update=3, total_steps=30
            )
            new_id = pool.pinned_id_for_label(tc.LEAGUE_CHAMPION_LABEL)
            champion_ids = [
                checkpoint_id
                for checkpoint_id in pool.selectable_ids
                if pool.checkpoint_labels.get(checkpoint_id)
                == tc.LEAGUE_CHAMPION_LABEL
            ]

            self.assertEqual([new_id], champion_ids)
            self.assertIn(new_id, pool.active_ids)
            self.assertNotIn(old_id, pool.active_ids)
            self.assertNotIn(duplicate_id, pool.active_ids)
            self.assertNotIn(old_id, pool.models)
            self.assertNotIn(duplicate_id, pool.models)
            sampled = pool._sample(1_000)
            self.assertTrue(
                all(
                    int(checkpoint_id) in pool.models
                    for checkpoint_id in torch.unique(sampled).tolist()
                )
            )

    def test_external_anchors_are_labeled_pinned_and_do_not_delay_snapshots(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = f"{tmp_dir}/reference.zip"
            tc.export_sb3_recurrent_ppo(tc.ActorCritic().eval(), source_path)
            pool = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=4,
                hidden_size=256,
                device=device,
                pool_dir=f"{tmp_dir}/pool",
                max_checkpoints=5,
                checkpoint_weight=1.0,
                best_weight=0.0,
                hungry_weight=0.0,
                random_weight=0.0,
                latest_probability=0.25,
                seed=9,
                load_existing=False,
                active_checkpoint_limit=4,
            )
            pool.add_external_anchor("old_655m", source_path)
            pool.add_external_anchor("v7_917m", source_path)
            self.assertEqual(-1, pool.last_snapshot_update)

            model = tc.ActorCritic()
            for update in range(1, 5):
                pool.add_checkpoint(model, update=update, total_steps=update * 10)
            candidates = pool.evaluation_candidates(history_limit=2)

            self.assertEqual(5, pool.checkpoint_count)
            self.assertEqual(
                ["old_655m", "v7_917m"],
                [label for kind, label, _, _ in candidates if kind == "anchor"],
            )
            self.assertEqual(
                2,
                sum(kind == "history" for kind, _, _, _ in candidates),
            )

    def test_resume_ignores_population_snapshots_from_future_updates(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=2,
                hidden_size=256,
                device=device,
                pool_dir=tmp_dir,
                max_checkpoints=4,
                checkpoint_weight=1.0,
                best_weight=0.0,
                hungry_weight=0.0,
                random_weight=0.0,
                latest_probability=0.0,
                seed=17,
                load_existing=False,
                active_checkpoint_limit=4,
            )
            model = tc.ActorCritic()
            current = pool.add_checkpoint(model, update=27_000, total_steps=100)
            future = pool.add_checkpoint(model, update=27_100, total_steps=200)
            restored_older = pool.add_checkpoint(
                model, update=26_000, total_steps=50
            )

            resumed = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=2,
                hidden_size=256,
                device=device,
                pool_dir=tmp_dir,
                max_checkpoints=4,
                checkpoint_weight=1.0,
                best_weight=0.0,
                hungry_weight=0.0,
                random_weight=0.0,
                latest_probability=0.0,
                seed=17,
                load_existing=True,
                active_checkpoint_limit=4,
                max_snapshot_update=27_000,
            )

            self.assertEqual(2, resumed.checkpoint_count)
            self.assertEqual(27_000, resumed.last_snapshot_update)
            self.assertIn(current, resumed.paths.values())
            self.assertIn(restored_older, resumed.paths.values())
            self.assertEqual(
                current,
                resumed.paths[resumed.selectable_ids[-1]],
            )
            self.assertFalse(future.exists())
            quarantined = list(
                future.parent.glob("discarded_after_u00027000/*.pt")
            )
            self.assertEqual([future.name], [path.name for path in quarantined])

    def test_evaluation_uses_four_histories_and_all_heuristics(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=4,
                hidden_size=256,
                device=device,
                pool_dir=tmp_dir,
                max_checkpoints=8,
                checkpoint_weight=1.0,
                best_weight=1.0,
                hungry_weight=1.0,
                random_weight=1.0,
                latest_probability=0.0,
                seed=94,
                load_existing=False,
                active_checkpoint_limit=8,
            )
            model = tc.ActorCritic()
            pool.add_anchor(model, 0, 0, label="fixed")
            for update in range(1, 7):
                pool.add_checkpoint(model, update, update * 10)
            hard_support_id = pool.selectable_ids[1]
            pool.nash_weights = {
                pool.opponent_key(hard_support_id): 0.9,
            }
            candidates = pool.evaluation_candidates(history_limit=4)
            history_ids = [
                checkpoint_id
                for kind, _, checkpoint_id, _ in candidates
                if kind == "history"
            ]
            self.assertEqual(
                4, sum(kind == "history" for kind, _, _, _ in candidates)
            )
            self.assertEqual(pool.selectable_ids[-1], history_ids[0])
            self.assertIn(hard_support_id, history_ids)
            self.assertEqual(
                {"best", "hungry", "random"},
                {
                    label
                    for kind, label, _, _ in candidates
                    if kind == "heuristic"
                },
            )
            self.assertEqual(
                5,
                len(pool.last_evaluation_ids),
            )

    def test_log_derived_cuda_heuristics_are_distinct_nash_anchors(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=32,
                hidden_size=256,
                device=device,
                pool_dir=tmp_dir,
                max_checkpoints=4,
                checkpoint_weight=0.50,
                best_weight=0.10,
                hungry_weight=0.05,
                random_weight=0.10,
                real_heuristic_weight=0.25,
                latest_probability=0.0,
                seed=195,
                load_existing=False,
                active_checkpoint_limit=4,
            )
            expected = {
                "forager",
                "hunter",
                "territorial",
                "edge_trapper",
                "survivor",
                "snake25",
                "snake25_interceptor",
                "snake25_denier",
                "snake25_duelist",
            }
            candidates = pool.evaluation_candidates(history_limit=0)
            labels = {
                label
                for kind, label, _, _ in candidates
                if kind == "heuristic"
            }
            self.assertTrue(expected.issubset(labels))
            self.assertEqual(
                len(expected),
                len(
                    {
                        tc.NATIVE_PROFILE_BY_POOL_ID[opponent_id]
                        for label, opponent_id in pool.scripted_opponents
                        if label in expected
                    }
                ),
            )

            samples = pool._sample(40_000)
            for label, opponent_id, _ in tc.LOG_DERIVED_HEURISTICS:
                self.assertGreater(
                    (samples == opponent_id).float().mean().item(),
                    0.020,
                    label,
                )

    def test_exact_cuda_heuristic_weights_can_focus_the_v20_anchors(self):
        device = torch.device("cpu")
        exact_weights = {
            "forager": 0.0025,
            "hunter": 0.12,
            "territorial": 0.0025,
            "edge_trapper": 0.0025,
            "survivor": 0.0025,
            "snake25": 0.12,
            "snake25_interceptor": 0.07,
            "snake25_denier": 0.07,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=64,
                hidden_size=256,
                device=device,
                pool_dir=tmp_dir,
                max_checkpoints=4,
                checkpoint_weight=0.50,
                best_weight=0.10,
                hungry_weight=0.005,
                random_weight=0.005,
                heuristic_weights=exact_weights,
                latest_probability=0.0,
                seed=196,
                load_existing=False,
                active_checkpoint_limit=4,
            )
            self.assertAlmostEqual(0.39, pool.real_heuristic_weight)
            for label, opponent_id, _ in tc.LOG_DERIVED_HEURISTICS:
                self.assertAlmostEqual(
                    exact_weights.get(label, 0.0),
                    pool.category_weights[opponent_id],
                )
            samples = pool._sample(100_000)
            self.assertGreater(
                (samples == tc.POOL_HUNTER).float().mean().item(), 0.11
            )
            self.assertLess(
                (samples == tc.POOL_FORAGER).float().mean().item(), 0.005
            )

    def test_heuristic_weight_parser_rejects_unknown_and_duplicates(self):
        self.assertEqual(
            {"hunter": 0.12, "snake25": 0.10},
            tc.parse_heuristic_weights(["hunter=0.12", "snake25=0.10"]),
        )
        with self.assertRaisesRegex(ValueError, "unknown heuristic"):
            tc.parse_heuristic_weights(["missing=0.1"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            tc.parse_heuristic_weights(["hunter=0.1", "hunter=0.2"])

    def test_zero_weight_scripted_opponents_are_not_in_training_mix(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=16,
                hidden_size=256,
                device=torch.device("cpu"),
                pool_dir=tmp_dir,
                max_checkpoints=4,
                checkpoint_weight=0.55,
                best_weight=0.10,
                hungry_weight=0.0,
                random_weight=0.01,
                heuristic_weights={
                    "hunter": 0.10,
                    "snake25": 0.12,
                    "snake25_duelist": 0.12,
                },
                latest_probability=0.0,
                seed=197,
                load_existing=False,
                active_checkpoint_limit=4,
            )

            self.assertEqual(
                {"best", "random", "hunter", "snake25", "snake25_duelist"},
                {label for label, _ in pool.scripted_opponents},
            )
            self.assertFalse(
                (pool._sample(20_000) == tc.POOL_HUNGRY).any().item()
            )
            self.assertEqual(3, len(pool.native_heuristics))
            for opponent_id, profile_id in pool.native_heuristics:
                self.assertEqual(
                    profile_id,
                    pool.native_profile_lut[-opponent_id].item(),
                )
            self.assertEqual(
                -1,
                pool.native_profile_lut[-tc.POOL_FORAGER].item(),
            )

    def test_focused_opponent_is_active_frequent_and_greedy(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = tc.CudaOpponentPool(
                model_factory=tc.ActorCritic,
                num_envs=64,
                hidden_size=256,
                device=device,
                pool_dir=tmp_dir,
                max_checkpoints=4,
                checkpoint_weight=1.0,
                best_weight=0.0,
                hungry_weight=0.0,
                random_weight=0.0,
                latest_probability=0.0,
                seed=27,
                load_existing=False,
                active_checkpoint_limit=2,
                anchor_probability=0.0,
                focus_label="champion",
                focus_probability=0.6,
                deterministic_probability=0.0,
            )
            model = tc.ActorCritic()
            with torch.no_grad():
                model.actor[-1].weight.zero_()
                model.actor[-1].bias.copy_(torch.tensor([100.0, 0.0, 0.0, 0.0]))
            pool.add_anchor(model, 1, 10, label="champion")
            pool.add_anchor(model, 1, 10, label="guard")
            pool.add_checkpoint(model, 2, 20)
            pool.enable_checkpoint_sampling()
            pool.begin_rollout()

            focus_id = pool.focus_checkpoint_id()
            self.assertIn(focus_id, pool.active_ids)
            samples = pool._sample(20_000)
            self.assertGreater((samples == focus_id).float().mean().item(), 0.58)

            pool.resample(torch.ones(64, dtype=torch.bool))
            focused = pool.assignments == focus_id
            self.assertTrue(focused.any().item())
            self.assertTrue(pool.deterministic_assignments[focused].all().item())
            self.assertFalse(
                pool.deterministic_assignments[~focused].any().item()
            )

            # Frozen policies must use their observable deployment mask, not
            # the privileged server mask. The model strongly prefers UP, but
            # the observation exposes only DOWN as safe.
            pool.deterministic_assignments.fill_(True)
            obs = torch.zeros(64, 3, *tc.OBS_SHAPE)
            obs[:, :, 1, 14, 13] = 1.0
            obs[:, :, 4, 14, 14] = 1.0
            server_legal = torch.ones(64, 3, 4, dtype=torch.bool)
            actions = pool.actions(None, obs, server_legal)
            torch.testing.assert_close(actions, torch.ones_like(actions))

    def test_tensorboard_writer_records_configuration_and_resumes(self):
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        with tempfile.TemporaryDirectory() as log_dir:
            args = SimpleNamespace(
                tensorboard_log_dir=log_dir,
                tensorboard_max_queue=1,
                tensorboard_flush_secs=1,
                resume_path=None,
                seed=7,
            )
            model = tc.ActorCritic()
            writer = tc.create_tensorboard_writer(args, 100, model)
            writer.add_scalar("charts/test", 1.0, 100)
            writer.close()

            args.resume_path = "checkpoint.pt"
            writer = tc.create_tensorboard_writer(args, 100, model)
            writer.add_scalar("charts/test", 2.0, 200)
            writer.close()

            events = EventAccumulator(log_dir)
            events.Reload()
            self.assertIn("run/configuration/text_summary", events.Tags()["tensors"])
            self.assertEqual(
                [(100, 1.0), (200, 2.0)],
                [(event.step, event.value) for event in events.Scalars("charts/test")],
            )

    def test_action_permutation_round_trip(self):
        actions = torch.arange(4)
        restored = tc.actions_cuda_to_model(tc.actions_model_to_cuda(actions))
        torch.testing.assert_close(restored, actions)

    def test_fp32_compute_keeps_environment_observations_in_fp32(self):
        self.assertEqual(
            torch.float32,
            tc.environment_obs_dtype(torch.float16, None),
        )
        self.assertEqual(
            torch.float16,
            tc.environment_obs_dtype(torch.float16, torch.float16),
        )

    def test_observable_mask_uses_only_visible_geometry(self):
        obs = torch.zeros(2, *tc.OBS_SHAPE)
        targets = ((14, 15), (14, 13), (13, 14), (15, 14))
        for x, y in targets:
            obs[:, 1, x, y] = 1.0
        obs[:, 4, 14, 14] = 1.0

        # Row zero: own body blocks UP, wall blocks DOWN, visible enemy blocks
        # RIGHT, and a visible head contests LEFT.
        obs[0, 2, 14, 15] = 0.3
        obs[0, 1, 14, 13] = -1.0
        obs[0, 6, 15, 14] = 1.0
        obs[0, 7, 12, 14] = 1.0
        mask = tc.observable_action_mask(obs)
        torch.testing.assert_close(
            mask[0], torch.tensor([False, False, True, False])
        )

        # Marking the own-body target as tail makes UP hard-legal, so the soft
        # mask can now exclude the contested LEFT move.
        obs[0, 5, 14, 15] = 1.0
        mask = tc.observable_action_mask(obs)
        self.assertTrue(mask[0, 0].item())
        self.assertFalse(mask[0, 2].item())
        self.assertTrue(mask[1].all().item())

    def test_rollout_action_masks_compute_observable_geometry_once(self):
        obs = torch.zeros(3, *tc.OBS_SHAPE)
        server = torch.ones(3, 4, dtype=torch.bool)
        hard = torch.tensor(
            [[True, False, True, False]] * 3,
            dtype=torch.bool,
        )
        conservative = torch.tensor(
            [[False, False, True, False]] * 3,
            dtype=torch.bool,
        )
        calls = 0

        def counted(_obs):
            nonlocal calls
            calls += 1
            return hard, conservative

        policy, use_mask, mobility = tc.rollout_action_masks(
            obs,
            server,
            "observable-hard",
            need_mobility_mask=True,
            observable_masks_fn=counted,
        )
        self.assertEqual(1, calls)
        self.assertTrue(use_mask)
        torch.testing.assert_close(policy, hard)
        torch.testing.assert_close(mobility, conservative)

    def test_observable_mask_allows_only_known_winning_head_to_heads(self):
        obs = torch.zeros(3, *tc.OBS_SHAPE)
        targets = ((14, 15), (14, 13), (13, 14), (15, 14))
        for x, y in targets:
            obs[:, 1, x, y] = 1.0
        obs[:, 4, 14, 14] = 1.0
        obs[:, 2, 14, 14] = torch.tensor([0.6, 0.6, 0.3])

        # A length-three enemy contests LEFT from (12, 14). It is fully
        # visible, so rows zero/two know its complete length. Row one reaches
        # the fog boundary, making its true length unknown.
        enemy_cells = ((12, 14), (11, 14), (11, 15))
        for x, y in enemy_cells:
            obs[:, 6, x, y] = 1.0
        obs[:, 7, 12, 14] = 1.0
        obs[1, 6, 9, 14] = 1.0

        mask = tc.observable_action_mask(obs)
        self.assertTrue(mask[0, 2].item())
        self.assertFalse(mask[1, 2].item())
        self.assertFalse(mask[2, 2].item())
        hard_mask = tc.observable_hard_action_mask(obs)
        self.assertTrue(hard_mask[:, 2].all().item())

    def test_actor_and_critic_parameter_groups_are_disjoint_and_complete(self):
        model = tc.ActorCritic()
        actor, critic = tc.actor_critic_parameter_groups(model)
        actor_ids = {id(parameter) for parameter in actor}
        critic_ids = {id(parameter) for parameter in critic}
        self.assertFalse(actor_ids & critic_ids)
        self.assertEqual(
            {id(parameter) for parameter in model.parameters()},
            actor_ids | critic_ids,
        )

    def test_mobility_potential_penalizes_entering_forced_lines(self):
        open_obs = torch.zeros(1, *tc.OBS_SHAPE)
        constrained_obs = torch.zeros_like(open_obs)
        targets = ((14, 15), (14, 13), (13, 14), (15, 14))
        for x, y in targets:
            open_obs[:, 1, x, y] = 1.0
            constrained_obs[:, 1, x, y] = 1.0
        open_obs[:, 4, 14, 14] = 1.0
        constrained_obs[:, 4, 14, 14] = 1.0
        constrained_obs[:, 1, 14, 15] = -1.0
        constrained_obs[:, 1, 14, 13] = -1.0

        open_potential = tc.observation_potential(open_obs, 0.0, 0.0, 0.03)
        constrained_potential = tc.observation_potential(
            constrained_obs, 0.0, 0.0, 0.03
        )
        self.assertGreater(open_potential.item(), constrained_potential.item())

    def test_potential_shaping_does_not_leak_auto_reset_observation(self):
        obs = torch.zeros(2, *tc.OBS_SHAPE)
        obs[:, 2, 14, 14] = 0.3
        obs[:, 4, 14, 14] = 0.5
        next_a = torch.zeros_like(obs)
        next_b = torch.ones_like(obs) * 20.0
        done = torch.ones(2, dtype=torch.bool)
        reward = torch.tensor([1.0, -1.0])

        shaped_a = tc.potential_shaped_reward(
            reward, obs, next_a, done, 0.99, 0.05, 0.02
        )
        shaped_b = tc.potential_shaped_reward(
            reward, obs, next_b, done, 0.99, 0.05, 0.02
        )
        torch.testing.assert_close(shaped_a, shaped_b)

    def test_tournament_reward_is_two_one_zero_zero(self):
        # Four independent episodes: first, second, third, and fourth.
        alive = torch.full((4,), 4, dtype=torch.long)
        first_step, alive = tc.environment_reward_for_scheme(
            torch.tensor([1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, -1.0]),
            torch.tensor([False, False, False, True]),
            "tournament",
            alive,
        )
        second_step, alive = tc.environment_reward_for_scheme(
            torch.tensor([1.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0, 0.0]),
            torch.tensor([False, True, True, False]),
            "tournament",
            alive,
        )
        final_step, alive = tc.environment_reward_for_scheme(
            torch.tensor([1.0 / 3.0, 0.0, 0.0, 0.0]),
            torch.tensor([True, False, False, False]),
            "tournament",
            alive,
        )
        episode_scores = first_step + second_step + final_step
        torch.testing.assert_close(
            episode_scores, torch.tensor([2.0, 1.0, 0.0, 0.0])
        )
        torch.testing.assert_close(alive, torch.tensor([4, 4, 4, 4]))

    def test_tournament_reward_cancels_guaranteed_second_on_no_winner_draw(self):
        reward, alive = tc.environment_reward_for_scheme(
            torch.tensor([2.0 / 3.0]),
            torch.tensor([False]),
            "tournament",
            torch.tensor([4]),
        )
        draw_reward, next_alive = tc.environment_reward_for_scheme(
            torch.tensor([0.0]),
            torch.tensor([True]),
            "tournament",
            alive,
        )
        self.assertEqual(1.0, reward.item())
        self.assertEqual(-1.0, draw_reward.item())
        self.assertEqual(4, next_alive.item())

    def test_tournament_reward_resets_duels_to_two_alive(self):
        reward, next_alive = tc.environment_reward_for_scheme(
            torch.tensor([1.0 / 3.0]),
            torch.tensor([True]),
            "tournament",
            torch.tensor([2]),
            torch.tensor([2]),
        )
        self.assertEqual(1.0, reward.item())
        self.assertEqual(2, next_alive.item())

    def test_gae_stops_at_episode_boundary(self):
        rewards = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
        dones = torch.tensor([[False], [True], [False], [False]])
        values = torch.tensor([[0.5], [0.6], [0.7], [0.8]])
        next_value = torch.tensor([0.9])
        advantages, _ = tc.compute_gae(
            rewards, dones, values, next_value, 0.99, 0.95
        )

        changed_rewards = rewards.clone()
        changed_rewards[2:] += 100.0
        changed_values = values.clone()
        changed_values[2:] -= 50.0
        changed_advantages, _ = tc.compute_gae(
            changed_rewards,
            dones,
            changed_values,
            torch.tensor([-123.0]),
            0.99,
            0.95,
        )
        torch.testing.assert_close(advantages[:2], changed_advantages[:2])

    def test_actor_and_critic_lstm_gradients_are_independent(self):
        model = tc.ActorCritic()
        obs = torch.randn(2, *tc.OBS_SHAPE)
        states = model.initial_state(2, torch.device("cpu"))
        logits, values, *_ = model(obs, *states)

        logits.sum().backward(retain_graph=True)
        self.assertIsNotNone(model.actor_lstm.weight_ih.grad)
        self.assertIsNone(model.critic_lstm.weight_ih.grad)

        model.zero_grad(set_to_none=True)
        values.sum().backward()
        self.assertIsNone(model.actor_lstm.weight_ih.grad)
        self.assertIsNotNone(model.critic_lstm.weight_ih.grad)

    def test_segmented_lstm_preserves_episode_resets_and_checkpoint_schema(self):
        torch.manual_seed(7)
        model = tc.ActorCritic()
        t_steps, batch = 7, 4
        features = torch.randn(t_steps, batch, model.feature_size)
        initial_h, initial_c = model.initial_actor_state(batch, torch.device("cpu"))
        initial_h.normal_()
        initial_c.normal_()
        dones = torch.tensor(
            [
                [False, False, True, False],
                [True, False, False, False],
                [False, True, False, False],
                [False, False, False, True],
                [True, False, True, False],
                [False, False, False, False],
                [False, True, False, False],
            ]
        )

        h, c = initial_h, initial_c
        reference = []
        for step in range(t_steps):
            h, c = model.actor_lstm(features[step], (h, c))
            reference.append(h)
            not_done = (~dones[step]).float().unsqueeze(-1)
            h = h * not_done
            c = c * not_done

        layout = tc._episode_segments(dones)
        actual = tc._forward_segmented_lstm(
            model.actor_lstm,
            features,
            initial_h,
            initial_c,
            layout,
        )
        torch.testing.assert_close(actual, torch.stack(reference), atol=2e-6, rtol=2e-5)
        critic_h = torch.randn_like(initial_h)
        critic_c = torch.randn_like(initial_c)
        expected_critic = tc._forward_segmented_lstm(
            model.critic_lstm,
            features,
            critic_h,
            critic_c,
            layout,
        )
        paired_actor, paired_critic = tc._forward_segmented_lstm_pair(
            model.actor_lstm,
            model.critic_lstm,
            features,
            features,
            initial_h,
            initial_c,
            critic_h,
            critic_c,
            layout,
        )
        torch.testing.assert_close(paired_actor, actual)
        torch.testing.assert_close(paired_critic, expected_critic)
        self.assertFalse(
            any("sequence_lstm" in key for key in model.state_dict())
        )

    def test_fast_policy_math_matches_categorical(self):
        torch.manual_seed(11)
        logits = torch.randn(5, 4)
        legal = torch.tensor(
            [
                [True, True, False, True],
                [False, True, True, True],
                [True, False, True, True],
                [True, True, True, False],
                [True, False, False, True],
            ]
        )
        reference = tc.masked_categorical(logits, legal)
        log_probs = tc.policy_log_probs(logits, legal, True)
        torch.testing.assert_close(log_probs, reference.logits)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        torch.testing.assert_close(entropy, reference.entropy())

    def test_fused_ppo_loss_components_match_reference_math(self):
        torch.manual_seed(19)
        logits = torch.randn(7, 3, 4, requires_grad=True)
        values = torch.randn(7, 3, requires_grad=True)
        legal = torch.rand(7, 3, 4) > 0.25
        legal[..., 0] = True
        actions = legal.float().argmax(dim=-1)
        old_logprobs = torch.randn(7, 3)
        old_values = torch.randn(7, 3)
        returns = torch.randn(7, 3)
        advantages = torch.randn(7, 3)

        actual = tc.ppo_loss_components(
            logits,
            values,
            legal,
            actions,
            old_logprobs,
            old_values,
            returns,
            advantages,
            True,
            0.2,
            0.1,
            0.01,
            0.5,
        )
        log_probs = tc.policy_log_probs(logits, legal, True)
        new_logprobs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
        logratio = new_logprobs - old_logprobs
        ratio = logratio.exp()
        policy_loss = torch.maximum(
            -advantages * ratio,
            -advantages * ratio.clamp(0.8, 1.2),
        ).mean()
        values_pred = old_values + (values - old_values).clamp(-0.1, 0.1)
        value_loss = torch.nn.functional.mse_loss(values_pred, returns)
        expected = (
            policy_loss + 0.5 * value_loss - 0.01 * entropy,
            policy_loss,
            value_loss,
            entropy,
            ((ratio - 1.0) - logratio).mean(),
            ((ratio - 1.0).abs() > 0.2).float().mean(),
        )
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor)

    def test_compiled_ppo_loss_accepts_scheduled_entropy_without_recompile_limit(self):
        torch._dynamo.reset()
        compiled_loss = torch.compile(
            tc.ppo_loss_components,
            backend="eager",
            dynamic=False,
            fullgraph=True,
        )
        torch.manual_seed(29)
        logits = torch.randn(4, 3, 4)
        values = torch.randn(4, 3)
        legal = torch.ones(4, 3, 4, dtype=torch.bool)
        actions = torch.zeros(4, 3, dtype=torch.long)
        old_logprobs = torch.randn(4, 3)
        old_values = torch.randn(4, 3)
        returns = torch.randn(4, 3)
        advantages = torch.randn(4, 3)

        # More calls than Dynamo's default recompile_limit. A Python float here
        # reproduces the training crash; scalar tensors must share one graph.
        for ent_coef in torch.linspace(0.008, 0.007, 12):
            result = compiled_loss(
                logits,
                values,
                legal,
                actions,
                old_logprobs,
                old_values,
                returns,
                advantages,
                True,
                0.2,
                None,
                ent_coef,
                0.5,
            )
            self.assertTrue(torch.isfinite(result[0]).item())
        torch._dynamo.reset()

    def test_channels_last_preserves_model_outputs_and_state_schema(self):
        torch.manual_seed(23)
        reference = tc.ActorCritic().eval()
        optimized = tc.ActorCritic().eval()
        optimized.load_state_dict(reference.state_dict())
        optimized.enable_channels_last()
        obs = torch.randn(6, *tc.OBS_SHAPE)
        states = reference.initial_state(6, torch.device("cpu"))

        expected = reference(obs, *states)
        actual = optimized(obs, *states)
        for expected_tensor, actual_tensor in zip(expected, actual):
            torch.testing.assert_close(actual_tensor, expected_tensor)
        self.assertEqual(reference.state_dict().keys(), optimized.state_dict().keys())

    def test_sb3_export_load_round_trip(self):
        source = tc.ActorCritic().eval()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = f"{tmp_dir}/policy.zip"
            tc.export_sb3_recurrent_ppo(source, path)
            restored = tc.load_sb3_recurrent_ppo(path, device="cpu")
            warm_started = tc.ActorCritic()
            metadata = tc.load_warm_start(
                path,
                warm_started,
                torch.device("cpu"),
            )

        source_state = source.state_dict()
        restored_state = restored.state_dict()
        self.assertEqual(source_state.keys(), restored_state.keys())
        for key in source_state:
            if not key.startswith("critic_cnn."):
                torch.testing.assert_close(source_state[key], restored_state[key])
                torch.testing.assert_close(
                    source_state[key], warm_started.state_dict()[key]
                )
            torch.testing.assert_close(
                restored_state[key], warm_started.state_dict()[key]
            )
        self.assertEqual((0, 0, "SB3 zip", None), metadata)

    def test_periodic_sb3_exports_are_named_and_retained(self):
        source = tc.ActorCritic().eval()
        with tempfile.TemporaryDirectory() as tmp_dir:
            first = tc.export_periodic_sb3_checkpoint(
                source, tmp_dir, "policy.zip", 100, 1, max_checkpoints=2
            )
            second = tc.export_periodic_sb3_checkpoint(
                source, tmp_dir, "policy.zip", 200, 2, max_checkpoints=2
            )
            third = tc.export_periodic_sb3_checkpoint(
                source, tmp_dir, "policy.zip", 300, 3, max_checkpoints=2
            )
            restored = tc.load_sb3_recurrent_ppo(str(third), device="cpu")

            self.assertEqual("policy_steps_000000000100_u00000001.zip", first.name)
            self.assertFalse(first.exists())
            self.assertTrue(second.exists())
            self.assertTrue(third.exists())
            self.assertFalse(list(first.parent.glob("*_tmp.zip")))
            for key, expected in source.state_dict().items():
                if not key.startswith("critic_cnn."):
                    torch.testing.assert_close(restored.state_dict()[key], expected)

    def test_periodic_training_checkpoints_restore_optimizer_and_are_retained(self):
        torch.manual_seed(33)
        source = tc.ActorCritic()
        optimizer = torch.optim.Adam(source.parameters(), lr=3e-5)
        optimizer.zero_grad(set_to_none=True)
        next(source.parameters()).square().mean().backward()
        optimizer.step()
        args = SimpleNamespace(num_envs=8, n_steps=4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            first = tc.save_periodic_training_checkpoint(
                source, optimizer, tmp_dir, "policy.zip", 100, 1, args,
                max_checkpoints=2,
            )
            second = tc.save_periodic_training_checkpoint(
                source, optimizer, tmp_dir, "policy.zip", 200, 2, args,
                max_checkpoints=2,
            )
            third = tc.save_periodic_training_checkpoint(
                source, optimizer, tmp_dir, "policy.zip", 300, 3, args,
                max_checkpoints=2,
            )
            restored = tc.ActorCritic()
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=9e-4)
            counters = tc.load_training_checkpoint(
                str(third), restored, restored_optimizer, torch.device("cpu")
            )

            self.assertEqual("policy_steps_000000000100_u00000001.pt", first.name)
            self.assertFalse(first.exists())
            self.assertTrue(second.exists())
            self.assertTrue(third.exists())
            self.assertFalse(list(first.parent.glob("*.tmp")))
            self.assertEqual((300, 3), counters)
            self.assertEqual(len(optimizer.state), len(restored_optimizer.state))
            self.assertEqual(
                optimizer.param_groups[0]["lr"],
                restored_optimizer.param_groups[0]["lr"],
            )
            for key, expected in source.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], expected)

    def test_warm_start_torch_checkpoint_loads_only_policy_weights(self):
        torch.manual_seed(31)
        source = tc.ActorCritic()
        target = tc.ActorCritic()
        with torch.no_grad():
            for parameter in target.parameters():
                parameter.zero_()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = f"{tmp_dir}/checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": source.state_dict(),
                    "optimizer_state_dict": {"param_groups": [{"lr": 3e-5}]},
                    "total_steps": 123_456,
                    "update": 17,
                },
                path,
            )
            metadata = tc.load_warm_start(path, target, torch.device("cpu"))

        for key, expected in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], expected)
        self.assertEqual((123_456, 17, "Torch checkpoint", 3e-5), metadata)

    def test_resume_restores_weights_optimizer_and_counters(self):
        torch.manual_seed(37)
        source = tc.ActorCritic()
        source_optimizer = torch.optim.Adam(source.parameters(), lr=1e-3)
        source_optimizer.zero_grad(set_to_none=True)
        next(source.parameters()).square().mean().backward()
        source_optimizer.step()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = f"{tmp_dir}/resume.pt"
            tc.save_training_checkpoint(
                path,
                source,
                source_optimizer,
                total_steps=654_321,
                update=23,
                args=SimpleNamespace(num_envs=8, n_steps=4),
            )
            target = tc.ActorCritic()
            target_optimizer = torch.optim.Adam(target.parameters(), lr=9e-4)
            counters = tc.load_training_checkpoint(
                path,
                target,
                target_optimizer,
                torch.device("cpu"),
            )

        for key, expected in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], expected)
        self.assertEqual((654_321, 23), counters)
        self.assertEqual(len(source_optimizer.state), len(target_optimizer.state))
        self.assertEqual(
            source_optimizer.param_groups[0]["lr"],
            target_optimizer.param_groups[0]["lr"],
        )

    def test_evaluation_layout_rotates_seats(self):
        self.assertEqual(
            (tc.EVAL_MODEL_A, tc.EVAL_MODEL_B, tc.EVAL_MODEL_A, tc.EVAL_MODEL_B),
            tc._evaluation_seat_types(0, "copies"),
        )
        self.assertEqual(
            (tc.EVAL_MODEL_B, tc.EVAL_MODEL_A, tc.EVAL_MODEL_B, tc.EVAL_MODEL_A),
            tc._evaluation_seat_types(1, "copies"),
        )
        duel_layouts = [tc._evaluation_seat_types(i, "duel") for i in range(12)]
        self.assertTrue(all(row.count(tc.EVAL_MODEL_A) == 1 for row in duel_layouts))
        self.assertTrue(all(row.count(tc.EVAL_MODEL_B) == 1 for row in duel_layouts))
        solo_pair = [
            tc._evaluation_seat_types(i, "solo-pair") for i in range(8)
        ]
        self.assertEqual(
            [1, 3, 1, 3, 1, 3, 1, 3],
            [row.count(tc.EVAL_MODEL_A) for row in solo_pair],
        )
        self.assertEqual(
            [0, 0, 1, 1, 2, 2, 3, 3],
            [
                next(
                    index
                    for index, seat_type in enumerate(row)
                    if row.count(seat_type) == 1
                )
                for row in solo_pair
            ],
        )
        self.assertEqual(
            (tc.EVAL_MODEL_A, tc.EVAL_MODEL_B, tc.EVAL_FILLER, tc.EVAL_FILLER),
            tc._evaluation_seat_types(0, "true-duel"),
        )
        self.assertEqual(
            (tc.EVAL_MODEL_B, tc.EVAL_MODEL_A, tc.EVAL_FILLER, tc.EVAL_FILLER),
            tc._evaluation_seat_types(1, "true-duel"),
        )


@unittest.skipUnless(torch.cuda.is_available(), "PyTorch CUDA is not available")
class TestCudaOpponentPool(unittest.TestCase):
    def test_reduce_overhead_mask_outputs_survive_graph_replay(self):
        device = torch.device("cuda")
        obs_a = torch.randn(
            16,
            *tc.OBS_SHAPE,
            dtype=torch.float16,
            device=device,
        )
        obs_b = torch.randn_like(obs_a)
        compiled = tc.compile_observable_action_masks("reduce-overhead")
        first = compiled(obs_a)
        expected = tc.observable_action_masks(obs_a)

        # A CUDA Graph reuses its output storage on this second invocation.
        # The wrapper must return owning clones so the first mask remains valid.
        compiled(obs_b)
        for actual, reference in zip(first, expected):
            torch.testing.assert_close(actual, reference)

    def test_pinned_anchor_survives_eviction_and_is_always_active(self):
        device = torch.device("cuda")
        with tempfile.TemporaryDirectory() as pool_dir:
            pool = tc.CudaOpponentPool(
                model_factory=lambda: tc.ActorCritic().to(device),
                num_envs=8,
                hidden_size=256,
                device=device,
                pool_dir=pool_dir,
                max_checkpoints=3,
                checkpoint_weight=1.0,
                best_weight=0.0,
                hungry_weight=0.0,
                random_weight=0.0,
                latest_probability=0.5,
                seed=13,
                load_existing=False,
                active_checkpoint_limit=2,
            )
            model = tc.ActorCritic().to(device)
            pool.add_anchor(model, update=10, total_steps=100)
            anchor_id = next(iter(pool.pinned_ids))
            for update in range(11, 15):
                pool.add_checkpoint(model, update=update, total_steps=update * 10)
            pool.begin_rollout()
            self.assertEqual(3, pool.checkpoint_count)
            self.assertIn(anchor_id, pool.selectable_ids)
            self.assertIn(anchor_id, pool.active_ids)

    def test_active_checkpoint_subset_rotates_and_keeps_latest(self):
        device = torch.device("cuda")
        with tempfile.TemporaryDirectory() as pool_dir:
            pool = tc.CudaOpponentPool(
                model_factory=lambda: tc.ActorCritic().to(device),
                num_envs=8,
                hidden_size=256,
                device=device,
                pool_dir=pool_dir,
                max_checkpoints=4,
                checkpoint_weight=1.0,
                best_weight=0.0,
                hungry_weight=0.0,
                random_weight=0.0,
                latest_probability=0.5,
                seed=17,
                load_existing=False,
                active_checkpoint_limit=2,
            )
            model = tc.ActorCritic().to(device)
            for update in range(4):
                pool._register(tc.FrozenActor(model), None, update)
            pool.enable_checkpoint_sampling()

            seen = set()
            for _ in range(20):
                pool.begin_rollout()
                self.assertEqual(2, len(pool.active_ids))
                self.assertIn(pool.selectable_ids[-1], pool.active_ids)
                seen.update(pool.active_ids)
            self.assertEqual(set(pool.selectable_ids), seen)

    def test_only_done_environments_are_resampled_and_reset(self):
        device = torch.device("cuda")
        with tempfile.TemporaryDirectory() as pool_dir:
            pool = tc.CudaOpponentPool(
                model_factory=lambda: tc.ActorCritic().to(device),
                num_envs=8,
                hidden_size=256,
                device=device,
                pool_dir=pool_dir,
                max_checkpoints=2,
                checkpoint_weight=0.0,
                best_weight=1.0,
                hungry_weight=1.0,
                random_weight=1.0,
                latest_probability=0.5,
                seed=11,
                load_existing=False,
            )
            before = pool.assignments.clone()
            pool.actor_h.fill_(1.0)
            pool.actor_c.fill_(1.0)
            done = torch.zeros(8, dtype=torch.bool, device=device)
            done[[1, 5]] = True
            pool.resample(done)

            keep = ~done
            torch.testing.assert_close(pool.assignments[keep], before[keep])
            torch.testing.assert_close(
                pool.actor_h[keep], torch.ones_like(pool.actor_h[keep])
            )
            torch.testing.assert_close(
                pool.actor_c[keep], torch.ones_like(pool.actor_c[keep])
            )
            torch.testing.assert_close(
                pool.actor_h[done], torch.zeros_like(pool.actor_h[done])
            )
            torch.testing.assert_close(
                pool.actor_c[done], torch.zeros_like(pool.actor_c[done])
            )

    def test_cuda_policy_comparison_accounts_for_every_game(self):
        device = torch.device("cuda")
        model = tc.ActorCritic().to(device).eval()
        stats = tc.compare_policies_cuda(
            model,
            model,
            games=16,
            num_envs=8,
            max_turns=40,
            seed=19,
            layout="copies",
            deterministic=True,
            device=device,
        )
        self.assertEqual(16.0, stats["games"])
        self.assertEqual(
            16.0,
            stats["a_wins"] + stats["b_wins"] + stats["draws"],
        )
        self.assertGreater(stats["avg_turns"], 0.0)

        heuristic_stats = tc.compare_policies_cuda(
            model,
            None,
            games=16,
            num_envs=8,
            max_turns=40,
            seed=20,
            layout="copies",
            model_b_kind="best",
            deterministic=True,
            device=device,
        )
        self.assertEqual(
            16.0,
            heuristic_stats["a_wins"]
            + heuristic_stats["b_wins"]
            + heuristic_stats["draws"],
        )


if __name__ == "__main__":
    unittest.main()
