import unittest

import numpy as np

import hisss


@unittest.skipUnless(hisss.cuda_available(), "CUDA backend is not available")
class TestCudaBlackoutVecEnv(unittest.TestCase):
    def test_reset_and_step_shapes(self):
        env = hisss.CudaBlackoutVecEnv(16, seed=123)
        try:
            obs, legal = env.reset()
            self.assertEqual((16, 14, 29, 29), obs.shape)
            self.assertEqual(np.float32, obs.dtype)
            self.assertEqual((16, 4), legal.shape)
            self.assertEqual(bool, legal.dtype)
            self.assertTrue(np.all(legal.sum(axis=1) > 0))

            actions = np.zeros((16,), dtype=np.int32)
            step = env.step(actions)
            self.assertEqual((16, 14, 29, 29), step.obs.shape)
            self.assertEqual((16,), step.rewards.shape)
            self.assertEqual((16,), step.done.shape)
            self.assertEqual((16, 4), step.legal_mask.shape)
        finally:
            env.close()

    def test_step_rewards_use_kill_reward_without_living_bonus(self):
        env = hisss.CudaBlackoutVecEnv(64, seed=808)
        allowed = np.array(
            [-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
            dtype=np.float32,
        )
        rng = np.random.default_rng(808)
        try:
            env.reset()
            for _ in range(40):
                step = env.step(rng.integers(0, 4, size=64, dtype=np.int32))
                distance = np.abs(step.rewards[:, None] - allowed[None, :]).min(axis=1)
                self.assertTrue(np.all(distance < 1e-6))
        finally:
            env.close()


@unittest.skipUnless(hisss.cuda_available(), "CUDA backend is not available")
class TestCudaBlackoutTorchVecEnv(unittest.TestCase):
    def test_true_duel_resets_keep_only_two_live_snakes(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA is not available")

        with self.assertRaises(ValueError):
            hisss.CudaBlackoutTorchVecEnv(4, duel_probability=1.01)
        env = hisss.CudaBlackoutTorchVecEnv(
            64,
            seed=41,
            duel_probability=1.0,
        )
        try:
            obs, legal = env.reset_all()
            self.assertTrue(legal[:, :2].any(dim=-1).all().item())
            self.assertFalse(legal[:, 2:].any().item())
            self.assertTrue(
                (obs[:, :2, 2, 14, 14] > 0).all().item()
            )
            self.assertTrue((obs[:, 2:] == 0).all().item())
        finally:
            env.close()

    def test_fp16_all_observations_match_fp32_on_custom_stream(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA is not available")

        fp32_env = hisss.CudaBlackoutTorchVecEnv(
            64, seed=818, obs_dtype=torch.float32
        )
        fp16_env = hisss.CudaBlackoutTorchVecEnv(
            64, seed=818, obs_dtype=torch.float16
        )
        stream = torch.cuda.Stream()
        try:
            with torch.cuda.stream(stream):
                fp32_obs, fp32_legal = fp32_env.reset_all()
                fp16_obs, fp16_legal = fp16_env.reset_all()
                for _ in range(40):
                    actions = fp32_env.best_actions()
                    fp32_step = fp32_env.step_all(actions)
                    fp16_step = fp16_env.step_all(actions)
                    torch.testing.assert_close(
                        fp16_step.obs, fp32_step.obs.half(), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        fp16_step.rewards, fp32_step.rewards, rtol=0, atol=0
                    )
                    torch.testing.assert_close(fp16_step.done, fp32_step.done)
                    torch.testing.assert_close(
                        fp16_step.legal_mask, fp32_step.legal_mask
                    )
            stream.synchronize()
            self.assertEqual(torch.float16, fp16_obs.dtype)
            torch.testing.assert_close(fp16_obs, fp32_obs.half(), rtol=0, atol=0)
            torch.testing.assert_close(fp16_legal, fp32_legal)
        finally:
            fp32_env.close()
            fp16_env.close()

    def test_parallel_all_encoder_matches_serial_snake_zero_encoder(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA is not available")

        num_envs = 64
        serial_env = hisss.CudaBlackoutTorchVecEnv(num_envs, seed=919)
        parallel_env = hisss.CudaBlackoutTorchVecEnv(num_envs, seed=919)
        try:
            serial_obs, serial_legal = serial_env.reset()
            all_obs, all_legal = parallel_env.reset_all()
            torch.testing.assert_close(all_obs[:, 0], serial_obs, rtol=0, atol=0)
            torch.testing.assert_close(all_legal[:, 0], serial_legal)

            for _ in range(50):
                agent_actions = serial_legal.float().argmax(dim=-1).to(torch.int32)
                all_actions = torch.full(
                    (num_envs, 4),
                    -1,
                    dtype=torch.int32,
                    device=serial_obs.device,
                )
                all_actions[:, 0] = agent_actions
                serial_step = serial_env.step(agent_actions)
                all_step = parallel_env.step_all(all_actions)

                torch.testing.assert_close(
                    all_step.obs[:, 0], serial_step.obs, rtol=0, atol=0
                )
                torch.testing.assert_close(all_step.rewards, serial_step.rewards)
                torch.testing.assert_close(all_step.done, serial_step.done)
                torch.testing.assert_close(
                    all_step.legal_mask[:, 0], serial_step.legal_mask
                )
                serial_obs = serial_step.obs
                serial_legal = serial_step.legal_mask
        finally:
            serial_env.close()
            parallel_env.close()

    def test_reset_and_step_return_cuda_tensors(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA is not available")

        env = hisss.CudaBlackoutTorchVecEnv(16, seed=123)
        try:
            obs, legal = env.reset()
            self.assertEqual((16, 14, 29, 29), tuple(obs.shape))
            self.assertEqual((16, 4), tuple(legal.shape))
            self.assertEqual("cuda", obs.device.type)
            self.assertEqual("cuda", legal.device.type)

            actions = torch.zeros((16,), dtype=torch.int32, device=obs.device)
            step = env.step(actions)
            self.assertEqual((16, 14, 29, 29), tuple(step.obs.shape))
            self.assertEqual((16,), tuple(step.rewards.shape))
            self.assertEqual((16,), tuple(step.done.shape))
            self.assertEqual((16, 4), tuple(step.legal_mask.shape))
            self.assertEqual("cuda", step.obs.device.type)
            self.assertEqual("cuda", step.rewards.device.type)
            self.assertEqual("cuda", step.done.device.type)
            self.assertEqual("cuda", step.legal_mask.device.type)
        finally:
            env.close()

    def test_reset_all_and_step_all_return_all_snake_perspectives(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA is not available")

        env = hisss.CudaBlackoutTorchVecEnv(16, seed=321)
        try:
            obs, legal = env.reset_all()
            self.assertEqual((16, 4, 14, 29, 29), tuple(obs.shape))
            self.assertEqual((16, 4, 4), tuple(legal.shape))
            self.assertEqual("cuda", obs.device.type)
            self.assertEqual("cuda", legal.device.type)
            self.assertTrue(legal.any(dim=-1).all().item())
            self.assertTrue((obs[:, :, 2, 14, 14] > 0).all().item())

            actions = torch.zeros((16, 4), dtype=torch.int32, device=obs.device)
            step = env.step_all(actions)
            self.assertEqual((16, 4, 14, 29, 29), tuple(step.obs.shape))
            self.assertEqual((16,), tuple(step.rewards.shape))
            self.assertEqual((16,), tuple(step.done.shape))
            self.assertEqual((16, 4, 4), tuple(step.legal_mask.shape))
            self.assertEqual("cuda", step.obs.device.type)
            self.assertEqual("cuda", step.rewards.device.type)
            self.assertEqual("cuda", step.done.device.type)
            self.assertEqual("cuda", step.legal_mask.device.type)
        finally:
            env.close()

    def test_step_all_returns_reset_observation_on_done(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA is not available")

        env = hisss.CudaBlackoutTorchVecEnv(64, seed=99)
        try:
            obs, _ = env.reset_all()
            self.assertTrue((obs[:, :, 2, 14, 14] > 0).all().item())
            saw_done = False
            for _ in range(300):
                actions = torch.randint(0, 4, (64, 4), dtype=torch.int32, device=obs.device)
                step = env.step_all(actions)
                if step.done.any().item():
                    saw_done = True
                    self.assertTrue((step.obs[step.done, :, 2, 14, 14] > 0).all().item())
                    break
            self.assertTrue(saw_done)
        finally:
            env.close()

    def test_best_actions_returns_legal_cuda_actions_for_all_snakes(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA is not available")

        env = hisss.CudaBlackoutTorchVecEnv(64, seed=404)
        try:
            _, legal = env.reset_all()
            actions = env.best_actions()
            self.assertEqual((64, 4), tuple(actions.shape))
            self.assertEqual(torch.int32, actions.dtype)
            self.assertEqual("cuda", actions.device.type)
            self.assertTrue(((actions >= 0) & (actions < 4)).all().item())
            selected_legal = legal.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
            self.assertTrue(selected_legal.all().item())

            step = env.step_all(actions)
            next_actions = env.best_actions()
            active = step.legal_mask.any(dim=-1)
            next_selected_legal = step.legal_mask.gather(
                -1,
                next_actions.long().unsqueeze(-1),
            ).squeeze(-1)
            self.assertTrue(next_selected_legal[active].all().item())

            selection = torch.rand((64, 4), device=actions.device) < 0.15
            selected_actions = env.best_actions(selection)
            selected_legal = step.legal_mask.gather(
                -1,
                selected_actions.long().unsqueeze(-1),
            ).squeeze(-1)
            selected_active = selection & active
            self.assertTrue(selected_legal[selected_active].all().item())
            self.assertTrue((selected_actions[~selection] == 0).all().item())
        finally:
            env.close()

    def test_log_derived_heuristics_return_legal_device_actions(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA is not available")

        env = hisss.CudaBlackoutTorchVecEnv(80, seed=405)
        try:
            _, legal = env.reset_all()
            profiles = (
                torch.arange(80 * 4, device=legal.device, dtype=torch.int32)
                .reshape(80, 4)
                .remainder(hisss.BLACKOUT_HEURISTIC_COUNT)
            )
            actions = env.heuristic_actions(profiles)
            self.assertEqual((80, 4), tuple(actions.shape))
            self.assertEqual(torch.int32, actions.dtype)
            self.assertEqual("cuda", actions.device.type)
            selected_legal = legal.gather(
                -1, actions.long().unsqueeze(-1)
            ).squeeze(-1)
            self.assertTrue(selected_legal.all().item())

            skipped = profiles.clone()
            skipped[:, 0] = -1
            skipped_actions = env.heuristic_actions(skipped)
            self.assertTrue((skipped_actions[:, 0] == 0).all().item())
            with self.assertRaises(ValueError):
                env.heuristic_actions(
                    [[hisss.BLACKOUT_HEURISTIC_COUNT, 0, 0, 0]] * 80
                )
        finally:
            env.close()

    def test_log_derived_heuristics_are_behaviorally_diverse(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA is not available")

        env = hisss.CudaBlackoutTorchVecEnv(96, seed=406)
        try:
            _, legal = env.reset_all()
            profile_count = hisss.BLACKOUT_HEURISTIC_COUNT
            disagreements = torch.zeros(
                (profile_count, profile_count), device=legal.device
            )
            comparisons = torch.zeros_like(disagreements)
            for turn in range(40):
                profile_actions = []
                for profile in range(hisss.BLACKOUT_HEURISTIC_COUNT):
                    profile_ids = torch.full(
                        (96, 4),
                        profile,
                        dtype=torch.int32,
                        device=legal.device,
                    )
                    profile_actions.append(env.heuristic_actions(profile_ids))
                stacked = torch.stack(profile_actions)
                active = legal.any(dim=-1)
                for first in range(profile_count):
                    for second in range(first + 1, profile_count):
                        disagreements[first, second] += (
                            (stacked[first] != stacked[second]) & active
                        ).sum()
                        comparisons[first, second] += active.sum()
                step = env.step_all(
                    stacked[turn % hisss.BLACKOUT_HEURISTIC_COUNT]
                )
                legal = step.legal_mask

            pairwise = {}
            for first in range(profile_count):
                for second in range(first + 1, profile_count):
                    pairwise[first, second] = (
                        disagreements[first, second] / comparisons[first, second]
                    ).item()
            # The original behavior clusters stay globally distinct. The
            # Snake-25 counters and duelist intentionally share a proven
            # safety core, but each must differ from the general strategy.
            old_profiles = range(hisss.BLACKOUT_HEURISTIC_SNAKE25)
            self.assertGreater(
                min(
                    pairwise[first, second]
                    for first in old_profiles
                    for second in old_profiles
                    if first < second
                ),
                0.20,
            )
            snake25 = hisss.BLACKOUT_HEURISTIC_SNAKE25
            for counter in (
                hisss.BLACKOUT_HEURISTIC_SNAKE25_INTERCEPTOR,
                hisss.BLACKOUT_HEURISTIC_SNAKE25_DENIER,
                hisss.BLACKOUT_HEURISTIC_SNAKE25_DUELIST,
            ):
                self.assertGreater(pairwise[snake25, counter], 0.20)
        finally:
            env.close()

    def test_step_all_executes_snake_zero_illegal_action_instead_of_replacing_it(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA is not available")

        env = hisss.CudaBlackoutTorchVecEnv(64, seed=505)
        try:
            _, legal = env.reset_all()
            # Initial snakes have no neck segment yet, so advance once to create
            # an unambiguously illegal reverse/body move for every live snake.
            first = env.best_actions()
            first_step = env.step_all(first)
            legal = first_step.legal_mask
            has_illegal = (~legal[:, 0]).any(dim=-1)
            self.assertTrue(has_illegal.any().item())

            actions = env.best_actions()
            illegal_action = (~legal[:, 0]).to(torch.int32).argmax(dim=-1).to(torch.int32)
            actions[has_illegal, 0] = illegal_action[has_illegal]
            step = env.step_all(actions)

            self.assertTrue(step.done[has_illegal].all().item())
            self.assertTrue((step.rewards[has_illegal] <= 0.0).all().item())
        finally:
            env.close()

    def test_eval_step_continues_after_snake_zero_dies(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA is not available")

        env = hisss.CudaBlackoutTorchVecEnv(128, seed=606)
        try:
            _, legal = env.reset_all()
            first = env.best_actions()
            first_step = env.step_all_eval(first)
            legal = first_step.legal_mask
            actions = env.best_actions()
            has_illegal = (~legal[:, 0]).any(dim=-1)
            illegal_action = (
                (~legal[:, 0]).to(torch.int32).argmax(dim=-1).to(torch.int32)
            )
            actions[has_illegal, 0] = illegal_action[has_illegal]
            step = env.step_all_eval(actions)

            snake_zero_died = has_illegal & ~step.alive[:, 0]
            continued = snake_zero_died & ~step.done
            self.assertTrue(continued.any().item())
            self.assertTrue((~step.legal_mask[continued, 0]).all().item())
            self.assertTrue(step.alive[continued, 1:].any(dim=-1).all().item())
        finally:
            env.close()

    def test_eval_step_returns_terminal_metadata_and_horizon_draw(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA is not available")

        env = hisss.CudaBlackoutTorchVecEnv(64, seed=707)
        try:
            obs, _ = env.reset_all()
            actions = env.best_actions()
            step = env.step_all_eval(actions, max_turns=1)
            self.assertTrue(step.done.all().item())
            self.assertTrue((step.winner == -1).all().item())
            self.assertTrue((step.turns == 1).all().item())
            self.assertTrue(step.alive.any(dim=-1).all().item())
            # Done observations already belong to freshly reset games.
            self.assertTrue((step.obs[:, :, 2, 14, 14] > 0).all().item())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
