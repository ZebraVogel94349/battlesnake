"""Checks for the privileged (unfogged) critic observation path.

CPU part (no GPU required, run anywhere with torch + sb3_contrib):
  1. actor logits are invariant to the privileged-channel content
  2. critic value DOES depend on the privileged channels
  3. pre-privileged checkpoint upgrade: the seeded critic_cnn computes exactly
     the actor CNN's features regardless of privileged content (zero conv0 pad)
  4. SB3 export -> import round trip preserves the actor and re-seeds the critic
  5. forward_sequence (two-stream) matches step-by-step _forward_impl

CUDA part (run on the pod after rebuilding the extension):
  6. obs shape is (14, 29, 29) on both env classes
  7. fogged channels 6/7 have NO pixels outside the r5 view diamond
  8. privileged channels are supersets of their fogged counterparts and carry
     information beyond the fog at least once across the sampled steps
  9. the float encoder (host env) and the templated fp16/fp32 encoder
     (torch env) produce identical seat-0 observations from the same seed

Usage:
    python check_priv_critic.py           # CPU checks only
    python check_priv_critic.py --cuda    # CPU + CUDA checks
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

from train_cuda import (
    ActorCritic,
    FrozenActor,
    N_POLICY_CHANNELS,
    N_PRIV_CHANNELS,
    OBS_SHAPE,
    export_sb3_recurrent_ppo,
    forward_sequence,
    load_model_state,
    load_sb3_recurrent_ppo,
)

VIEW_RADIUS = 5


def _random_obs(batch: int, generator: torch.Generator) -> torch.Tensor:
    return torch.rand((batch, *OBS_SHAPE), generator=generator)


def _vary_priv(obs: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    varied = obs.clone()
    varied[:, N_POLICY_CHANNELS:] = torch.rand(
        varied[:, N_POLICY_CHANNELS:].shape, generator=generator
    )
    return varied


def check_actor_invariance():
    generator = torch.Generator().manual_seed(0)
    model = ActorCritic().eval()
    obs = _random_obs(4, generator)
    varied = _vary_priv(obs, generator)
    h, c = model.initial_actor_state(4, torch.device("cpu"))
    with torch.no_grad():
        logits_a, *_ = model.actor_forward(obs, h, c)
        logits_b, *_ = model.actor_forward(varied, h, c)
    assert torch.equal(logits_a, logits_b), "actor depends on privileged channels"

    actor_h, actor_c, critic_h, critic_c = model.initial_state(4, torch.device("cpu"))
    with torch.no_grad():
        logits_a, values_a, *_ = model(obs, actor_h, actor_c, critic_h, critic_c)
        logits_b, values_b, *_ = model(varied, actor_h, actor_c, critic_h, critic_c)
    assert torch.equal(logits_a, logits_b), "full forward: actor sees priv channels"
    assert not torch.allclose(values_a, values_b), (
        "critic ignores privileged channels — it should depend on them"
    )
    print("[cpu] actor invariant to priv channels, critic uses them: OK")


def check_frozen_actor():
    generator = torch.Generator().manual_seed(1)
    model = ActorCritic().eval()
    frozen = FrozenActor(model)
    obs = _random_obs(4, generator)
    h, c = frozen.initial_actor_state(4, torch.device("cpu"))
    with torch.no_grad():
        logits_full, *_ = frozen(obs, h, c)
        logits_policy, *_ = frozen(obs[:, :N_POLICY_CHANNELS], h, c)
        logits_model, *_ = model.actor_forward(obs, h, c)
    assert torch.equal(logits_full, logits_policy), "FrozenActor slice mismatch"
    assert torch.equal(logits_full, logits_model), "FrozenActor != model actor"
    print("[cpu] FrozenActor slices privileged channels: OK")


def check_pre_privileged_upgrade():
    generator = torch.Generator().manual_seed(2)
    source = ActorCritic().eval()
    old_state = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if not key.startswith("critic_cnn.")
    }
    upgraded = ActorCritic().eval()
    legacy = load_model_state(upgraded, old_state, allow_legacy=False)
    assert not legacy

    obs = _random_obs(8, generator)
    with torch.no_grad():
        critic_features = upgraded.critic_cnn(obs)
        actor_features = upgraded.cnn(obs[:, :N_POLICY_CHANNELS])
    assert torch.allclose(critic_features, actor_features, atol=1e-6), (
        "seeded critic_cnn must equal the actor CNN on the policy slice"
    )
    varied = _vary_priv(obs, generator)
    with torch.no_grad():
        critic_features_varied = upgraded.critic_cnn(varied)
    assert torch.allclose(critic_features, critic_features_varied, atol=1e-6), (
        "zero-padded conv0 must make the seeded critic ignore priv content"
    )
    print("[cpu] pre-privileged checkpoint upgrade (zero-pad seeding): OK")


def check_export_import_round_trip():
    generator = torch.Generator().manual_seed(3)
    model = ActorCritic().eval()
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = str(Path(tmp_dir) / "round_trip.zip")
        export_sb3_recurrent_ppo(model, path)
        imported = load_sb3_recurrent_ppo(path, device="cpu")

    obs = _random_obs(4, generator)
    h, c = model.initial_actor_state(4, torch.device("cpu"))
    with torch.no_grad():
        logits_a, *_ = model.actor_forward(obs, h, c)
        logits_b, *_ = imported.actor_forward(obs, h, c)
        critic_features = imported.critic_cnn(obs)
        critic_features_varied = imported.critic_cnn(_vary_priv(obs, generator))
    assert torch.allclose(logits_a, logits_b, atol=1e-6), (
        "export/import changed the actor"
    )
    assert torch.allclose(critic_features, critic_features_varied, atol=1e-6), (
        "imported critic_cnn must start with a zeroed privileged slice"
    )
    print("[cpu] SB3 export -> import round trip: OK")


def check_forward_sequence_consistency():
    generator = torch.Generator().manual_seed(4)
    model = ActorCritic().eval()
    t_steps, batch = 6, 3
    obs = torch.rand((t_steps, batch, *OBS_SHAPE), generator=generator)
    dones = torch.zeros((t_steps, batch), dtype=torch.bool)
    dones[2, 1] = True  # exercise an episode boundary
    actor_h, actor_c, critic_h, critic_c = model.initial_state(
        batch, torch.device("cpu")
    )
    with torch.no_grad():
        seq_logits, seq_values = forward_sequence(
            model, obs, dones, actor_h, actor_c, critic_h, critic_c
        )

    step_logits = torch.empty_like(seq_logits)
    step_values = torch.empty_like(seq_values)
    ah, ac, ch, cc = actor_h, actor_c, critic_h, critic_c
    with torch.no_grad():
        for t in range(t_steps):
            logits, values, ah, ac, ch, cc = model(obs[t], ah, ac, ch, cc)
            step_logits[t] = logits
            step_values[t] = values
            keep = (~dones[t]).float().unsqueeze(-1)
            ah, ac, ch, cc = ah * keep, ac * keep, ch * keep, cc * keep
    assert torch.allclose(seq_logits, step_logits, atol=1e-5), (
        "forward_sequence logits diverge from step-by-step forward"
    )
    assert torch.allclose(seq_values, step_values, atol=1e-5), (
        "forward_sequence values diverge from step-by-step forward"
    )
    print("[cpu] forward_sequence matches step-by-step forward: OK")


def _fog_diamond() -> np.ndarray:
    xs = np.arange(29).reshape(29, 1)
    ys = np.arange(29).reshape(1, 29)
    return (np.abs(xs - 14) + np.abs(ys - 14)) <= VIEW_RADIUS


def check_cuda_observations(num_envs: int = 64, steps: int = 40, seed: int = 0):
    import hisss

    diamond = _fog_diamond()
    outside = ~diamond

    torch_env = hisss.CudaBlackoutTorchVecEnv(num_envs, seed=seed, device="cuda")
    obs_all, legal_all = torch_env.reset_all()
    assert tuple(obs_all.shape) == (num_envs, 4, *OBS_SHAPE), obs_all.shape

    priv_outside_seen = np.zeros(N_PRIV_CHANNELS, dtype=bool)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        obs_np = obs_all.float().cpu().numpy().reshape(num_envs * 4, *OBS_SHAPE)
        alive = obs_np[:, 3].reshape(num_envs * 4, -1).max(axis=1) > 0.0
        obs_np = obs_np[alive]

        for fogged_ch in (6, 7):
            assert not (obs_np[:, fogged_ch][:, outside] > 0).any(), (
                f"fogged channel {fogged_ch} leaks outside the view diamond"
            )
        # Priv supersets of their fogged counterparts inside the diamond.
        assert ((obs_np[:, 0] > 0) <= (obs_np[:, 9] > 0)).all(), (
            "priv food (9) must contain the fogged food channel (0)"
        )
        assert ((obs_np[:, 6] > 0) <= (obs_np[:, 10] > 0)).all(), (
            "priv enemy body (10) must contain the fogged occupancy (6)"
        )
        assert ((obs_np[:, 7] > 0) <= (obs_np[:, 11] > 0)).all(), (
            "priv enemy head (11) must contain the fogged head channel (7)"
        )
        # Health is painted on the same cells as the body decay (0 hp allowed).
        assert ((obs_np[:, 12] > 0) <= (obs_np[:, 10] > 0)).all(), (
            "priv enemy health (12) written off the enemy body"
        )
        assert ((obs_np[:, 13] > 0) <= (obs_np[:, 10] > 0)).all(), (
            "priv enemy tail (13) written off the enemy body"
        )
        for priv_index in range(N_PRIV_CHANNELS):
            channel = obs_np[:, N_POLICY_CHANNELS + priv_index]
            if (channel[:, outside] > 0).any():
                priv_outside_seen[priv_index] = True

        legal_np = legal_all.cpu().numpy().reshape(num_envs * 4, 4)
        actions = np.array(
            [
                rng.choice(np.nonzero(row)[0]) if row.any() else 0
                for row in legal_np
            ],
            dtype=np.int64,
        ).reshape(num_envs, 4)
        step = torch_env.step_all(
            torch.as_tensor(actions, device=obs_all.device)
        )
        obs_all = step.obs
        legal_all = step.legal_mask
    # The tail channel is a single pixel per enemy; give it the most slack.
    assert priv_outside_seen[:4].all(), (
        f"privileged channels never showed content beyond the fog: "
        f"{priv_outside_seen}"
    )
    if not priv_outside_seen[4]:
        print("[cuda] WARN: priv tail never observed outside the fog "
              f"in {steps} steps (rare but possible; rerun with more steps)")
    print("[cuda] fog geometry + privileged superset checks: OK")

    # Float encoder (host env) vs templated encoder (torch env), same seed.
    host_env = hisss.CudaBlackoutVecEnv(num_envs, seed=seed)
    host_obs, host_legal = host_env.reset()
    fresh_torch = hisss.CudaBlackoutTorchVecEnv(num_envs, seed=seed, device="cuda")
    torch_obs, torch_legal = fresh_torch.reset_all()
    torch_seat0 = torch_obs[:, 0].float().cpu().numpy()
    if np.allclose(host_obs, torch_seat0, atol=1e-3):
        print("[cuda] float vs templated encoder parity (reset, seat 0): OK")
    else:
        mismatch = np.abs(host_obs - torch_seat0).max(axis=(0, 2, 3))
        raise AssertionError(
            f"host/templated encoder mismatch, per-channel max diff: {mismatch}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="also run env checks")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    assert OBS_SHAPE == (N_POLICY_CHANNELS + N_PRIV_CHANNELS, 29, 29)
    check_actor_invariance()
    check_frozen_actor()
    check_pre_privileged_upgrade()
    check_export_import_round_trip()
    check_forward_sequence_consistency()
    if args.cuda:
        check_cuda_observations(args.num_envs, args.steps)
    else:
        print("CPU checks passed. Run with --cuda on the pod for env checks.")


if __name__ == "__main__":
    main()
