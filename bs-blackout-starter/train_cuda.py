"""GPU-first recurrent PPO training for Battlesnake Blackout.

Rollouts stay on the GPU and opponents are fixed for a complete episode.  The
self-play pool contains frozen actor checkpoints plus CUDA Hungry, Random, and
a best_agent-inspired CUDA heuristic.  The learner uses independent
actor and critic LSTMs, matching the architecture exported to SB3/``ppo.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import hisss


# The CUDA env emits 9 fogged policy channels followed by 5 privileged
# (unfogged) channels. Only the critic reads the privileged slice; the actor
# and every deployed/frozen policy see channels [:N_POLICY_CHANNELS].
N_POLICY_CHANNELS = 9
N_PRIV_CHANNELS = 5
OBS_SHAPE = (N_POLICY_CHANNELS + N_PRIV_CHANNELS, 29, 29)
MODEL_TO_CUDA_ACTION = (0, 2, 3, 1)  # model: UP, DOWN, LEFT, RIGHT; CUDA: UP, RIGHT, DOWN, LEFT
OBS_CENTER = 14
OBS_VIEW_RADIUS = 5
OBS_VIEW_BOUNDARY = tuple(
    (x, y)
    for x in range(29)
    for y in range(29)
    if abs(x - OBS_CENTER) + abs(y - OBS_CENTER) == OBS_VIEW_RADIUS
)
_ACTION_PERM_CACHE: dict[torch.device, torch.Tensor] = {}
_INVERSE_ACTION_PERM_CACHE: dict[torch.device, torch.Tensor] = {}
_VIEW_BOUNDARY_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
_SEQUENCE_LSTM_CACHE: weakref.WeakKeyDictionary[nn.LSTMCell, nn.LSTM] = (
    weakref.WeakKeyDictionary()
)
ROLLOUT_MASK_REVISION = "inductor-no-cudagraph-v3"


def validate_cuda_all_seat_reset(
    obs_all: torch.Tensor,
    legal_all: torch.Tensor,
    *,
    context: str,
    allow_duels: bool = False,
) -> None:
    """Fail fast when Python and the native Hisss CUDA encoder disagree.

    Normally every snake is alive immediately after a reset, so each seat must
    contain its own body and health at the centered head cell. Mixed duel
    training deliberately leaves seats two and three empty, but seats zero and
    one must still be fully encoded. A stale ``liblink.so`` used to leave most
    all-seat observation buffers unwritten, which made roughly one third of
    otherwise normal games end in draws.
    """

    expected_obs = (obs_all.shape[0], 4, *OBS_SHAPE)
    expected_legal = (obs_all.shape[0], 4, 4)
    present = (
        (obs_all[:, :, 2, OBS_CENTER, OBS_CENTER] > 0)
        & (obs_all[:, :, 4, OBS_CENTER, OBS_CENTER] > 0)
        & legal_all.any(dim=-1)
    )
    valid_presence = (
        bool(present.all().item())
        if not allow_duels
        else bool(
            (
                present[:, 0]
                & present[:, 1]
                & (present[:, 2] == present[:, 3])
            ).all().item()
        )
    )
    valid = (
        tuple(obs_all.shape) == expected_obs
        and tuple(legal_all.shape) == expected_legal
        and bool(torch.isfinite(obs_all).all().item())
        and valid_presence
    )
    if valid:
        return
    raise RuntimeError(
        f"Invalid CUDA all-seat observations after reset ({context}). "
        "The native Hisss extension is probably stale or incompatible with "
        "the Python package. Rebuild it from the Hisss repository root with "
        "`.venv/bin/python -m pip install -e .` before training."
    )


@dataclass
class Rollout:
    obs: torch.Tensor
    legal_mask: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    env_rewards: torch.Tensor
    illegal_actions: torch.Tensor
    actor_h0: torch.Tensor
    actor_c0: torch.Tensor
    critic_h0: torch.Tensor
    critic_c0: torch.Tensor
    seq_len: int
    use_action_mask: bool


def compile_mode_without_cudagraphs(mode: str) -> str:
    """Keep Inductor fusion but avoid CUDA-Graph output lifetime constraints."""

    return "default" if mode == "reduce-overhead" else mode


class ActorCritic(nn.Module):
    """Independent recurrent actor and critic with a privileged critic CNN.

    ``self.cnn`` is the ACTOR's extractor over the fogged policy channels; its
    parameter names are unchanged from the shared-CNN era so old checkpoints,
    pool snapshots, and the SB3 export stay compatible. ``self.critic_cnn``
    additionally consumes the privileged (unfogged) channels — training-only
    information that never reaches the deployed policy.

    ``Tanh`` in both MLP heads is intentional: it is SB3's default activation
    and therefore makes :func:`export_sb3_recurrent_ppo` numerically faithful.
    """

    def __init__(
        self,
        n_policy_channels: int = N_POLICY_CHANNELS,
        n_priv_channels: int = N_PRIV_CHANNELS,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_size = 256
        self.n_policy_channels = n_policy_channels
        self.use_channels_last = False
        object.__setattr__(self, "_training_cnn", None)
        object.__setattr__(self, "_training_critic_cnn", None)
        object.__setattr__(self, "_rollout_forward", None)
        self.cnn = self._make_cnn(n_policy_channels)
        self.critic_cnn = self._make_cnn(n_policy_channels + n_priv_channels)
        self.actor_lstm = nn.LSTMCell(self.feature_size, hidden_size)
        self.critic_lstm = nn.LSTMCell(self.feature_size, hidden_size)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 4),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self._init_sb3_weights()

    def _make_cnn(self, n_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, self.feature_size),
            nn.ReLU(),
        )

    def enable_channels_last(self):
        """Use Tensor-Core-friendly NHWC convolution kernels internally."""

        self.cnn.to(memory_format=torch.channels_last)
        self.critic_cnn.to(memory_format=torch.channels_last)
        self.use_channels_last = True
        return self

    def policy_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Drop the privileged critic channels if the tensor carries them."""

        if obs.shape[1] == self.n_policy_channels:
            return obs
        return obs[:, : self.n_policy_channels]

    def cnn_forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.use_channels_last:
            obs = obs.contiguous(memory_format=torch.channels_last)
        compiled_cnn = self._training_cnn
        return self.cnn(obs) if compiled_cnn is None else compiled_cnn(obs)

    def enable_compiled_training_cnn(self, mode: str = "default"):
        """Compile learner graphs without changing checkpoint parameters.

        The recurrent PPO learner invokes actor CNN, critic CNN, and loss as
        separate compiled callables before one shared backward. CUDA Graph
        Trees cannot begin a new fast-path step while outputs of either CNN
        still require that backward. Keep those Autograd callables in regular
        Inductor mode, while the inference-only rollout forward may safely use
        the requested ``reduce-overhead`` CUDA-Graph mode.
        """

        learner_mode = compile_mode_without_cudagraphs(mode)
        compiled = torch.compile(self.cnn, mode=learner_mode, dynamic=False)
        compiled_critic = torch.compile(
            self.critic_cnn,
            mode=learner_mode,
            dynamic=False,
        )
        compiled_rollout = torch.compile(
            self._forward_impl,
            mode=mode,
            dynamic=False,
        )
        # Bypass Module registration: the compiled wrapper references the same
        # parameters, but must not add `_orig_mod` names to state_dict exports.
        object.__setattr__(self, "_training_cnn", compiled)
        object.__setattr__(self, "_training_critic_cnn", compiled_critic)
        object.__setattr__(self, "_rollout_forward", compiled_rollout)
        return self

    def training_cnn_forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.use_channels_last:
            obs = obs.contiguous(memory_format=torch.channels_last)
        training_cnn = self._training_cnn
        return self.cnn(obs) if training_cnn is None else training_cnn(obs)

    def training_critic_cnn_forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.use_channels_last:
            obs = obs.contiguous(memory_format=torch.channels_last)
        training_cnn = self._training_critic_cnn
        return self.critic_cnn(obs) if training_cnn is None else training_cnn(obs)

    def _init_sb3_weights(self):
        """Match the orthogonal initialization used by SB3 policies."""

        for cnn in (self.cnn, self.critic_cnn):
            for module in cnn.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                    nn.init.zeros_(module.bias)
        for branch, output_gain in ((self.actor, 0.01), (self.critic, 1.0)):
            linear_layers = [m for m in branch if isinstance(m, nn.Linear)]
            for module in linear_layers[:-1]:
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
            nn.init.orthogonal_(linear_layers[-1].weight, gain=output_gain)
            nn.init.zeros_(linear_layers[-1].bias)

    def initial_state(self, batch_size: int, device: torch.device):
        actor_h = torch.zeros(batch_size, self.hidden_size, device=device)
        actor_c = torch.zeros(batch_size, self.hidden_size, device=device)
        critic_h = torch.zeros(batch_size, self.hidden_size, device=device)
        critic_c = torch.zeros(batch_size, self.hidden_size, device=device)
        return actor_h, actor_c, critic_h, critic_c

    def initial_actor_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        return h, c

    def actor_forward(self, obs: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        features = self.cnn_forward(self.policy_obs(obs))
        h_next, c_next = self.actor_lstm(features, (h, c))
        return self.actor(h_next), h_next, c_next

    def _forward_impl(
        self,
        obs: torch.Tensor,
        actor_h: torch.Tensor,
        actor_c: torch.Tensor,
        critic_h: torch.Tensor,
        critic_c: torch.Tensor,
    ):
        if self.use_channels_last:
            obs = obs.contiguous(memory_format=torch.channels_last)
        features = self.cnn(self.policy_obs(obs))
        critic_features = self.critic_cnn(obs)
        actor_h_next, actor_c_next = self.actor_lstm(features, (actor_h, actor_c))
        critic_h_next, critic_c_next = self.critic_lstm(
            critic_features, (critic_h, critic_c)
        )
        logits = self.actor(actor_h_next)
        values = self.critic(critic_h_next).squeeze(-1)
        return (
            logits,
            values,
            actor_h_next,
            actor_c_next,
            critic_h_next,
            critic_c_next,
        )

    def forward(
        self,
        obs: torch.Tensor,
        actor_h: torch.Tensor,
        actor_c: torch.Tensor,
        critic_h: torch.Tensor,
        critic_c: torch.Tensor,
    ):
        rollout_forward = self._rollout_forward
        if rollout_forward is not None:
            return rollout_forward(
                obs,
                actor_h,
                actor_c,
                critic_h,
                critic_c,
            )
        return self._forward_impl(
            obs,
            actor_h,
            actor_c,
            critic_h,
            critic_c,
        )


class FrozenActor(nn.Module):
    """Actor-only snapshot used by the opponent pool.

    Omitting the critic halves the recurrent state and avoids a useless value
    forward for every frozen opponent.
    """

    def __init__(self, model: ActorCritic):
        super().__init__()
        self.hidden_size = model.hidden_size
        self.n_policy_channels = model.n_policy_channels
        self.use_channels_last = model.use_channels_last
        self.cnn = copy.deepcopy(model.cnn)
        self.actor_lstm = copy.deepcopy(model.actor_lstm)
        self.actor = copy.deepcopy(model.actor)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, obs: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        if obs.shape[1] != self.n_policy_channels:
            obs = obs[:, : self.n_policy_channels]
        if self.use_channels_last:
            obs = obs.contiguous(memory_format=torch.channels_last)
        features = self.cnn(obs)
        h_next, c_next = self.actor_lstm(features, (h, c))
        return self.actor(h_next), h_next, c_next

    def initial_actor_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        return h, c

    def actor_forward(self, obs: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        return self.forward(obs, h, c)


def _copy_param(dst: dict, dst_name: str, src: dict, src_name: str):
    if dst[dst_name].shape != src[src_name].shape:
        raise ValueError(
            f"Shape mismatch for {dst_name}: {tuple(dst[dst_name].shape)} "
            f"!= {tuple(src[src_name].shape)}"
        )
    dst[dst_name].copy_(src[src_name].detach().cpu())


def export_sb3_recurrent_ppo(model: ActorCritic, path: str):
    """Export the custom CUDA-trained model into ppo.py's SB3 zip format.

    The privileged ``critic_cnn`` is deliberately NOT exported: the ppo.py
    skeleton is 9-channel and never evaluates the value head at inference, so
    the actor CNN fills all extractor slots (values in the zip are meaningless
    but unused). Importing such a zip via :func:`load_sb3_recurrent_ppo`
    re-seeds the privileged critic from the actor CNN.
    """
    from obs_config import build_model

    sb3_model = build_model(device="cpu")
    src = model.state_dict()
    dst = sb3_model.policy.state_dict()

    for prefix in (
        "features_extractor",
        "pi_features_extractor",
        "vf_features_extractor",
    ):
        _copy_param(dst, f"{prefix}.cnn.0.weight", src, "cnn.0.weight")
        _copy_param(dst, f"{prefix}.cnn.0.bias", src, "cnn.0.bias")
        _copy_param(dst, f"{prefix}.cnn.2.weight", src, "cnn.2.weight")
        _copy_param(dst, f"{prefix}.cnn.2.bias", src, "cnn.2.bias")
        _copy_param(dst, f"{prefix}.cnn.4.weight", src, "cnn.4.weight")
        _copy_param(dst, f"{prefix}.cnn.4.bias", src, "cnn.4.bias")
        _copy_param(dst, f"{prefix}.linear.0.weight", src, "cnn.7.weight")
        _copy_param(dst, f"{prefix}.linear.0.bias", src, "cnn.7.bias")

    _copy_param(dst, "mlp_extractor.policy_net.0.weight", src, "actor.0.weight")
    _copy_param(dst, "mlp_extractor.policy_net.0.bias", src, "actor.0.bias")
    _copy_param(dst, "mlp_extractor.policy_net.2.weight", src, "actor.2.weight")
    _copy_param(dst, "mlp_extractor.policy_net.2.bias", src, "actor.2.bias")
    _copy_param(dst, "action_net.weight", src, "actor.4.weight")
    _copy_param(dst, "action_net.bias", src, "actor.4.bias")

    _copy_param(dst, "mlp_extractor.value_net.0.weight", src, "critic.0.weight")
    _copy_param(dst, "mlp_extractor.value_net.0.bias", src, "critic.0.bias")
    _copy_param(dst, "mlp_extractor.value_net.2.weight", src, "critic.2.weight")
    _copy_param(dst, "mlp_extractor.value_net.2.bias", src, "critic.2.bias")
    _copy_param(dst, "value_net.weight", src, "critic.4.weight")
    _copy_param(dst, "value_net.bias", src, "critic.4.bias")

    for dst_prefix, src_prefix in (
        ("lstm_actor", "actor_lstm"),
        ("lstm_critic", "critic_lstm"),
    ):
        _copy_param(
            dst, f"{dst_prefix}.weight_ih_l0", src, f"{src_prefix}.weight_ih"
        )
        _copy_param(
            dst, f"{dst_prefix}.weight_hh_l0", src, f"{src_prefix}.weight_hh"
        )
        _copy_param(
            dst, f"{dst_prefix}.bias_ih_l0", src, f"{src_prefix}.bias_ih"
        )
        _copy_param(
            dst, f"{dst_prefix}.bias_hh_l0", src, f"{src_prefix}.bias_hh"
        )

    sb3_model.policy.load_state_dict(dst, strict=True)
    save_arg = path[:-4] if path.endswith(".zip") else path
    sb3_model.save(save_arg)
    final_path = save_arg + ".zip"
    if final_path != path and path.endswith(".zip"):
        os.replace(final_path, path)


def export_periodic_sb3_checkpoint(
    model: ActorCritic,
    directory: str,
    prefix: str,
    total_steps: int,
    update: int,
    max_checkpoints: int = 0,
) -> Path:
    """Atomically export a named intermediate policy and apply retention."""

    target_dir = Path(directory).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = Path(prefix).stem or "ppo_bs_lstm_cuda"
    target = target_dir / (
        f"{safe_prefix}_steps_{total_steps:012d}_u{update:08d}.zip"
    )
    # Keep the temporary basename extension-free before ``.zip``. SB3 treats
    # an intermediate suffix such as ``.tmp`` as the final filename and would
    # otherwise omit the expected ZIP suffix.
    temporary = target.with_name(f"{target.stem}_tmp.zip")
    export_sb3_recurrent_ppo(model, str(temporary))
    os.replace(temporary, target)

    if max_checkpoints > 0:
        exports = sorted(
            target_dir.glob(f"{safe_prefix}_steps_*_u*.zip"),
            key=lambda path: path.stat().st_mtime,
        )
        for stale in exports[:-max_checkpoints]:
            stale.unlink(missing_ok=True)
    return target


def export_league_champion(model: ActorCritic, path: str) -> Path:
    """Atomically replace the ppo.py-compatible deployable champion."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}_tmp.zip")
    export_sb3_recurrent_ppo(model, str(temporary))
    os.replace(temporary, target)
    return target


def load_sb3_recurrent_ppo(
    path: str,
    device: torch.device | str = "cuda",
) -> ActorCritic:
    """Load a ppo.py-compatible SB3 zip into the lightweight CUDA model."""

    from obs_config import build_model

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    sb3_model = build_model(device="cpu")
    # Loading the complete pickled model has historically crashed in this
    # project. set_parameters only reads the state dictionaries from the zip.
    sb3_model.set_parameters(path, device="cpu")
    src = sb3_model.policy.state_dict()
    model = ActorCritic()
    dst = model.state_dict()

    feature_prefix = "features_extractor"
    for dst_name, src_name in (
        ("cnn.0.weight", f"{feature_prefix}.cnn.0.weight"),
        ("cnn.0.bias", f"{feature_prefix}.cnn.0.bias"),
        ("cnn.2.weight", f"{feature_prefix}.cnn.2.weight"),
        ("cnn.2.bias", f"{feature_prefix}.cnn.2.bias"),
        ("cnn.4.weight", f"{feature_prefix}.cnn.4.weight"),
        ("cnn.4.bias", f"{feature_prefix}.cnn.4.bias"),
        ("cnn.7.weight", f"{feature_prefix}.linear.0.weight"),
        ("cnn.7.bias", f"{feature_prefix}.linear.0.bias"),
        ("actor.0.weight", "mlp_extractor.policy_net.0.weight"),
        ("actor.0.bias", "mlp_extractor.policy_net.0.bias"),
        ("actor.2.weight", "mlp_extractor.policy_net.2.weight"),
        ("actor.2.bias", "mlp_extractor.policy_net.2.bias"),
        ("actor.4.weight", "action_net.weight"),
        ("actor.4.bias", "action_net.bias"),
        ("critic.0.weight", "mlp_extractor.value_net.0.weight"),
        ("critic.0.bias", "mlp_extractor.value_net.0.bias"),
        ("critic.2.weight", "mlp_extractor.value_net.2.weight"),
        ("critic.2.bias", "mlp_extractor.value_net.2.bias"),
        ("critic.4.weight", "value_net.weight"),
        ("critic.4.bias", "value_net.bias"),
    ):
        _copy_param(dst, dst_name, src, src_name)

    # SB3 zips carry only the 9-channel shared CNN. Seed the privileged critic
    # CNN from it with the privileged conv0 input slice zeroed, so the
    # imported critic reproduces the source model's values exactly.
    for dst_name, src_name in (
        ("critic_cnn.0.bias", f"{feature_prefix}.cnn.0.bias"),
        ("critic_cnn.2.weight", f"{feature_prefix}.cnn.2.weight"),
        ("critic_cnn.2.bias", f"{feature_prefix}.cnn.2.bias"),
        ("critic_cnn.4.weight", f"{feature_prefix}.cnn.4.weight"),
        ("critic_cnn.4.bias", f"{feature_prefix}.cnn.4.bias"),
        ("critic_cnn.7.weight", f"{feature_prefix}.linear.0.weight"),
        ("critic_cnn.7.bias", f"{feature_prefix}.linear.0.bias"),
    ):
        _copy_param(dst, dst_name, src, src_name)
    shared_conv0 = src[f"{feature_prefix}.cnn.0.weight"].detach().cpu()
    dst["critic_cnn.0.weight"].zero_()
    dst["critic_cnn.0.weight"][:, : shared_conv0.shape[1]].copy_(shared_conv0)

    for dst_prefix, src_prefix in (
        ("actor_lstm", "lstm_actor"),
        ("critic_lstm", "lstm_critic"),
    ):
        for dst_suffix, src_suffix in (
            ("weight_ih", "weight_ih_l0"),
            ("weight_hh", "weight_hh_l0"),
            ("bias_ih", "bias_ih_l0"),
            ("bias_hh", "bias_hh_l0"),
        ):
            _copy_param(
                dst,
                f"{dst_prefix}.{dst_suffix}",
                src,
                f"{src_prefix}.{src_suffix}",
            )

    model.load_state_dict(dst, strict=True)
    model.to(torch.device(device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def set_actor_frozen(model: ActorCritic, frozen: bool):
    """Freeze/unfreeze the actor-side parameters for critic warmup.

    After a warm start the privileged critic must learn its zero-initialised
    unfogged inputs; freezing ``cnn``/``actor_lstm``/``actor`` keeps the
    rollout policy at the warm-started one meanwhile. The critic LSTM and head
    stay trainable.
    """

    for module in (model.cnn, model.actor_lstm, model.actor):
        for parameter in module.parameters():
            parameter.requires_grad_(not frozen)


def actor_critic_parameter_groups(
    model: ActorCritic,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return disjoint trainable branches for independent gradient clipping."""

    actor_parameters = [
        parameter
        for module in (model.cnn, model.actor_lstm, model.actor)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    critic_parameters = [
        parameter
        for module in (model.critic_cnn, model.critic_lstm, model.critic)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    return actor_parameters, critic_parameters


def actor_reference_l2_loss(
    actor_parameters: list[nn.Parameter],
    reference_parameters: list[torch.Tensor] | None,
    coefficient: float | torch.Tensor,
) -> torch.Tensor:
    """Penalize cumulative actor drift from one immutable strong policy.

    PPO's clipped objective only constrains a policy relative to the rollout
    that immediately preceded the current update.  Thousands of individually
    small updates can therefore forget an older capability.  L2-SP supplies a
    global trust region around a selected reference actor while leaving the
    critic free to adapt to the current opponent distribution.
    """

    if not actor_parameters:
        if reference_parameters:
            return reference_parameters[0].new_zeros(())
        return torch.zeros(())
    if not reference_parameters:
        return actor_parameters[0].new_zeros(())
    if len(actor_parameters) != len(reference_parameters):
        raise ValueError(
            "actor reference parameter count does not match the learner actor"
        )
    penalty = actor_parameters[0].new_zeros((), dtype=torch.float32)
    for parameter, reference in zip(actor_parameters, reference_parameters):
        if parameter.shape != reference.shape:
            raise ValueError(
                "actor reference parameter shape does not match the learner actor"
            )
        penalty = penalty + (parameter.float() - reference.float()).square().sum()
    return 0.5 * coefficient * penalty


def masked_categorical(logits: torch.Tensor, legal_mask: torch.Tensor):
    masked_logits = torch.where(
        legal_mask,
        logits,
        torch.full_like(logits, -1e9),
    )
    return torch.distributions.Categorical(logits=masked_logits)


def policy_distribution(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    use_action_mask: bool,
):
    if use_action_mask:
        return masked_categorical(logits, legal_mask)
    return torch.distributions.Categorical(logits=logits)


def policy_log_probs(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    use_action_mask: bool,
) -> torch.Tensor:
    """Normalize policy logits once for sampling, loss, and entropy metrics."""

    logits = logits.float()
    if use_action_mask:
        logits = logits.masked_fill(~legal_mask, -1e9)
    return F.log_softmax(logits, dim=-1)


def ppo_loss_components(
    logits: torch.Tensor,
    values: torch.Tensor,
    legal_mask: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    use_action_mask: bool,
    clip_range: float,
    clip_range_vf: float | None,
    ent_coef: float | torch.Tensor,
    vf_coef: float,
) -> tuple[torch.Tensor, ...]:
    """Compute PPO's pointwise loss in one compiler-fusible graph."""

    logits = logits.float()
    values = values.float()
    log_probs = policy_log_probs(logits, legal_mask, use_action_mask)
    new_logprobs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

    logratio = new_logprobs - old_logprobs
    ratio = torch.exp(logratio)
    pg_loss_1 = -advantages * ratio
    pg_loss_2 = -advantages * torch.clamp(
        ratio,
        1.0 - clip_range,
        1.0 + clip_range,
    )
    policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

    if clip_range_vf is None:
        values_pred = values
    else:
        values_pred = old_values + torch.clamp(
            values - old_values,
            -clip_range_vf,
            clip_range_vf,
        )
    value_loss = F.mse_loss(values_pred, returns)
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    approx_kl = ((ratio - 1.0) - logratio).mean()
    clipfrac = ((ratio - 1.0).abs() > clip_range).float().mean()
    return loss, policy_loss, value_loss, entropy, approx_kl, clipfrac


def sample_policy_actions(log_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    actions = torch.multinomial(log_probs.exp(), 1).squeeze(-1)
    selected_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return actions, selected_log_probs


def _observable_action_masks_impl(
    obs: torch.Tensor,
    boundary_x: torch.Tensor,
    boundary_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return hard and conservative masks derived from the fogged observation.

    ``hard`` excludes only observable certain-death geometry and starvation.
    ``conservative`` additionally avoids head-to-head contests that are not
    known wins.  It falls back to ``hard`` when every hard-legal move is
    contested, matching the historical deployment policy.
    """

    # Model order: UP, DOWN, LEFT, RIGHT. Observation spatial axes are x, y.
    targets = ((14, 15), (14, 13), (13, 14), (15, 14))
    hard_parts = []
    contested_parts = []
    enemy_heads = obs[:, 7] > 0.0
    enemy_body = obs[:, 6] > 0.0
    enemy_reaches_fog = enemy_body[:, boundary_x, boundary_y].any(dim=-1)
    visible_enemy_length = enemy_body.sum(dim=(-2, -1))
    own_length = obs[:, 2, OBS_CENTER, OBS_CENTER].float() * 10.0
    # Summing all fully visible enemy bodies can only overestimate the length
    # of the particular head contesting a move, so this remains conservative
    # when multiple enemies touch each other in the compressed enemy channel.
    known_winning_contest = (
        ~enemy_reaches_fog
        & (visible_enemy_length > 0)
        & (own_length > visible_enemy_length.float() + 0.5)
    )
    health = obs[:, 4, 14, 14]
    for x, y in targets:
        in_bounds = obs[:, 1, x, y] > 0.0
        own_body = obs[:, 2, x, y] > 0.0
        own_tail = obs[:, 5, x, y] > 0.0
        enemy_occupied = enemy_body[:, x, y]
        food = obs[:, 0, x, y] > 0.0
        avoids_starvation = (health > 0.01) | food
        hard_parts.append(
            in_bounds
            & ~(own_body & ~own_tail)
            & ~enemy_occupied
            & avoids_starvation
        )
        contested_parts.append(
            (
                enemy_heads[:, x + 1, y]
                | enemy_heads[:, x - 1, y]
                | enemy_heads[:, x, y + 1]
                | enemy_heads[:, x, y - 1]
            )
            & ~known_winning_contest
        )
    hard = torch.stack(hard_parts, dim=-1)
    soft = hard & ~torch.stack(contested_parts, dim=-1)
    conservative = torch.where(soft.any(dim=-1, keepdim=True), soft, hard)
    return hard, conservative


def view_boundary_indices(
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached CUDA/CPU indices used by the observable action mask."""

    device = torch.device(device)
    boundary_indices = _VIEW_BOUNDARY_CACHE.get(device)
    if boundary_indices is None:
        boundary_x, boundary_y = zip(*OBS_VIEW_BOUNDARY)
        boundary_indices = (
            torch.tensor(boundary_x, dtype=torch.long, device=device),
            torch.tensor(boundary_y, dtype=torch.long, device=device),
        )
        _VIEW_BOUNDARY_CACHE[device] = boundary_indices
    return boundary_indices


def observable_action_masks(
    obs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return hard and conservative masks derived from the fogged observation."""

    boundary_x, boundary_y = view_boundary_indices(obs.device)
    return _observable_action_masks_impl(obs, boundary_x, boundary_y)


def compile_observable_action_masks(
    mode: str = "default",
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Compile the fixed-shape learner mask without CUDA-Graph output reuse.

    Frozen opponents retain the eager/dynamic path because their batch sizes
    vary with episode assignments.  Learner rollout batches are fixed, making
    them a good fit for Inductor without repeated recompilation. The caller's
    ``reduce-overhead`` mode is deliberately resolved to ``default`` here:
    these outputs live across the separately captured learner-model graph, and
    PyTorch may otherwise overwrite them before policy masking consumes them.
    """

    compiler_mode = compile_mode_without_cudagraphs(mode)
    compiled_impl = torch.compile(
        _observable_action_masks_impl,
        mode=compiler_mode,
        dynamic=False,
        fullgraph=True,
    )

    def compiled(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        boundary_x, boundary_y = view_boundary_indices(obs.device)
        return compiled_impl(obs, boundary_x, boundary_y)

    return compiled


def observable_hard_action_mask(obs: torch.Tensor) -> torch.Tensor:
    """Mask only observable certain-death moves and let PPO price contests."""

    hard, _ = observable_action_masks(obs)
    return hard


def observable_action_mask(obs: torch.Tensor) -> torch.Tensor:
    """Historical conservative deployment mask, retained for compatibility."""

    _, conservative = observable_action_masks(obs)
    return conservative


def training_action_mask(
    obs: torch.Tensor,
    server_legal_mask: torch.Tensor,
    mode: str,
    observable_masks_fn: Callable[
        [torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ] = observable_action_masks,
) -> tuple[torch.Tensor, bool]:
    if mode == "none":
        return server_legal_mask, False
    if mode == "observable":
        _, conservative = observable_masks_fn(obs)
        return conservative, True
    if mode == "observable-hard":
        hard, _ = observable_masks_fn(obs)
        return hard, True
    if mode == "server":
        return server_legal_mask, True
    raise ValueError(f"unsupported action-mask mode: {mode}")


def rollout_action_masks(
    obs: torch.Tensor,
    server_legal_mask: torch.Tensor,
    mode: str,
    *,
    need_mobility_mask: bool,
    observable_masks_fn: Callable[
        [torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ] = observable_action_masks,
) -> tuple[torch.Tensor, bool, torch.Tensor | None]:
    """Compute policy and shaping masks together, at most once per state."""

    mobility_mask = None
    if mode in ("observable", "observable-hard") or need_mobility_mask:
        hard, conservative = observable_masks_fn(obs)
        mobility_mask = conservative if need_mobility_mask else None
        if mode == "observable":
            return conservative, True, mobility_mask
        if mode == "observable-hard":
            return hard, True, mobility_mask
    policy_mask, use_action_mask = training_action_mask(
        obs,
        server_legal_mask,
        mode,
        observable_masks_fn,
    )
    return policy_mask, use_action_mask, mobility_mask


def action_permutation(device: torch.device) -> torch.Tensor:
    device = torch.device(device)
    perm = _ACTION_PERM_CACHE.get(device)
    if perm is None:
        perm = torch.tensor(MODEL_TO_CUDA_ACTION, dtype=torch.long, device=device)
        _ACTION_PERM_CACHE[device] = perm
    return perm


def legal_cuda_to_model(legal_mask: torch.Tensor) -> torch.Tensor:
    return legal_mask.index_select(-1, action_permutation(legal_mask.device))


def actions_model_to_cuda(actions: torch.Tensor) -> torch.Tensor:
    perm = action_permutation(actions.device)
    return perm[actions.long()].to(torch.int32)


def actions_cuda_to_model(actions: torch.Tensor) -> torch.Tensor:
    """Convert CUDA order (UP, RIGHT, DOWN, LEFT) to model action indices."""

    device = torch.device(actions.device)
    inverse = _INVERSE_ACTION_PERM_CACHE.get(device)
    if inverse is None:
        inverse = torch.empty(4, dtype=torch.long, device=device)
        perm = action_permutation(device)
        inverse[perm] = torch.arange(4, device=device)
        _INVERSE_ACTION_PERM_CACHE[device] = inverse
    return inverse[actions.long()]


def rollout_obs_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported rollout obs dtype: {name}")


def amp_dtype(name: str) -> torch.dtype | None:
    if name == "float32":
        return None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported AMP dtype: {name}")


def environment_obs_dtype(
    rollout_dtype: torch.dtype,
    compute_dtype: torch.dtype | None,
) -> torch.dtype:
    """Keep FP32 observations when the policy itself runs without AMP."""

    return torch.float32 if compute_dtype is None else rollout_dtype


def lr_for_update(
    args,
    update: int,
    start_lr: float | None = None,
    update_offset: int = 0,
) -> float:
    """Return the LR for one local update without surprising continuations.

    ``cosine`` is monotonic and is the safe default.  ``cosine-restarts`` is
    retained only so old command lines remain reproducible; it intentionally
    jumps at one and two thirds of a run.
    """

    peak = args.lr
    if args.lr_schedule == "constant":
        return peak if start_lr is None else min(peak, start_lr)
    floor = min(args.lr_floor, peak)
    decay_updates = getattr(args, "lr_decay_updates", None) or args.updates
    schedule_update = update_offset + update
    progress = min(
        max((schedule_update - 1) / max(decay_updates - 1, 1), 0.0),
        1.0,
    )
    if args.lr_schedule == "cosine":
        scheduled = floor + 0.5 * (peak - floor) * (
            1.0 + math.cos(math.pi * progress)
        )
        return scheduled if start_lr is None else min(scheduled, start_lr)
    if args.lr_schedule != "cosine-restarts":
        raise ValueError(f"unsupported LR schedule: {args.lr_schedule}")
    progress = min(progress, 1.0 - 1e-9)
    peaks = (peak, peak * 0.5, peak * 0.25)
    cycle, cycle_progress = divmod(progress * len(peaks), 1.0)
    peak = peaks[int(cycle)]
    scheduled = floor + 0.5 * (peak - floor) * (
        1.0 + math.cos(math.pi * cycle_progress)
    )
    return scheduled if start_lr is None else min(scheduled, start_lr)


def entropy_coef_for_update(args, update: int, update_offset: int = 0) -> float:
    """Cosine-decay entropy regularization while preserving old constant runs."""

    start = float(args.ent_coef)
    final_arg = getattr(args, "ent_coef_final", None)
    if final_arg is None:
        return start
    final = float(final_arg)
    decay_updates = getattr(args, "ent_decay_updates", None) or args.updates
    schedule_update = update_offset + update
    progress = min(
        max((schedule_update - 1) / max(decay_updates - 1, 1), 0.0),
        1.0,
    )
    return final + 0.5 * (start - final) * (
        1.0 + math.cos(math.pi * progress)
    )


def phase_schedule_offset(
    completed_updates: int,
    schedule_origin_update: int | None,
    *,
    resumed: bool,
) -> int:
    """Map an absolute checkpoint counter onto a phase-local schedule.

    Historical runs treated every resumed checkpoint as part of a schedule
    that started at absolute update zero.  A new training phase warm-started
    from a mature policy therefore reached both the LR and entropy floors on
    its first update whenever ``completed_updates > decay_updates``.  An
    explicit origin lets a phase keep its own monotonic schedule across any
    number of interruptions without resetting model provenance counters.

    Omitting the option preserves the legacy single-run resume behaviour.
    """

    completed = int(completed_updates)
    if schedule_origin_update is None:
        return completed if resumed else 0
    origin = int(schedule_origin_update)
    if origin < 0:
        raise ValueError("--schedule-origin-update must be non-negative")
    if origin > completed:
        raise ValueError(
            "--schedule-origin-update cannot exceed the loaded checkpoint update"
        )
    return completed - origin


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr


def _sequence_lstm(cell: nn.LSTMCell) -> nn.LSTM:
    """Expose an LSTMCell's parameters through cuDNN's sequence interface.

    The adapter is deliberately kept outside the module tree: its parameters
    are the exact same Parameter objects as the LSTMCell's, so checkpoints and
    the SB3 export format remain unchanged and the optimizer sees no duplicates.
    """

    sequence = _SEQUENCE_LSTM_CACHE.get(cell)
    if sequence is None:
        sequence = nn.LSTM(
            cell.input_size,
            cell.hidden_size,
            device=cell.weight_ih.device,
            dtype=cell.weight_ih.dtype,
        )
        sequence.weight_ih_l0 = cell.weight_ih
        sequence.weight_hh_l0 = cell.weight_hh
        sequence.bias_ih_l0 = cell.bias_ih
        sequence.bias_hh_l0 = cell.bias_hh
        sequence.flatten_parameters()
        _SEQUENCE_LSTM_CACHE[cell] = sequence
    sequence.train(cell.training)
    return sequence


def _episode_segments(
    dones: torch.Tensor,
    output_device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Build a padded episode layout while preserving every recurrent reset."""

    t_steps, batch = dones.shape
    if t_steps <= 0 or batch <= 0:
        raise ValueError("dones must have non-empty time and batch dimensions")
    done_cpu = dones.detach().to(device="cpu", dtype=torch.bool)
    # A segment starts at t=0 and immediately after every terminal transition.
    # Vectorizing this layout replaces hundreds of per-environment ``nonzero``
    # calls and Python list appends in every PPO minibatch.
    starts = torch.empty_like(done_cpu)
    starts[0] = True
    if t_steps > 1:
        starts[1:] = done_cpu[:-1]
    segment_counts = starts.sum(dim=0, dtype=torch.long)
    segment_offsets = segment_counts.cumsum(dim=0) - segment_counts
    segment_ids = (
        starts.long().cumsum(dim=0)
        - 1
        + segment_offsets.view(1, batch)
    )
    times = torch.arange(t_steps, dtype=torch.long).view(t_steps, 1).expand(
        t_steps, batch
    )
    last_start = torch.cummax(
        torch.where(starts, times, 0),
        dim=0,
    ).values
    steps_in_segment = times - last_start

    # Environment-major order already places every segment contiguously and
    # retains chronological order inside it.
    source_indices = (
        torch.arange(t_steps * batch, dtype=torch.long)
        .view(t_steps, batch)
        .T.reshape(-1)
    )
    ordered_segments = segment_ids.T.reshape(-1)
    num_segments = int(segment_counts.sum())
    max_length = int(
        torch.bincount(ordered_segments, minlength=num_segments).max()
    )
    padded_indices = (
        steps_in_segment.T.reshape(-1) * num_segments + ordered_segments
    )
    first_segment_indices = segment_offsets
    first_env_indices = torch.arange(batch, dtype=torch.long)
    device = output_device or dones.device
    return (
        source_indices.to(device=device),
        padded_indices.to(device=device),
        first_segment_indices.to(device=device),
        first_env_indices.to(device=device),
        max_length,
        num_segments,
    )


def _forward_segmented_lstm(
    cell: nn.LSTMCell,
    features: torch.Tensor,
    initial_h: torch.Tensor,
    initial_c: torch.Tensor,
    layout: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int],
) -> torch.Tensor:
    """Run all uninterrupted episode fragments in one cuDNN LSTM call."""

    source, padded_index, first_segment, first_env, max_length, num_segments = layout
    t_steps, batch, feature_size = features.shape
    padded = features.new_zeros((max_length, num_segments, feature_size))
    padded.reshape(-1, feature_size).index_copy_(
        0,
        padded_index,
        features.reshape(-1, feature_size).index_select(0, source),
    )

    h0 = initial_h.new_zeros((num_segments, initial_h.shape[-1]))
    c0 = initial_c.new_zeros((num_segments, initial_c.shape[-1]))
    h0.index_copy_(0, first_segment, initial_h.index_select(0, first_env))
    c0.index_copy_(0, first_segment, initial_c.index_select(0, first_env))
    output, _ = _sequence_lstm(cell)(padded, (h0.unsqueeze(0), c0.unsqueeze(0)))

    flat_output = output.new_empty((t_steps * batch, output.shape[-1]))
    flat_output.index_copy_(
        0,
        source,
        output.reshape(-1, output.shape[-1]).index_select(0, padded_index),
    )
    return flat_output.reshape(t_steps, batch, output.shape[-1])


def _forward_segmented_lstm_pair(
    actor_cell: nn.LSTMCell,
    critic_cell: nn.LSTMCell,
    actor_features: torch.Tensor,
    critic_features: torch.Tensor,
    actor_h: torch.Tensor,
    actor_c: torch.Tensor,
    critic_h: torch.Tensor,
    critic_c: torch.Tensor,
    layout: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run actor and critic from one shared episode-fragment packing.

    The two branches consume different feature streams (the critic's CNN sees
    the privileged channels) but share the same episode segmentation.
    """

    source, padded_index, first_segment, first_env, max_length, num_segments = layout
    t_steps, batch, feature_size = actor_features.shape

    def pack(features: torch.Tensor) -> torch.Tensor:
        padded = features.new_zeros((max_length, num_segments, feature_size))
        padded.reshape(-1, feature_size).index_copy_(
            0,
            padded_index,
            features.reshape(-1, feature_size).index_select(0, source),
        )
        return padded

    def run_branch(
        cell: nn.LSTMCell,
        padded: torch.Tensor,
        initial_h: torch.Tensor,
        initial_c: torch.Tensor,
    ) -> torch.Tensor:
        h0 = initial_h.new_zeros((num_segments, initial_h.shape[-1]))
        c0 = initial_c.new_zeros((num_segments, initial_c.shape[-1]))
        h0.index_copy_(0, first_segment, initial_h.index_select(0, first_env))
        c0.index_copy_(0, first_segment, initial_c.index_select(0, first_env))
        output, _ = _sequence_lstm(cell)(
            padded, (h0.unsqueeze(0), c0.unsqueeze(0))
        )
        flat_output = output.new_empty((t_steps * batch, output.shape[-1]))
        flat_output.index_copy_(
            0,
            source,
            output.reshape(-1, output.shape[-1]).index_select(0, padded_index),
        )
        return flat_output.reshape(t_steps, batch, output.shape[-1])

    return (
        run_branch(actor_cell, pack(actor_features), actor_h, actor_c),
        run_branch(critic_cell, pack(critic_features), critic_h, critic_c),
    )


def forward_sequence(
    model: ActorCritic,
    obs: torch.Tensor,
    dones: torch.Tensor,
    actor_h: torch.Tensor,
    actor_c: torch.Tensor,
    critic_h: torch.Tensor,
    critic_c: torch.Tensor,
    episode_layout: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int
    ]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one recurrent minibatch shaped T x B x C x H x W."""

    t_steps, batch = obs.shape[:2]
    # The rollout is already FP16 when AMP is enabled. Keeping it in that dtype
    # avoids an FP16 -> FP32 gather followed immediately by Autocast's FP32 ->
    # FP16 conversion inside the first convolution.
    flat_obs = obs.reshape(t_steps * batch, *OBS_SHAPE)
    features = model.training_cnn_forward(model.policy_obs(flat_obs))
    features = features.reshape(t_steps, batch, model.feature_size)
    critic_features = model.training_critic_cnn_forward(flat_obs)
    critic_features = critic_features.reshape(t_steps, batch, model.feature_size)
    layout = episode_layout or _episode_segments(dones)
    actor_output, critic_output = _forward_segmented_lstm_pair(
        model.actor_lstm,
        model.critic_lstm,
        features,
        critic_features,
        actor_h,
        actor_c,
        critic_h,
        critic_c,
        layout,
    )
    return model.actor(actor_output), model.critic(critic_output).squeeze(-1)


def random_legal_actions(legal_mask: torch.Tensor) -> torch.Tensor:
    scores = torch.rand(legal_mask.shape, device=legal_mask.device)
    scores = torch.where(legal_mask, scores, torch.full_like(scores, -1.0))
    return scores.argmax(dim=-1)


def hungry_legal_actions(obs: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Simple CUDA Hungry opponent: move toward nearest visible food if legal."""

    device = obs.device
    n = obs.shape[0]
    food = obs[:, 0] > 0.0
    coords = torch.arange(29, device=device)
    xs = coords.view(1, 29, 1)
    ys = coords.view(1, 1, 29)
    dist = (xs - 14).abs() + (ys - 14).abs()
    masked_dist = torch.where(food, dist, torch.full_like(dist, 10_000))
    flat = masked_dist.reshape(n, -1).argmin(dim=-1)
    has_food = food.reshape(n, -1).any(dim=-1)
    target_x = flat // 29
    target_y = flat % 29
    dx = target_x - 14
    dy = target_y - 14
    horizontal = dx.abs() >= dy.abs()
    desired = torch.where(
        horizontal,
        torch.where(dx >= 0, torch.full_like(dx, 3), torch.full_like(dx, 2)),
        torch.where(dy >= 0, torch.full_like(dy, 0), torch.full_like(dy, 1)),
    )
    random_actions = random_legal_actions(legal_mask)
    desired_legal = legal_mask.gather(1, desired.unsqueeze(1)).squeeze(1)
    return torch.where(has_food & desired_legal, desired, random_actions)


def observation_potential(
    obs: torch.Tensor,
    length_coef: float,
    health_coef: float,
    mobility_coef: float = 0.0,
    mobility_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bounded state potential based only on information available to the agent.

    The centered own-body value is ``length / 10`` and channel four contains
    normalized health. Observable mobility penalizes entering locally forced
    lines before the eventual death reward arrives. Using a potential
    difference provides dense feedback without introducing a farmable per-turn
    bonus or changing the terminal objective.
    """

    length = obs[:, 2, 14, 14].float() * 10.0
    health = obs[:, 4, 14, 14].float().clamp(0.0, 1.0)
    potential = (
        length_coef * torch.tanh((length - 3.0) / 6.0)
        + health_coef * health
    )
    if mobility_coef != 0.0:
        if mobility_mask is None:
            mobility_mask = observable_action_mask(obs)
        safe_moves = mobility_mask.sum(dim=-1).float()
        mobility = ((safe_moves - 1.0) / 3.0).clamp(0.0, 1.0)
        potential = potential + mobility_coef * mobility
    return potential


def potential_shaped_reward_from_values(
    env_reward: torch.Tensor,
    current_potential: torch.Tensor | None,
    next_potential: torch.Tensor | None,
    done: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Apply cached potential values without re-reading either observation."""

    if current_potential is None or next_potential is None:
        return env_reward
    return (
        env_reward
        + gamma * next_potential * (~done).float()
        - current_potential
    )


def potential_shaped_reward(
    env_reward: torch.Tensor,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    length_coef: float,
    health_coef: float,
    mobility_coef: float = 0.0,
) -> torch.Tensor:
    """Apply ``gamma * Phi(s') - Phi(s)`` without leaking auto-reset states."""

    if length_coef == 0.0 and health_coef == 0.0 and mobility_coef == 0.0:
        return env_reward
    current = observation_potential(
        obs, length_coef, health_coef, mobility_coef
    )
    following = observation_potential(
        next_obs, length_coef, health_coef, mobility_coef
    )
    return potential_shaped_reward_from_values(
        env_reward,
        current,
        following,
        done,
        gamma,
    )


def environment_reward_for_scheme(
    env_reward: torch.Tensor,
    done: torch.Tensor,
    scheme: str,
    alive_before: torch.Tensor,
    reset_alive_count: int | torch.Tensor = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the native rank signal and track the number of surviving snakes.

    The native four-player reward is an incremental, normalized rank signal:
    a surviving learner gets ``(before-after)/3`` and an eliminated learner
    gets ``-after/3``.  This lets us recover ``after`` without another native
    output.

    Tournament points are paid as soon as they become guaranteed: one point
    upon reaching the final two and one more for winning. Their undiscounted
    episode sum is still exactly 2/1/0/0. Paying both points only at the end
    made long wins nearly worthless under ``gamma < 1`` and caused the policy
    to regress toward quick second places.
    """

    if scheme not in ("kill", "tournament"):
        raise ValueError(f"unsupported reward scheme: {scheme}")
    alive_before = alive_before.to(dtype=torch.long, device=env_reward.device)
    rank_delta = (env_reward * 3.0).round().to(torch.long)
    alive_after = torch.where(
        env_reward > 0.0,
        alive_before - rank_delta,
        torch.where(
            env_reward < 0.0,
            -rank_delta,
            torch.where(done, torch.zeros_like(alive_before), alive_before),
        ),
    ).clamp_(0, 4)
    reset_alive = torch.as_tensor(
        reset_alive_count,
        dtype=torch.long,
        device=env_reward.device,
    ).expand_as(alive_after)
    next_alive = torch.where(done, reset_alive, alive_after)
    if scheme == "kill":
        return env_reward, next_alive

    def guaranteed_points(alive: torch.Tensor) -> torch.Tensor:
        return torch.where(
            alive <= 1,
            torch.full_like(alive, 2),
            torch.where(alive == 2, torch.ones_like(alive), torch.zeros_like(alive)),
        )

    learner_survived = (~done) | (env_reward > 0.0)
    points_before = guaranteed_points(alive_before)
    points_after = torch.where(
        learner_survived,
        guaranteed_points(alive_after),
        (alive_after == 1).to(torch.long),
    )
    tournament_reward = (points_after - points_before).to(env_reward.dtype)
    return tournament_reward, next_alive


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimates for an auto-reset vector environment."""

    n_steps, num_envs = rewards.shape
    advantages = torch.empty_like(rewards)
    last_gae = torch.zeros(num_envs, device=rewards.device)
    for t in reversed(range(n_steps)):
        next_value = next_values if t == n_steps - 1 else values[t + 1]
        next_nonterminal = (~dones[t]).float()
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


POOL_RANDOM = -3
POOL_HUNGRY = -2
POOL_BEST = -1
POOL_CHECKPOINT_CATEGORY = -4
POOL_FORAGER = -5
POOL_HUNTER = -6
POOL_TERRITORIAL = -7
POOL_EDGE_TRAPPER = -8
POOL_SURVIVOR = -9
POOL_SNAKE25 = -10
POOL_SNAKE25_INTERCEPTOR = -11
POOL_SNAKE25_DENIER = -12
POOL_SNAKE25_DUELIST = -13
LOG_DERIVED_HEURISTICS = (
    ("forager", POOL_FORAGER, hisss.BLACKOUT_HEURISTIC_FORAGER),
    ("hunter", POOL_HUNTER, hisss.BLACKOUT_HEURISTIC_HUNTER),
    ("territorial", POOL_TERRITORIAL, hisss.BLACKOUT_HEURISTIC_TERRITORIAL),
    ("edge_trapper", POOL_EDGE_TRAPPER, hisss.BLACKOUT_HEURISTIC_EDGE_TRAPPER),
    ("survivor", POOL_SURVIVOR, hisss.BLACKOUT_HEURISTIC_SURVIVOR),
    ("snake25", POOL_SNAKE25, hisss.BLACKOUT_HEURISTIC_SNAKE25),
    (
        "snake25_interceptor",
        POOL_SNAKE25_INTERCEPTOR,
        hisss.BLACKOUT_HEURISTIC_SNAKE25_INTERCEPTOR,
    ),
    (
        "snake25_denier",
        POOL_SNAKE25_DENIER,
        hisss.BLACKOUT_HEURISTIC_SNAKE25_DENIER,
    ),
    (
        "snake25_duelist",
        POOL_SNAKE25_DUELIST,
        hisss.BLACKOUT_HEURISTIC_SNAKE25_DUELIST,
    ),
)
SCRIPTED_OPPONENTS = (
    ("best", POOL_BEST),
    ("hungry", POOL_HUNGRY),
    ("random", POOL_RANDOM),
    *((label, opponent_id) for label, opponent_id, _ in LOG_DERIVED_HEURISTICS),
)
SCRIPTED_OPPONENT_LABELS = dict(
    (opponent_id, label) for label, opponent_id in SCRIPTED_OPPONENTS
)
NATIVE_PROFILE_BY_POOL_ID = {
    opponent_id: profile_id
    for _, opponent_id, profile_id in LOG_DERIVED_HEURISTICS
}
NATIVE_PROFILE_BY_LABEL = {
    label: profile_id for label, _, profile_id in LOG_DERIVED_HEURISTICS
}
CHECKPOINT_SCHEMA_VERSION = 2
LEAGUE_CHAMPION_LABEL = "league_champion"
ANCHOR_LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
CHECKPOINT_NAME_RE = re.compile(r"_steps_(\d+)_u(\d+)")


def parse_anchor_specs(specs: list[str] | None) -> list[tuple[str, Path]]:
    """Parse repeatable ``LABEL=PATH`` anchor arguments."""

    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(
                f"invalid anchor {spec!r}; expected LABEL=PATH"
            )
        label, raw_path = spec.split("=", 1)
        if not label or not ANCHOR_LABEL_RE.fullmatch(label):
            raise ValueError(
                f"invalid anchor label {label!r}; use letters, digits, '.', '-', or '_'"
            )
        if label in seen:
            raise ValueError(f"duplicate anchor label: {label}")
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"anchor model does not exist: {path}")
        seen.add(label)
        result.append((label, path))
    return result


def parse_labeled_scores(
    specs: list[str] | None,
    *,
    option_name: str,
) -> dict[str, float]:
    """Parse repeatable ``LABEL=SCORE`` arguments with strict validation."""

    result: dict[str, float] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(
                f"invalid {option_name} {spec!r}; expected LABEL=SCORE"
            )
        label, raw_score = spec.split("=", 1)
        if not label or not ANCHOR_LABEL_RE.fullmatch(label):
            raise ValueError(
                f"invalid {option_name} label {label!r}; use letters, digits, "
                "'.', '-', or '_'"
            )
        if label in result:
            raise ValueError(f"duplicate {option_name} label: {label}")
        try:
            score = float(raw_score)
        except ValueError as exc:
            raise ValueError(
                f"invalid {option_name} score {raw_score!r} for {label}"
            ) from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"{option_name} score for {label} must be finite and in [0, 1]"
            )
        result[label] = score
    return result


def parse_heuristic_weights(specs: list[str] | None) -> dict[str, float]:
    """Parse exact per-profile ``LABEL=WEIGHT`` opponent-pool floors."""

    result: dict[str, float] = {}
    valid_labels = set(NATIVE_PROFILE_BY_LABEL)
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(
                f"invalid heuristic weight {spec!r}; expected LABEL=WEIGHT"
            )
        label, raw_weight = spec.split("=", 1)
        if label not in valid_labels:
            raise ValueError(
                f"unknown heuristic profile {label!r}; choose one of "
                + ", ".join(sorted(valid_labels))
            )
        if label in result:
            raise ValueError(f"duplicate heuristic weight label: {label}")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(
                f"invalid heuristic weight {raw_weight!r} for {label}"
            ) from exc
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"heuristic weight for {label} must be finite and non-negative"
            )
        result[label] = weight
    return result


def checkpoint_counters_from_name(path: Path) -> tuple[int, int]:
    """Recover informational counters from a named exported checkpoint."""

    match = CHECKPOINT_NAME_RE.search(path.stem)
    if match is None:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def seed_privileged_critic_cnn(
    model: ActorCritic,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Seed ``critic_cnn`` from the actor CNN for pre-privileged checkpoints.

    conv0 is zero-padded across the privileged input channels, so the seeded
    critic computes bit-identical values to the source's shared-CNN critic
    until training learns to use the unfogged planes.
    """

    if "critic_cnn.0.weight" in state_dict:
        return state_dict
    seeded = dict(state_dict)
    for key, value in state_dict.items():
        if key.startswith("cnn.") and key != "cnn.0.weight":
            seeded["critic_" + key] = value.clone()
    conv0 = state_dict["cnn.0.weight"]
    target_shape = model.critic_cnn[0].weight.shape
    padded = conv0.new_zeros(target_shape)
    padded[:, : conv0.shape[1]] = conv0
    seeded["critic_cnn.0.weight"] = padded
    return seeded


def load_model_state(
    model: ActorCritic,
    state_dict: dict[str, torch.Tensor],
    *,
    allow_legacy: bool = True,
) -> bool:
    """Load a v2 state dict, optionally upgrading the old shared-LSTM format.

    Checkpoints from before the privileged critic carry no ``critic_cnn``;
    those are upgraded in-flight via :func:`seed_privileged_critic_cnn`.

    Returns whether a legacy state was upgraded.  Old recurrent weights seed
    both new branches; their old ReLU-head semantics cannot be preserved and a
    caller should warn when using such a checkpoint for continued training.
    """

    legacy = "lstm.weight_ih" in state_dict
    if not legacy:
        model.load_state_dict(
            seed_privileged_critic_cnn(model, state_dict), strict=True
        )
        return False
    if not allow_legacy:
        raise ValueError("legacy shared-LSTM checkpoint is not valid for this pool")

    upgraded = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("lstm.")
    }
    for suffix in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
        value = state_dict[f"lstm.{suffix}"]
        upgraded[f"actor_lstm.{suffix}"] = value
        upgraded[f"critic_lstm.{suffix}"] = value.clone()
    model.load_state_dict(
        seed_privileged_critic_cnn(model, upgraded), strict=True
    )
    return True


def checkpoint_model_state(payload) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict) and payload and all(
        isinstance(key, str) for key in payload
    ):
        return payload
    raise ValueError("checkpoint does not contain a model state dict")


def comparison_score(comparison: dict[str, float]) -> float:
    """Return A's match points per game, counting a draw as half a point."""

    games = float(comparison.get("games", 0.0))
    if games <= 0.0:
        raise ValueError("comparison must contain a positive game count")
    return (
        float(comparison.get("a_wins", 0.0))
        + 0.5 * float(comparison.get("draws", 0.0))
    ) / games


def aggregate_comparison_score(comparisons: list[dict[str, float]]) -> float:
    """Aggregate match points before dividing, so every game has equal weight."""

    if not comparisons:
        raise ValueError("at least one comparison is required")
    games = sum(float(comparison.get("games", 0.0)) for comparison in comparisons)
    if games <= 0.0:
        raise ValueError("comparisons must contain a positive total game count")
    points = sum(
        float(comparison.get("a_wins", 0.0))
        + 0.5 * float(comparison.get("draws", 0.0))
        for comparison in comparisons
    )
    return points / games


def solve_zero_sum_nash_distribution(
    payoff_matrix: torch.Tensor,
    *,
    iterations: int = 2_000,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Approximate both mixed strategies of a rectangular zero-sum game.

    Rows maximize the learner's match score while columns minimize it.  The
    averaged multiplicative-weights iterates converge to the minimax/Nash set
    and need no SciPy dependency.  Returning both strategies makes the helper
    directly testable even though training consumes only the opponent mix.
    """

    matrix = torch.as_tensor(payoff_matrix, dtype=torch.float64, device="cpu")
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("payoff_matrix must be a non-empty 2-D tensor")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError("payoff_matrix must contain only finite values")

    # Centering preserves the equilibrium while making the update magnitude
    # independent of the conventional [0, 1] match-score offset.
    matrix = matrix.clamp(0.0, 1.0) - 0.5
    n_rows, n_columns = matrix.shape
    scale = max(2, n_rows, n_columns)
    learning_rate = min(0.5, math.sqrt(2.0 * math.log(scale) / iterations))
    row_logits = torch.zeros(n_rows, dtype=torch.float64)
    column_logits = torch.zeros(n_columns, dtype=torch.float64)
    average_rows = torch.zeros_like(row_logits)
    average_columns = torch.zeros_like(column_logits)

    for _ in range(iterations):
        rows = torch.softmax(row_logits, dim=0)
        columns = torch.softmax(column_logits, dim=0)
        average_rows += rows
        average_columns += columns
        row_logits += learning_rate * (matrix @ columns)
        column_logits -= learning_rate * (rows @ matrix)
        # Softmax is shift invariant; recentering avoids needless growth over
        # long solves and keeps resumed results numerically reproducible.
        row_logits -= row_logits.mean()
        column_logits -= column_logits.mean()

    row_distribution = average_rows / average_rows.sum()
    column_distribution = average_columns / average_columns.sum()
    value = float(
        (row_distribution @ matrix @ column_distribution).item() + 0.5
    )
    return row_distribution.float(), column_distribution.float(), value


def league_promotion_decision(
    comparisons: list[dict[str, float]],
    evaluation_results: list[dict[str, float | str]],
    *,
    threshold: float,
    guard_labels: list[str],
    min_guard_score: float,
    guard_thresholds: dict[str, float] | None = None,
) -> tuple[bool, float, float | None, str]:
    """Apply the champion gate and explain a rejection in one compact reason."""

    direct_score = aggregate_comparison_score(comparisons)
    anchor_scores = {
        str(result["label"]): float(result["current_win_rate"])
        + 0.5 * float(result["draw_rate"])
        for result in evaluation_results
        if result.get("kind") in ("anchor", "heuristic")
    }
    guard_thresholds = guard_thresholds or {}
    required_labels = list(dict.fromkeys([*guard_labels, *guard_thresholds]))
    missing = [label for label in required_labels if label not in anchor_scores]
    if missing:
        return (
            False,
            direct_score,
            None,
            "missing guard evaluation: " + ", ".join(missing),
        )
    guard_score = (
        min(anchor_scores[label] for label in required_labels)
        if required_labels
        else None
    )
    if direct_score < threshold:
        return (
            False,
            direct_score,
            guard_score,
            f"direct score {direct_score:.3f} < {threshold:.3f}",
        )
    failed_guards = [
        (label, anchor_scores[label], guard_thresholds.get(label, min_guard_score))
        for label in required_labels
        if anchor_scores[label] < guard_thresholds.get(label, min_guard_score)
    ]
    if failed_guards:
        label, score, required_score = min(
            failed_guards,
            key=lambda item: item[1] - item[2],
        )
        return (
            False,
            direct_score,
            guard_score,
            f"guard score for {label} {score:.3f} < {required_score:.3f}",
        )
    return True, direct_score, guard_score, "promotion gate passed"


def league_generalization_decision(
    evaluation_results: list[dict[str, float | str]],
    champion_anchor_scores: dict[str, float],
    *,
    min_mean_improvement: float,
    max_anchor_regression: float,
) -> tuple[bool, float | None, float | None, str]:
    """Compare a candidate with the champion on the same fixed anchor suite.

    A direct head-to-head win can be a cyclic counterstrategy rather than a
    generally stronger policy.  This second gate requires the mean fixed-anchor
    score to improve while bounding every individual anchor regression.
    """

    if max_anchor_regression < 0.0:
        raise ValueError("max_anchor_regression must be non-negative")
    candidate_scores = {
        str(result["label"]): float(result["current_win_rate"])
        + 0.5 * float(result["draw_rate"])
        for result in evaluation_results
        if result.get("kind") == "anchor"
        and str(result.get("label")) != LEAGUE_CHAMPION_LABEL
    }
    labels = sorted(champion_anchor_scores)
    if not labels:
        return False, None, None, "champion fixed-anchor baseline is empty"
    missing = [label for label in labels if label not in candidate_scores]
    if missing:
        return (
            False,
            None,
            None,
            "missing relative anchor evaluation: " + ", ".join(missing),
        )
    deltas = [
        candidate_scores[label] - champion_anchor_scores[label]
        for label in labels
    ]
    mean_improvement = sum(deltas) / len(deltas)
    worst_regression = min(deltas)
    if mean_improvement < min_mean_improvement:
        return (
            False,
            mean_improvement,
            worst_regression,
            f"fixed-anchor mean delta {mean_improvement:+.3f} < "
            f"{min_mean_improvement:+.3f}",
        )
    if worst_regression < -max_anchor_regression:
        worst_label = min(
            labels,
            key=lambda label: (
                candidate_scores[label] - champion_anchor_scores[label]
            ),
        )
        return (
            False,
            mean_improvement,
            worst_regression,
            f"relative score for {worst_label} regressed by "
            f"{-worst_regression:.3f} > {max_anchor_regression:.3f}",
        )
    return (
        True,
        mean_improvement,
        worst_regression,
        "fixed-anchor generalization gate passed",
    )


class CudaOpponentPool:
    """Episode-stable mixture of frozen policies and CUDA scripted agents."""

    def __init__(
        self,
        model_factory: Callable[[], ActorCritic],
        num_envs: int,
        hidden_size: int,
        device: torch.device,
        pool_dir: str,
        max_checkpoints: int,
        checkpoint_weight: float,
        best_weight: float,
        hungry_weight: float,
        random_weight: float,
        latest_probability: float,
        seed: int,
        load_existing: bool,
        real_heuristic_weight: float = 0.0,
        heuristic_weights: dict[str, float] | None = None,
        inference_amp_dtype: torch.dtype | None = None,
        active_checkpoint_limit: int = 2,
        active_rotation_interval: int = 1,
        anchor_probability: float = 0.25,
        focus_label: str | None = None,
        focus_probability: float = 0.0,
        deterministic_probability: float = 0.0,
        copies_probability: float = 0.0,
        anchor_targets: dict[str, float] | None = None,
        anchor_priority_exponent: float = 2.0,
        anchor_min_weight: float = 0.05,
        anchor_score_ema: float = 0.5,
        anchor_uniform_floor: float = 0.0,
        nash_history_size: int = 16,
        nash_iterations: int = 2_000,
        nash_exploration: float = 0.05,
        nash_score_half_life_updates: int = 0,
        champion_min_weight: float = 0.0,
        deduplicate_champion: bool = False,
        action_mask_mode: str = "observable",
        parallel_inference: bool = False,
        min_action_disagreement: float = 0.0,
        diversity_probe_size: int = 512,
        max_snapshot_update: int | None = None,
        duel_opponent_label: str | None = None,
        duel_opponent_probability: float = 0.0,
        duel_opponent_weights: dict[str, float] | None = None,
    ):
        if not 1 <= max_checkpoints <= 256:
            raise ValueError("max_checkpoints must be between 1 and 256")
        weights = (checkpoint_weight, best_weight, hungry_weight, random_weight)
        if heuristic_weights is None:
            per_profile = real_heuristic_weight / len(LOG_DERIVED_HEURISTICS)
            resolved_heuristic_weights = {
                label: per_profile for label, _, _ in LOG_DERIVED_HEURISTICS
            }
        else:
            if real_heuristic_weight != 0.0:
                raise ValueError(
                    "real_heuristic_weight and heuristic_weights are mutually "
                    "exclusive"
                )
            unknown_profiles = set(heuristic_weights) - set(
                NATIVE_PROFILE_BY_LABEL
            )
            if unknown_profiles:
                raise ValueError(
                    "unknown heuristic profiles: "
                    + ", ".join(sorted(unknown_profiles))
                )
            resolved_heuristic_weights = {
                label: float(heuristic_weights.get(label, 0.0))
                for label, _, _ in LOG_DERIVED_HEURISTICS
            }
        heuristic_total_weight = sum(resolved_heuristic_weights.values())
        all_category_weights = (*weights, *resolved_heuristic_weights.values())
        if (
            any(weight < 0.0 for weight in all_category_weights)
            or sum(all_category_weights) <= 0.0
        ):
            raise ValueError("opponent-pool weights must be non-negative and non-zero")
        if not 0.0 <= latest_probability <= 1.0:
            raise ValueError("latest_probability must be in [0, 1]")
        if not 0.0 <= anchor_probability <= 1.0:
            raise ValueError("anchor_probability must be in [0, 1]")
        if not 0.0 <= focus_probability <= 1.0:
            raise ValueError("focus_probability must be in [0, 1]")
        if not 0.0 <= deterministic_probability <= 1.0:
            raise ValueError("deterministic_probability must be in [0, 1]")
        if not 0.0 <= copies_probability <= 1.0:
            raise ValueError("copies_probability must be in [0, 1]")
        if anchor_priority_exponent <= 0.0:
            raise ValueError("anchor_priority_exponent must be positive")
        if not 0.0 < anchor_min_weight <= 1.0:
            raise ValueError("anchor_min_weight must be in (0, 1]")
        if not 0.0 < anchor_score_ema <= 1.0:
            raise ValueError("anchor_score_ema must be in (0, 1]")
        if anchor_uniform_floor < 0.0:
            raise ValueError("anchor_uniform_floor must be non-negative")
        if not 1 <= active_checkpoint_limit <= min(max_checkpoints, 32):
            raise ValueError(
                "active_checkpoint_limit must be between 1 and min(max_checkpoints, 32)"
            )
        if active_rotation_interval <= 0:
            raise ValueError("active_rotation_interval must be positive")
        if nash_history_size <= 0:
            raise ValueError("nash_history_size must be positive")
        if nash_iterations <= 0:
            raise ValueError("nash_iterations must be positive")
        if not 0.0 <= nash_exploration < 1.0:
            raise ValueError("nash_exploration must be in [0, 1)")
        if nash_score_half_life_updates < 0:
            raise ValueError("nash_score_half_life_updates must be non-negative")
        checkpoint_share = checkpoint_weight / sum(all_category_weights)
        if not 0.0 <= champion_min_weight <= checkpoint_share:
            raise ValueError(
                "champion_min_weight must be between zero and the normalized "
                "checkpoint weight"
            )
        if anchor_uniform_floor + champion_min_weight > checkpoint_share:
            raise ValueError(
                "anchor_uniform_floor plus champion_min_weight cannot exceed "
                "the normalized checkpoint weight"
            )
        if not 0.0 <= duel_opponent_probability <= 1.0:
            raise ValueError("duel_opponent_probability must be in [0, 1]")
        resolved_duel_opponent_weights = dict(duel_opponent_weights or {})
        if resolved_duel_opponent_weights and (
            duel_opponent_label is not None or duel_opponent_probability != 0.0
        ):
            raise ValueError(
                "duel_opponent_weights and the legacy duel opponent options "
                "are mutually exclusive"
            )
        if any(
            not math.isfinite(weight) or not 0.0 <= weight <= 1.0
            for weight in resolved_duel_opponent_weights.values()
        ):
            raise ValueError("duel opponent weights must be finite and in [0, 1]")
        if sum(resolved_duel_opponent_weights.values()) > 1.0 + 1e-9:
            raise ValueError("duel opponent weights must sum to at most one")
        for label, weight in resolved_duel_opponent_weights.items():
            if (
                weight > 0.0
                and label in NATIVE_PROFILE_BY_LABEL
                and resolved_heuristic_weights[label] <= 0.0
            ):
                raise ValueError(
                    f"native duel opponent {label!r} must have a positive "
                    "configured heuristic weight"
                )
        if duel_opponent_label is not None:
            if duel_opponent_label not in NATIVE_PROFILE_BY_LABEL:
                raise ValueError(
                    "duel_opponent_label must be one of "
                    + ", ".join(sorted(NATIVE_PROFILE_BY_LABEL))
                )
            if resolved_heuristic_weights[duel_opponent_label] <= 0.0:
                raise ValueError(
                    "duel_opponent_label must have a positive configured "
                    "heuristic weight"
                )
        if action_mask_mode not in ("observable", "observable-hard"):
            raise ValueError(
                "opponent action_mask_mode must be observable or observable-hard"
            )
        if not 0.0 <= min_action_disagreement <= 1.0:
            raise ValueError("min_action_disagreement must be in [0, 1]")
        if diversity_probe_size <= 0:
            raise ValueError("diversity_probe_size must be positive")

        self.model_factory = model_factory
        self.num_envs = num_envs
        self.hidden_size = hidden_size
        self.device = device
        self.pool_dir = Path(pool_dir)
        self.max_checkpoints = max_checkpoints
        self.weights = weights
        self.real_heuristic_weight = heuristic_total_weight
        self.heuristic_weights = resolved_heuristic_weights
        self.category_weights = {
            POOL_CHECKPOINT_CATEGORY: checkpoint_weight,
            POOL_BEST: best_weight,
            POOL_HUNGRY: hungry_weight,
            POOL_RANDOM: random_weight,
            **{
                opponent_id: resolved_heuristic_weights[label]
                for label, opponent_id, _ in LOG_DERIVED_HEURISTICS
            },
        }
        self.scripted_opponents = (
            *(
                (label, opponent_id)
                for label, opponent_id in SCRIPTED_OPPONENTS[:3]
                if self.category_weights[opponent_id] > 0.0
            ),
            *(
                (label, opponent_id)
                for label, opponent_id, _ in LOG_DERIVED_HEURISTICS
                if resolved_heuristic_weights[label] > 0.0
            ),
        )
        self.native_heuristics = tuple(
            (opponent_id, profile_id)
            for label, opponent_id, profile_id in LOG_DERIVED_HEURISTICS
            if resolved_heuristic_weights[label] > 0.0
        )
        # Native opponent ids are contiguous negative values. A tiny device
        # lookup replaces two Python loops and up to eighteen pointwise CUDA
        # launches on every simulator step when only a few profiles are active.
        self.native_profile_lut = torch.full(
            (-POOL_SNAKE25_DUELIST + 1,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        for opponent_id, profile_id in self.native_heuristics:
            self.native_profile_lut[-opponent_id] = profile_id
        self.latest_probability = latest_probability
        self.anchor_probability = anchor_probability
        self.focus_label = focus_label
        self.focus_probability = focus_probability
        self.deterministic_probability = deterministic_probability
        self.copies_probability = copies_probability
        self.anchor_targets = dict(anchor_targets or {})
        self.anchor_priority_exponent = anchor_priority_exponent
        self.anchor_min_weight = anchor_min_weight
        self.anchor_score_ema = anchor_score_ema
        self.anchor_uniform_floor = anchor_uniform_floor
        self.anchor_scores: dict[str, float] = {}
        self.nash_history_size = nash_history_size
        self.nash_iterations = nash_iterations
        self.nash_exploration = nash_exploration
        self.nash_score_half_life_updates = nash_score_half_life_updates
        self.champion_min_weight = champion_min_weight
        self.deduplicate_champion = deduplicate_champion
        self.action_mask_mode = action_mask_mode
        self.parallel_inference = parallel_inference and device.type == "cuda"
        self.nash_rows: list[dict[str, float]] = []
        self.nash_latest_scores: dict[str, float] = {}
        self.nash_score_updates: dict[str, int] = {}
        self.nash_weights: dict[str, float] = {}
        self.nash_value = 0.5
        self.nash_state_update = -1
        self.min_action_disagreement = min_action_disagreement
        self.diversity_probe_size = diversity_probe_size
        self.duel_opponent_id = next(
            (
                opponent_id
                for label, opponent_id, _ in LOG_DERIVED_HEURISTICS
                if label == duel_opponent_label
            ),
            None,
        )
        self.duel_opponent_probability = duel_opponent_probability
        self.duel_opponent_weights = resolved_duel_opponent_weights
        self.inference_amp_dtype = inference_amp_dtype
        self.active_checkpoint_limit = active_checkpoint_limit
        self.active_rotation_interval = active_rotation_interval
        self.active_rotation_cursor = 0
        self.last_active_rotation_update: int | None = None
        self.max_snapshot_update = max_snapshot_update
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self.cpu_rng = random.Random(seed)
        self.models: dict[int, FrozenActor] = {}
        self.paths: dict[int, Path] = {}
        self.checkpoint_steps: dict[int, int] = {}
        self.checkpoint_labels: dict[int, str] = {}
        self.selectable_ids: list[int] = []
        self.active_ids: list[int] = []
        self.pinned_ids: set[int] = set()
        self.retired_ids: set[int] = set()
        self.last_evaluation_ids: list[int] = []
        self.last_evaluation_keys: list[str] = []
        self.evaluation_cursor = 0
        self.next_id = 0
        self.last_snapshot_update = -1
        self.last_snapshot_attempt_update = -1
        self.checkpoints_enabled = False
        self._nash_sample_cache_key: tuple | None = None
        self._nash_sample_ids: torch.Tensor | None = None
        self._nash_sample_weights: torch.Tensor | None = None
        self.inference_streams = (
            [
                torch.cuda.Stream(device=device)
                for _ in range(active_checkpoint_limit)
            ]
            if self.parallel_inference
            else []
        )
        self.assignment_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self.assignment_ready = (
            torch.cuda.Event() if device.type == "cuda" else None
        )
        self.assignments_cpu = (
            torch.empty(
                (num_envs, 3),
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            )
            if device.type == "cuda"
            else None
        )

        self.assignments = torch.empty(
            (num_envs, 3), dtype=torch.long, device=device
        )
        self.deterministic_assignments = torch.zeros(
            (num_envs, 3), dtype=torch.bool, device=device
        )
        self.duel_rows = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )
        self.copy_rows = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )
        self.best_selection = torch.zeros(
            (num_envs, 4), dtype=torch.bool, device=device
        )
        self.native_profile_ids = torch.full(
            (num_envs, 4), -1, dtype=torch.int32, device=device
        )
        self.actor_h = torch.zeros(
            (num_envs, 3, hidden_size), device=device
        )
        self.actor_c = torch.zeros_like(self.actor_h)

        if load_existing:
            self._load_existing()
            self._load_nash_state()
        self.resample(torch.ones(num_envs, dtype=torch.bool, device=device))

    @property
    def checkpoint_count(self) -> int:
        return len(self.selectable_ids)

    def _register(
        self,
        actor: FrozenActor,
        path: Path | None,
        update: int,
        total_steps: int = 0,
        *,
        pinned: bool = False,
        label: str | None = None,
        track_snapshot_update: bool = True,
    ) -> int:
        checkpoint_id = self.next_id
        self.next_id += 1
        self.models[checkpoint_id] = actor.to(self.device)
        if path is not None:
            self.paths[checkpoint_id] = path
        self.checkpoint_steps[checkpoint_id] = total_steps
        if label is not None:
            self.checkpoint_labels[checkpoint_id] = label
        self.selectable_ids.append(checkpoint_id)
        if pinned:
            self.pinned_ids.add(checkpoint_id)
        if track_snapshot_update:
            self.last_snapshot_update = max(self.last_snapshot_update, update)
            self.last_snapshot_attempt_update = max(
                self.last_snapshot_attempt_update, update
            )
        return checkpoint_id

    def pinned_id_for_label(self, label: str) -> int | None:
        return next(
            (
                checkpoint_id
                for checkpoint_id in self.pinned_ids
                if self.checkpoint_labels.get(checkpoint_id) == label
            ),
            None,
        )

    def training_selectable_ids(self) -> list[int]:
        """Exclude a fixed retention copy while it equals the league champion.

        Keeping the original champion under a permanent label is necessary
        after later promotions, but before the first promotion that label and
        ``league_champion`` can refer to the exact same checkpoint. Sampling
        both gives one policy two Nash columns and two independent floors.
        Matching non-zero provenance steps let training collapse that duplicate
        without removing either policy from evaluation or future retention.
        """

        ids = list(self.selectable_ids)
        if not self.deduplicate_champion:
            return ids
        champion_id = self.pinned_id_for_label(LEAGUE_CHAMPION_LABEL)
        if champion_id is None:
            return ids
        champion_steps = self.checkpoint_steps.get(champion_id, 0)
        if champion_steps <= 0:
            return ids
        return [
            checkpoint_id
            for checkpoint_id in ids
            if checkpoint_id == champion_id
            or checkpoint_id not in self.pinned_ids
            or self.checkpoint_steps.get(checkpoint_id, 0) != champion_steps
        ]

    def duel_opponent_id_for_label(self, label: str) -> int | None:
        """Resolve a forced-duel label to a native profile or pinned policy."""

        native_id = next(
            (
                opponent_id
                for native_label, opponent_id, _ in LOG_DERIVED_HEURISTICS
                if native_label == label
            ),
            None,
        )
        return native_id if native_id is not None else self.pinned_id_for_label(label)

    def forced_neural_duel_ids(self) -> set[int]:
        """Return fixed policies already receiving dedicated duel exposure."""

        result = set()
        for label, probability in self.duel_opponent_weights.items():
            if probability <= 0.0:
                continue
            opponent_id = self.duel_opponent_id_for_label(label)
            if opponent_id is not None and opponent_id >= 0:
                result.add(opponent_id)
        return result

    def focus_checkpoint_id(self) -> int | None:
        if not self.focus_label:
            return None
        return self.pinned_id_for_label(self.focus_label)

    def opponent_key(self, opponent_id: int) -> str:
        scripted_label = SCRIPTED_OPPONENT_LABELS.get(opponent_id)
        if scripted_label is not None:
            return f"heuristic:{scripted_label}"
        label = self.checkpoint_labels.get(opponent_id)
        if label is not None:
            return f"anchor:{label}"
        path = self.paths.get(opponent_id)
        if path is not None:
            return f"snapshot:{path.stem}"
        return f"checkpoint:{opponent_id}"

    def _active_nash_keys(self) -> set[str]:
        keys = {
            self.opponent_key(checkpoint_id)
            for checkpoint_id in self.training_selectable_ids()
        }
        keys.update(
            self.opponent_key(opponent_id)
            for _, opponent_id in self.scripted_opponents
        )
        return keys

    @property
    def nash_state_path(self) -> Path:
        return self.pool_dir / "nash_state.json"

    def _save_nash_state(self, update: int | None = None) -> None:
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        if update is not None:
            self.nash_state_update = int(update)
        path = self.nash_state_path
        tmp_path = path.with_suffix(".tmp")
        payload = {
            "version": 1,
            "update": self.nash_state_update,
            "rows": self.nash_rows,
            "latest_scores": self.nash_latest_scores,
            "score_updates": self.nash_score_updates,
            "weights": self.nash_weights,
            "value": self.nash_value,
            "last_evaluation_keys": self.last_evaluation_keys,
            "evaluation_cursor": self.evaluation_cursor,
            "active_rotation_cursor": self.active_rotation_cursor,
            "last_active_rotation_update": self.last_active_rotation_update,
            "anchor_scores": self.anchor_scores,
        }
        serialized = json.dumps(payload, sort_keys=True)
        tmp_path.write_text(serialized, encoding="utf-8")
        os.replace(tmp_path, path)
        if self.nash_state_update >= 0:
            history_dir = self.pool_dir / "nash_state_history"
            history_dir.mkdir(parents=True, exist_ok=True)
            history_path = history_dir / (
                f"nash_state_u{self.nash_state_update:08d}.json"
            )
            history_tmp = history_path.with_suffix(".tmp")
            history_tmp.write_text(serialized, encoding="utf-8")
            os.replace(history_tmp, history_path)
            history_paths = sorted(history_dir.glob("nash_state_u*.json"))
            for stale_path in history_paths[:-32]:
                stale_path.unlink(missing_ok=True)

    def _load_nash_state(self) -> None:
        path = self.nash_state_path
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            source_path = path
            if payload.get("version") != 1:
                raise ValueError("unsupported state version")
            state_update = int(payload.get("update", -1))
            if (
                self.max_snapshot_update is not None
                and state_update > self.max_snapshot_update
            ):
                future_update = state_update
                history_dir = self.pool_dir / "nash_state_history"
                eligible = []
                for candidate in history_dir.glob("nash_state_u*.json"):
                    match = re.search(r"_u(\d+)\.json$", candidate.name)
                    candidate_update = int(match.group(1)) if match else -1
                    if candidate_update <= self.max_snapshot_update:
                        eligible.append((candidate_update, candidate))
                if not eligible:
                    raise ValueError(
                        f"Nash state update {state_update} is newer than resume "
                        f"update {self.max_snapshot_update}, with no rollback state"
                    )
                selected_update, selected_path = max(eligible)
                payload = json.loads(selected_path.read_text(encoding="utf-8"))
                source_path = selected_path
                state_update = int(payload.get("update", selected_update))
                print(
                    f"[nash] rolled state back from update {future_update} "
                    f"to {state_update}: {selected_path}",
                    flush=True,
                )
            self.nash_rows = [
                {str(key): float(value) for key, value in row.items()}
                for row in payload.get("rows", [])[-self.nash_history_size :]
            ]
            self.nash_latest_scores = {
                str(key): float(value)
                for key, value in payload.get("latest_scores", {}).items()
            }
            self.nash_score_updates = {
                str(key): int(value)
                for key, value in payload.get("score_updates", {}).items()
            }
            for key in self.nash_latest_scores:
                self.nash_score_updates.setdefault(key, state_update)
            self.nash_weights = {
                str(key): float(value)
                for key, value in payload.get("weights", {}).items()
            }
            self.nash_value = float(payload.get("value", 0.5))
            self.nash_state_update = state_update
            self.last_evaluation_keys = [
                str(key) for key in payload.get("last_evaluation_keys", [])
            ]
            self.evaluation_cursor = int(payload.get("evaluation_cursor", 0))
            self.active_rotation_cursor = int(
                payload.get("active_rotation_cursor", 0)
            )
            raw_rotation_update = payload.get("last_active_rotation_update")
            self.last_active_rotation_update = (
                int(raw_rotation_update)
                if raw_rotation_update is not None
                else None
            )
            self.anchor_scores = {
                str(key): float(value)
                for key, value in payload.get("anchor_scores", {}).items()
                if str(key) in self.anchor_targets
            }
            print(
                f"[nash] restored {len(self.nash_rows)} payoff rows from "
                f"{source_path}",
                flush=True,
            )
        except Exception as exc:
            print(f"[nash] ignored invalid state {path}: {exc}", flush=True)

    def update_nash_distribution(
        self,
        evaluation_results: list[dict[str, float | str]],
        *,
        update: int | None = None,
    ) -> dict[str, float]:
        """Update the empirical win-rate matrix and solve its opponent Nash mix.

        Each evaluation point contributes a learner row. Unmeasured opponents
        carry a time-decayed latest score into subsequent rows, producing a
        dense, bounded-history matrix without an O(population²) tournament.
        """

        observed = 0
        observed_update = int(
            update if update is not None else max(0, self.nash_state_update)
        )
        for result in evaluation_results:
            key = str(result.get("opponent_key", ""))
            if not key:
                continue
            score = float(result["current_win_rate"]) + 0.5 * float(
                result["draw_rate"]
            )
            self.nash_latest_scores[key] = min(1.0, max(0.0, score))
            self.nash_score_updates[key] = observed_update
            observed += 1
        if not observed:
            return {}

        active_keys = self._active_nash_keys()
        row = {}
        for key, score in self.nash_latest_scores.items():
            if key not in active_keys:
                continue
            if self.nash_score_half_life_updates:
                last_observed = self.nash_score_updates.get(key, observed_update)
                age = max(0, observed_update - last_observed)
                retention = 2.0 ** (
                    -age / float(self.nash_score_half_life_updates)
                )
                score = 0.5 + (score - 0.5) * retention
            row[key] = score
        self.nash_rows.append(row)
        self.nash_rows = self.nash_rows[-self.nash_history_size :]
        columns = sorted(
            key
            for key in active_keys
            if any(key in historical_row for historical_row in self.nash_rows)
        )
        if not columns:
            return {}
        matrix = torch.tensor(
            [
                [historical_row.get(key, 0.5) for key in columns]
                for historical_row in self.nash_rows
            ],
            dtype=torch.float64,
        )
        _, opponent_mix, self.nash_value = solve_zero_sum_nash_distribution(
            matrix,
            iterations=self.nash_iterations,
        )
        if self.nash_exploration:
            opponent_mix = (
                (1.0 - self.nash_exploration) * opponent_mix
                + self.nash_exploration / len(columns)
            )
        opponent_mix /= opponent_mix.sum()
        self.nash_weights = {
            key: float(weight)
            for key, weight in zip(columns, opponent_mix.tolist())
        }
        self._save_nash_state(update)
        return dict(self.nash_weights)

    def nash_sampling_weight(self, opponent_id: int) -> float:
        key = self.opponent_key(opponent_id)
        if key in self.nash_weights:
            return self.nash_weights[key]
        active_count = max(1, len(self.active_ids) + len(self.scripted_opponents))
        return self.nash_exploration / active_count

    def nash_training_distribution(
        self,
    ) -> tuple[list[int], torch.Tensor]:
        """Return the constrained Nash mixture actually used for training.

        The unconstrained equilibrium can give a scripted opponent almost no
        mass when some historical checkpoint is marginally harder. The
        configured heuristic category weights therefore act as guaranteed
        floors, while the configured checkpoint share remains free Nash mass
        that may reinforce either neural or heuristic opponents.
        """

        sampling_ids = self.active_ids or self.training_selectable_ids()
        opponents = [opponent_id for _, opponent_id in self.scripted_opponents]
        if self.checkpoints_enabled:
            opponents = [*sampling_ids, *opponents]
        raw_nash = [
            self.nash_sampling_weight(opponent_id) for opponent_id in opponents
        ]
        raw_nash = [
            weight * self.anchor_sampling_weight(opponent_id)
            for opponent_id, weight in zip(opponents, raw_nash)
        ]
        raw_total = sum(raw_nash)
        if raw_total <= 0.0:
            raw_nash = [1.0 / len(opponents)] * len(opponents)
        else:
            raw_nash = [weight / raw_total for weight in raw_nash]

        checkpoint_weight = self.category_weights[POOL_CHECKPOINT_CATEGORY]
        total_weight = sum(self.category_weights.values())
        checkpoint_share = checkpoint_weight / total_weight
        champion_id = self.pinned_id_for_label(LEAGUE_CHAMPION_LABEL)
        forced_duel_ids = self.forced_neural_duel_ids()
        champion_floor = (
            self.champion_min_weight
            if champion_id is not None and champion_id in opponents
            else 0.0
        )
        uniform_anchor_ids = [
            opponent_id
            for opponent_id in opponents
            if opponent_id in self.pinned_ids
            and opponent_id != champion_id
            and opponent_id not in forced_duel_ids
        ]
        anchor_floor = (
            self.anchor_uniform_floor if uniform_anchor_ids else 0.0
        )
        per_anchor_floor = (
            anchor_floor / len(uniform_anchor_ids)
            if uniform_anchor_ids
            else 0.0
        )
        heuristic_floors = {
            opponent_id: self.category_weights[opponent_id] / total_weight
            for _, opponent_id in self.scripted_opponents
        }
        constrained = []
        for opponent_id, nash_weight in zip(opponents, raw_nash):
            weight = nash_weight * (
                checkpoint_share - champion_floor - anchor_floor
            )
            weight += heuristic_floors.get(opponent_id, 0.0)
            if opponent_id == champion_id:
                weight += champion_floor
            elif opponent_id in uniform_anchor_ids:
                weight += per_anchor_floor
            constrained.append(weight)
        constrained_total = sum(constrained)
        weights = torch.tensor(
            [weight / constrained_total for weight in constrained],
            dtype=torch.float32,
            device=self.device,
        )
        return opponents, weights

    def effective_training_weights(self) -> dict[str, float]:
        """Expose the normalized constrained mixture for diagnostics."""

        if not self.nash_weights:
            return {}
        opponents, weights = self.nash_training_distribution()
        return {
            self.opponent_key(opponent_id): float(weight)
            for opponent_id, weight in zip(opponents, weights.tolist())
        }

    def anchor_sampling_weight(self, checkpoint_id: int) -> float:
        """Return PFSP priority for one anchor, or one before it has a score."""

        label = self.checkpoint_labels.get(checkpoint_id)
        if label is None or label not in self.anchor_targets:
            return 1.0
        score = self.anchor_scores.get(label)
        if score is None:
            return 1.0
        target = self.anchor_targets[label]
        if target <= 0.0:
            return self.anchor_min_weight
        relative_deficit = max(0.0, target - score) / target
        return self.anchor_min_weight + relative_deficit**self.anchor_priority_exponent

    def update_anchor_priorities(
        self,
        evaluation_results: list[dict[str, float | str]],
    ) -> dict[str, tuple[float, float]]:
        """Update smoothed anchor scores and return ``label: (score, weight)``."""

        updated: dict[str, tuple[float, float]] = {}
        for result in evaluation_results:
            if result.get("kind") != "anchor":
                continue
            label = str(result["label"])
            if label not in self.anchor_targets:
                continue
            observed_score = float(result["current_win_rate"]) + 0.5 * float(
                result["draw_rate"]
            )
            previous = self.anchor_scores.get(label)
            score = (
                observed_score
                if previous is None
                else (
                    self.anchor_score_ema * observed_score
                    + (1.0 - self.anchor_score_ema) * previous
                )
            )
            self.anchor_scores[label] = score
            checkpoint_id = self.pinned_id_for_label(label)
            if checkpoint_id is not None:
                updated[label] = (
                    score,
                    self.anchor_sampling_weight(checkpoint_id),
                )
        return updated

    def _load_existing(self):
        anchors = sorted(
            self.pool_dir.glob("anchor_*.pt"), key=lambda path: path.name
        )
        snapshot_slots = max(0, self.max_checkpoints - len(anchors))
        all_snapshots = sorted(
            self.pool_dir.glob("snap_*.pt"),
            key=lambda path: (*checkpoint_counters_from_name(path), path.name),
        )
        snapshots = []
        for path in all_snapshots:
            match = re.search(r"_u(\d+)\.pt$", path.name)
            snapshot_update = int(match.group(1)) if match else None
            if (
                self.max_snapshot_update is not None
                and snapshot_update is not None
                and snapshot_update > self.max_snapshot_update
            ):
                quarantine_dir = self.pool_dir / (
                    f"discarded_after_u{self.max_snapshot_update:08d}"
                )
                quarantine_dir.mkdir(parents=True, exist_ok=True)
                quarantine_path = quarantine_dir / path.name
                if quarantine_path.exists():
                    quarantine_path = quarantine_dir / (
                        f"{path.stem}_{time.time_ns()}{path.suffix}"
                    )
                path.rename(quarantine_path)
                print(
                    f"[selfplay] quarantined future snapshot {path.name} while "
                    f"resuming update {self.max_snapshot_update}: "
                    f"{quarantine_path}",
                    flush=True,
                )
                continue
            snapshots.append(path)
        stale_paths = snapshots[:-snapshot_slots] if snapshot_slots else snapshots
        for stale_path in stale_paths:
            stale_path.unlink(missing_ok=True)
        paths = [(path, True) for path in anchors]
        if snapshot_slots:
            paths.extend((path, False) for path in snapshots[-snapshot_slots:])
        loaded_anchor_labels: set[str] = set()
        for path, pinned in paths:
            try:
                payload = torch.load(path, map_location=self.device, weights_only=False)
                if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                    raise ValueError("incompatible schema")
                label = payload.get("label")
                if pinned and label is not None and label in loaded_anchor_labels:
                    print(
                        f"[selfplay] skipped duplicate pinned anchor "
                        f"label={label}: {path}",
                        flush=True,
                    )
                    continue
                model = self.model_factory()
                load_model_state(model, checkpoint_model_state(payload), allow_legacy=False)
                actor = FrozenActor(model)
                self._register(
                    actor,
                    path,
                    int(payload.get("update", -1)),
                    int(payload.get("total_steps", 0)),
                    pinned=pinned or bool(payload.get("pinned", False)),
                    label=label,
                    track_snapshot_update=not (
                        pinned or bool(payload.get("pinned", False))
                    ),
                )
                if pinned and label is not None:
                    loaded_anchor_labels.add(str(label))
                del model
            except Exception as exc:
                print(f"[selfplay] skipped incompatible pool checkpoint {path}: {exc}")

    @torch.inference_mode()
    def _behavioral_signature(
        self,
        actor: FrozenActor,
        probe_obs: torch.Tensor,
    ) -> torch.Tensor:
        flat_obs = probe_obs.reshape(-1, *OBS_SHAPE)
        sample_count = min(self.diversity_probe_size, flat_obs.shape[0])
        if sample_count <= 0:
            raise ValueError("diversity probe contains no observations")
        if sample_count < flat_obs.shape[0]:
            indices = torch.linspace(
                0,
                flat_obs.shape[0] - 1,
                sample_count,
                device=flat_obs.device,
            ).long()
            flat_obs = flat_obs.index_select(0, indices)
        flat_obs = flat_obs.to(self.device)
        use_amp = self.device.type == "cuda" and self.inference_amp_dtype is not None
        if not use_amp:
            flat_obs = flat_obs.float()
        h, c = actor.initial_actor_state(flat_obs.shape[0], self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.inference_amp_dtype if use_amp else None,
            enabled=use_amp,
        ):
            logits, _, _ = actor.actor_forward(flat_obs, h, c)
        legal = observable_action_mask(flat_obs)
        return masked_categorical(logits.float(), legal).logits.argmax(dim=-1)

    @torch.inference_mode()
    def minimum_action_disagreement(
        self,
        candidate: FrozenActor,
        probe_obs: torch.Tensor,
    ) -> tuple[float, int | None]:
        """Return the candidate's nearest behavioral neighbor in the pool."""

        if not self.selectable_ids:
            return 1.0, None
        candidate_signature = self._behavioral_signature(candidate, probe_obs)
        minimum = 1.0
        closest_id = None
        for checkpoint_id in self.selectable_ids:
            reference_signature = self._behavioral_signature(
                self.models[checkpoint_id], probe_obs
            )
            disagreement = float(
                (candidate_signature != reference_signature).float().mean().item()
            )
            if disagreement < minimum:
                minimum = disagreement
                closest_id = checkpoint_id
        return minimum, closest_id

    def add_checkpoint(
        self,
        model: ActorCritic,
        update: int,
        total_steps: int,
        *,
        probe_obs: torch.Tensor | None = None,
    ) -> Path | None:
        """Persist the policy only when it adds behavioral population diversity."""

        self.last_snapshot_attempt_update = max(
            self.last_snapshot_attempt_update, update
        )
        actor = FrozenActor(model).to(self.device)
        diversity = 1.0
        closest_id = None
        if self.min_action_disagreement > 0.0 and probe_obs is not None:
            diversity, closest_id = self.minimum_action_disagreement(
                actor, probe_obs
            )
            if diversity < self.min_action_disagreement:
                closest = (
                    self.opponent_key(closest_id)
                    if closest_id is not None
                    else "unknown"
                )
                print(
                    f"[selfplay] skipped checkpoint update={update}: "
                    f"action_disagreement={diversity:.3f} < "
                    f"{self.min_action_disagreement:.3f} nearest={closest}",
                    flush=True,
                )
                return None

        self.pool_dir.mkdir(parents=True, exist_ok=True)
        path = self.pool_dir / f"snap_{total_steps:012d}_u{update:08d}.pt"
        tmp_path = path.with_suffix(".tmp")
        cpu_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        torch.save(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "activation": "tanh",
                "separate_critic_lstm": True,
                "update": update,
                "total_steps": total_steps,
                "model_state_dict": cpu_state,
            },
            tmp_path,
        )
        os.replace(tmp_path, path)
        checkpoint_id = self._register(actor, path, update, total_steps)

        self._trim_to_capacity()
        self._collect_retired()
        print(
            f"[selfplay] froze checkpoint id={checkpoint_id} update={update} "
            f"diversity={diversity:.3f} "
            f"pool={self.checkpoint_count}/{self.max_checkpoints}: {path}"
        )
        return path

    def _trim_to_capacity(self):
        while len(self.selectable_ids) > self.max_checkpoints:
            retired = next(
                (
                    checkpoint_id
                    for checkpoint_id in self.selectable_ids
                    if checkpoint_id not in self.pinned_ids
                ),
                None,
            )
            if retired is None:
                raise RuntimeError("opponent pool contains only pinned checkpoints")
            self.selectable_ids.remove(retired)
            self.retired_ids.add(retired)
            self.checkpoint_steps.pop(retired, None)
            self.checkpoint_labels.pop(retired, None)
            retired_path = self.paths.pop(retired, None)
            if retired_path is not None:
                retired_path.unlink(missing_ok=True)

    def add_anchor(
        self,
        model: ActorCritic,
        update: int,
        total_steps: int,
        *,
        label: str = "continuation",
    ) -> Path:
        """Persist and pin a continuation source so it cannot be forgotten."""

        existing_id = self.pinned_id_for_label(label)
        if existing_id is not None:
            return self.paths[existing_id]

        self.pool_dir.mkdir(parents=True, exist_ok=True)
        path = self.pool_dir / (
            f"anchor_{label}_{total_steps:012d}_u{update:08d}_{time.time_ns()}.pt"
        )
        tmp_path = path.with_suffix(".tmp")
        torch.save(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "activation": "tanh",
                "separate_critic_lstm": True,
                "pinned": True,
                "label": label,
                "update": update,
                "total_steps": total_steps,
                "model_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            },
            tmp_path,
        )
        os.replace(tmp_path, path)
        self._register(
            FrozenActor(model),
            path,
            update,
            total_steps,
            pinned=True,
            label=label,
            track_snapshot_update=False,
        )
        self._trim_to_capacity()
        self._collect_retired()
        print(f"[selfplay] pinned anchor label={label}: {path}")
        return path

    def add_external_anchor(self, label: str, source: str | Path) -> Path:
        """Import and pin an external Torch/SB3 policy under a stable label."""

        if not ANCHOR_LABEL_RE.fullmatch(label):
            raise ValueError(f"invalid anchor label: {label!r}")
        existing_id = self.pinned_id_for_label(label)
        if existing_id is not None:
            print(
                f"[selfplay] reusing pinned anchor label={label}: "
                f"{self.paths[existing_id]}"
            )
            return self.paths[existing_id]

        source_path = Path(source).expanduser()
        if source_path.suffix.lower() == ".zip":
            imported = load_sb3_recurrent_ppo(str(source_path), device=self.device)
            model = self.model_factory()
            model.load_state_dict(imported.state_dict(), strict=True)
            del imported
            total_steps, update = checkpoint_counters_from_name(source_path)
        else:
            payload = torch.load(
                source_path, map_location=self.device, weights_only=False
            )
            model = self.model_factory()
            load_model_state(
                model,
                checkpoint_model_state(payload),
                allow_legacy=False,
            )
            total_steps = int(payload.get("total_steps", 0))
            update = int(payload.get("update", 0))
            del payload

        path = self.add_anchor(
            model,
            update=update,
            total_steps=total_steps,
            label=label,
        )
        del model
        print(
            f"[selfplay] imported external anchor label={label} "
            f"source={source_path}"
        )
        return path

    def replace_anchor(
        self,
        label: str,
        model: ActorCritic,
        update: int,
        total_steps: int,
    ) -> Path:
        """Replace one pinned policy without disrupting active episodes."""

        # A resumed run can contain more than one persisted copy of a labeled
        # anchor when an earlier promotion was interrupted between updating
        # the stable champion and pruning its pool entry.  Retire every copy so
        # the label remains unique after replacement.
        existing_ids = [
            checkpoint_id
            for checkpoint_id in self.selectable_ids
            if checkpoint_id in self.pinned_ids
            and self.checkpoint_labels.get(checkpoint_id) == label
        ]
        for existing_id in existing_ids:
            self.pinned_ids.remove(existing_id)
            self.selectable_ids.remove(existing_id)
            self.retired_ids.add(existing_id)
            self.checkpoint_steps.pop(existing_id, None)
            self.checkpoint_labels.pop(existing_id, None)
            old_path = self.paths.pop(existing_id, None)
            if old_path is not None:
                old_path.unlink(missing_ok=True)

        # active_ids is also the source for new episode assignments.  Leaving
        # a retired id here can sample it after _collect_retired has released
        # the model, producing a KeyError on the next rollout.  Existing
        # assignments remain valid because their retired models are retained
        # until those episodes finish.
        if existing_ids:
            retired = set(existing_ids)
            self.active_ids = [
                checkpoint_id
                for checkpoint_id in self.active_ids
                if checkpoint_id not in retired
            ]
            self.last_active_rotation_update = None
            self._nash_sample_cache_key = None
            self._nash_sample_ids = None
            self._nash_sample_weights = None
        path = self.add_anchor(
            model,
            update=update,
            total_steps=total_steps,
            label=label,
        )
        if self.checkpoints_enabled:
            self.begin_rollout(update)
        self._collect_retired()
        return path

    def enable_checkpoint_sampling(self):
        self.checkpoints_enabled = True

    def begin_rollout(self, update: int | None = None):
        """Select a bounded, episode-stable neural-opponent working set.

        Fixed anchors are rotated through the working set instead of loading
        all of them into every simulator step.  The league champion, forced
        neural duel opponents, and newest self-play snapshot remain active.
        Holding a selection for several updates also lets episodes using the
        previous selection drain before the next rotation.
        """

        ids = self.training_selectable_ids()
        if not ids:
            self.active_ids = []
            return

        required: list[int] = []

        def require(checkpoint_id: int | None) -> None:
            if (
                checkpoint_id is not None
                and checkpoint_id >= 0
                and checkpoint_id in ids
                and checkpoint_id not in required
            ):
                required.append(checkpoint_id)

        require(self.focus_checkpoint_id())
        if self.champion_min_weight > 0.0:
            require(self.pinned_id_for_label(LEAGUE_CHAMPION_LABEL))
        for checkpoint_id in self.forced_neural_duel_ids():
            require(checkpoint_id)
        latest_history_id = next(
            (
                checkpoint_id
                for checkpoint_id in reversed(ids)
                if checkpoint_id not in self.pinned_ids
            ),
            None,
        )
        require(latest_history_id)

        if len(required) > self.active_checkpoint_limit:
            labels = ", ".join(
                self.checkpoint_labels.get(
                    checkpoint_id,
                    self.opponent_key(checkpoint_id),
                )
                for checkpoint_id in required
            )
            raise RuntimeError(
                "pool-active-checkpoints is smaller than the mandatory neural "
                f"working set ({len(required)}): {labels}"
            )

        active_is_valid = (
            bool(self.active_ids)
            and all(checkpoint_id in ids for checkpoint_id in self.active_ids)
            and all(checkpoint_id in self.active_ids for checkpoint_id in required)
        )
        if (
            update is not None
            and self.last_active_rotation_update is not None
            and update >= self.last_active_rotation_update
            and update - self.last_active_rotation_update
            < self.active_rotation_interval
            and active_is_valid
        ):
            return

        def finish(selection: list[int]) -> None:
            self.active_ids = sorted(selection)
            if update is not None:
                self.last_active_rotation_update = int(update)

        if len(ids) <= self.active_checkpoint_limit:
            finish(ids)
            return

        if self.anchor_uniform_floor > 0.0:
            rotating_anchors = sorted(
                (
                    checkpoint_id
                    for checkpoint_id in ids
                    if checkpoint_id in self.pinned_ids
                    and checkpoint_id not in required
                ),
                key=lambda checkpoint_id: (
                    self.checkpoint_labels.get(checkpoint_id, ""),
                    checkpoint_id,
                ),
            )
            rotating_slots = min(
                len(rotating_anchors),
                self.active_checkpoint_limit - len(required),
            )
            if rotating_slots:
                start = self.active_rotation_cursor % len(rotating_anchors)
                required.extend(
                    rotating_anchors[(start + offset) % len(rotating_anchors)]
                    for offset in range(rotating_slots)
                )
                self.active_rotation_cursor = (
                    start + rotating_slots
                ) % len(rotating_anchors)

            prioritized = sorted(
                (
                    checkpoint_id
                    for checkpoint_id in ids
                    if checkpoint_id not in required
                ),
                key=lambda checkpoint_id: (
                    checkpoint_id in self.last_evaluation_ids,
                    self.nash_sampling_weight(checkpoint_id),
                    checkpoint_id,
                ),
                reverse=True,
            )
            required.extend(
                prioritized[: self.active_checkpoint_limit - len(required)]
            )
            finish(required)
            return

        if not self.nash_weights and not self.last_evaluation_ids:
            pinned = [
                checkpoint_id
                for checkpoint_id in ids
                if checkpoint_id in self.pinned_ids
                and checkpoint_id not in required
            ]
            pinned_slots = self.active_checkpoint_limit - len(required)
            if pinned_slots > 0:
                required.extend(
                    pinned
                    if len(pinned) <= pinned_slots
                    else self.cpu_rng.sample(pinned, pinned_slots)
                )
            candidates = [
                checkpoint_id
                for checkpoint_id in ids
                if checkpoint_id not in required
            ]
            remaining = self.active_checkpoint_limit - len(required)
            older = self.cpu_rng.sample(candidates, remaining) if remaining else []
            finish(older + required)
            return

        evaluation_ids = [
            checkpoint_id
            for checkpoint_id in ids
            if checkpoint_id in self.last_evaluation_ids
            or self.opponent_key(checkpoint_id) in self.last_evaluation_keys
        ]
        prioritized = sorted(
            (
                checkpoint_id
                for checkpoint_id in dict.fromkeys([*evaluation_ids, *ids])
                if checkpoint_id in ids and checkpoint_id not in required
            ),
            key=lambda checkpoint_id: (
                checkpoint_id in evaluation_ids,
                self.nash_sampling_weight(checkpoint_id),
                checkpoint_id in self.pinned_ids,
                checkpoint_id,
            ),
            reverse=True,
        )
        required.extend(
            prioritized[: self.active_checkpoint_limit - len(required)]
        )
        finish(required)

    def evaluation_candidates(
        self,
        history_limit: int = 2,
    ) -> list[tuple[str, str, int, FrozenActor | None]]:
        """Return anchors, refreshed Nash support, and scripted opponents.

        The latest learner snapshot is always measured. Half of the remaining
        history budget refreshes the strongest currently supported Nash
        opponents; the rest goes to the least recently measured population
        members. This prevents a high-weight opponent from retaining a stale
        score while still exploring the full bounded population.
        """

        if history_limit < 0:
            return []
        result: list[tuple[str, str, int, FrozenActor | None]] = []
        pinned = sorted(
            (
                (
                    self.checkpoint_labels.get(
                        checkpoint_id, f"anchor_{checkpoint_id}"
                    ),
                    checkpoint_id,
                )
                for checkpoint_id in self.selectable_ids
                if checkpoint_id in self.pinned_ids
            ),
            key=lambda item: item[0],
        )
        for label, checkpoint_id in pinned:
            result.append(
                ("anchor", label, checkpoint_id, self.models[checkpoint_id])
            )
        history = [
            checkpoint_id
            for checkpoint_id in self.selectable_ids
            if checkpoint_id not in self.pinned_ids
        ]
        count = min(history_limit, len(history))
        selected_history: list[int] = []
        if count:
            selected_history.append(history[-1])
            support_slots = min(count - 1, max(1, history_limit // 2))
            support = sorted(
                (
                    checkpoint_id
                    for checkpoint_id in history
                    if checkpoint_id not in selected_history
                ),
                key=lambda checkpoint_id: (
                    self.nash_sampling_weight(checkpoint_id),
                    checkpoint_id,
                ),
                reverse=True,
            )
            selected_history.extend(support[:support_slots])
            stale = sorted(
                (
                    checkpoint_id
                    for checkpoint_id in history
                    if checkpoint_id not in selected_history
                ),
                key=lambda checkpoint_id: (
                    self.nash_score_updates.get(
                        self.opponent_key(checkpoint_id), -1
                    ),
                    checkpoint_id,
                ),
            )
            selected_history.extend(stale[: count - len(selected_history)])
        for offset, checkpoint_id in enumerate(selected_history):
            result.append(
                (
                    "history",
                    f"slot_{offset + 1}",
                    checkpoint_id,
                    self.models[checkpoint_id],
                )
            )
        self.evaluation_cursor += len(selected_history)
        self.last_evaluation_ids = [
            checkpoint_id
            for kind, _, checkpoint_id, _ in result
            if kind in ("anchor", "history")
        ]
        self.last_evaluation_keys = [
            self.opponent_key(checkpoint_id)
            for checkpoint_id in self.last_evaluation_ids
        ]
        result.extend(
            ("heuristic", label, opponent_id, None)
            for label, opponent_id in self.scripted_opponents
        )
        return result

    def _sample(self, count: int) -> torch.Tensor:
        sampling_ids = self.active_ids or self.training_selectable_ids()
        if self.nash_weights:
            cache_key = (
                self.checkpoints_enabled,
                tuple(sampling_ids),
                tuple(sorted(self.pinned_ids)),
                tuple(sorted(self.anchor_scores.items())),
                tuple(sorted(self.nash_weights.items())),
                tuple(sorted(self.forced_neural_duel_ids())),
            )
            if cache_key != self._nash_sample_cache_key:
                opponents, weights = self.nash_training_distribution()
                self._nash_sample_ids = torch.tensor(
                    opponents,
                    dtype=torch.long,
                    device=self.device,
                )
                self._nash_sample_weights = weights
                self._nash_sample_cache_key = cache_key
            if (
                self._nash_sample_ids is None
                or self._nash_sample_weights is None
            ):
                raise RuntimeError("Nash sampling cache was not initialized")
            indices = torch.multinomial(
                self._nash_sample_weights,
                count,
                replacement=True,
                generator=self.generator,
            )
            return self._nash_sample_ids[indices]

        checkpoint_weight = self.category_weights[POOL_CHECKPOINT_CATEGORY]
        if not self.checkpoints_enabled or not sampling_ids:
            checkpoint_weight = 0.0
        categories_list = [
            POOL_CHECKPOINT_CATEGORY,
            *(opponent_id for _, opponent_id in self.scripted_opponents),
        ]
        python_weights = [
            checkpoint_weight,
            *(
                self.category_weights[opponent_id]
                for _, opponent_id in self.scripted_opponents
            ),
        ]
        if sum(python_weights) <= 0.0:
            python_weights[-1] = 1.0
        category_weights = torch.tensor(
            python_weights,
            dtype=torch.float32,
            device=self.device,
        )
        categories = torch.tensor(
            categories_list,
            dtype=torch.long,
            device=self.device,
        )
        sampled_categories = categories[
            torch.multinomial(
                category_weights,
                count,
                replacement=True,
                generator=self.generator,
            )
        ]
        checkpoint_positions = sampled_categories == POOL_CHECKPOINT_CATEGORY
        if sampling_ids:
            ids = torch.tensor(sampling_ids, dtype=torch.long, device=self.device)
            random_indices = torch.randint(
                len(sampling_ids),
                (count,),
                device=self.device,
                generator=self.generator,
            )
            sampled_ids = ids[random_indices]
            choose_latest = (
                torch.rand(
                    count,
                    device=self.device,
                    generator=self.generator,
                )
                < self.latest_probability
            )
            sampled_ids = torch.where(choose_latest, ids[-1], sampled_ids)
            active_anchors = [
                checkpoint_id
                for checkpoint_id in sampling_ids
                if checkpoint_id in self.pinned_ids
                and self.checkpoint_labels.get(checkpoint_id)
                != LEAGUE_CHAMPION_LABEL
            ]
            if active_anchors and self.anchor_probability > 0.0:
                anchor_ids = torch.tensor(
                    active_anchors, dtype=torch.long, device=self.device
                )
                anchor_weights = torch.tensor(
                    [
                        self.anchor_sampling_weight(checkpoint_id)
                        for checkpoint_id in active_anchors
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                sampled_anchors = anchor_ids[
                    torch.multinomial(
                        anchor_weights,
                        count,
                        replacement=True,
                        generator=self.generator,
                    )
                ]
                choose_anchor = (
                    torch.rand(
                        count,
                        device=self.device,
                        generator=self.generator,
                    )
                    < self.anchor_probability
                )
                sampled_ids = torch.where(choose_anchor, sampled_anchors, sampled_ids)
            focus_id = self.focus_checkpoint_id()
            if (
                focus_id is not None
                and focus_id in sampling_ids
                and self.focus_probability > 0.0
            ):
                choose_focus = (
                    torch.rand(
                        count,
                        device=self.device,
                        generator=self.generator,
                    )
                    < self.focus_probability
                )
                sampled_ids = torch.where(
                    choose_focus,
                    torch.full_like(sampled_ids, focus_id),
                    sampled_ids,
                )
            sampled_categories = torch.where(
                checkpoint_positions,
                sampled_ids,
                sampled_categories,
            )
        return sampled_categories

    def resample(
        self,
        episode_done: torch.Tensor,
        duel_rows: torch.Tensor | None = None,
    ):
        """Replace opponents only for environments whose episode just ended."""

        # Keep this fixed-shape and entirely on the device. CUDA ``nonzero``
        # must reveal its dynamic result size to Python and therefore forced a
        # full stream synchronization after every simulator step. Sampling
        # candidates for all rows is cheap; ``where`` only commits the rows
        # whose episode actually ended.
        done_rows = episode_done.to(device=self.device, dtype=torch.bool).view(
            self.num_envs, 1
        )
        new_assignments = self._sample(self.num_envs * 3).reshape(
            self.num_envs, 3
        )
        if duel_rows is not None:
            duel_rows = duel_rows.to(device=self.device, dtype=torch.bool).view(-1)
            if duel_rows.numel() != self.num_envs:
                raise ValueError("duel_rows must contain one flag per environment")
        # Nash/PFSP evaluates one named policy at a time.  Drawing all three
        # seats independently turns that one-policy payoff into an unrelated
        # product distribution of artificial mixed lineups.  Copy rows keep
        # the sampled policy episode-stable in every opponent seat, matching
        # the homogeneous lineups used by evaluation and normal deployment.
        new_copy_rows = (
            torch.rand(
                self.num_envs,
                device=self.device,
                generator=self.generator,
            ) < self.copies_probability
        )
        if duel_rows is not None:
            new_copy_rows &= ~duel_rows
        new_assignments = torch.where(
            new_copy_rows.view(-1, 1),
            new_assignments[:, :1].expand(-1, 3),
            new_assignments,
        )
        if self.duel_opponent_weights and duel_rows is not None:
            forceable = duel_rows & episode_done.to(
                device=self.device, dtype=torch.bool
            ).view(-1)
            draw = torch.rand(
                self.num_envs,
                device=self.device,
                generator=self.generator,
            )
            lower = 0.0
            forced_ids = new_assignments[:, 0]
            for label, probability in self.duel_opponent_weights.items():
                if probability <= 0.0:
                    continue
                opponent_id = self.duel_opponent_id_for_label(label)
                if opponent_id is None:
                    raise RuntimeError(
                        f"forced duel opponent {label!r} is not loaded"
                    )
                upper = lower + probability
                choose = forceable & (draw >= lower) & (draw < upper)
                forced_ids = torch.where(
                    choose,
                    torch.full_like(forced_ids, opponent_id),
                    forced_ids,
                )
                lower = upper
            new_assignments[:, 0] = forced_ids
        elif self.duel_opponent_id is not None and duel_rows is not None:
            force_duelist = duel_rows & episode_done.to(
                device=self.device, dtype=torch.bool
            ).view(-1)
            if self.duel_opponent_probability < 1.0:
                force_duelist &= (
                    torch.rand(
                        self.num_envs,
                        device=self.device,
                        generator=self.generator,
                    ) < self.duel_opponent_probability
                )
            new_assignments[:, 0] = torch.where(
                force_duelist,
                torch.full_like(
                    new_assignments[:, 0], self.duel_opponent_id
                ),
                new_assignments[:, 0],
            )
        if duel_rows is not None:
            self.duel_rows.copy_(
                torch.where(
                    episode_done.to(device=self.device, dtype=torch.bool).view(-1),
                    duel_rows,
                    self.duel_rows,
                )
            )
        self.assignments.copy_(
            torch.where(done_rows, new_assignments, self.assignments)
        )
        self.copy_rows.copy_(
            torch.where(done_rows.view(-1), new_copy_rows, self.copy_rows)
        )
        deterministic = (
            torch.rand(
                new_assignments.shape,
                device=self.device,
                generator=self.generator,
            )
            < self.deterministic_probability
        ) & (new_assignments >= 0)
        deterministic = torch.where(
            new_copy_rows.view(-1, 1),
            deterministic[:, :1].expand(-1, 3),
            deterministic,
        )
        focus_id = self.focus_checkpoint_id()
        if focus_id is not None:
            # The focused opponent is the exact greedy policy used by the
            # champion gate, rather than merely another stochastic sample from
            # the same weights.
            deterministic |= new_assignments == focus_id
        self.deterministic_assignments.copy_(
            torch.where(
                done_rows,
                deterministic,
                self.deterministic_assignments,
            )
        )
        keep_state = (~done_rows).view(self.num_envs, 1, 1)
        self.actor_h.mul_(keep_state)
        self.actor_c.mul_(keep_state)
        if self.assignment_ready is not None:
            self.assignment_ready.record(torch.cuda.current_stream(self.device))

    def _collect_retired(self):
        for checkpoint_id in list(self.retired_ids):
            if not bool((self.assignments == checkpoint_id).any().item()):
                self.models.pop(checkpoint_id, None)
                self.retired_ids.remove(checkpoint_id)

    @torch.inference_mode()
    def actions(
        self,
        env: hisss.CudaBlackoutTorchVecEnv,
        obs: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return model-order actions for the three opponent slots."""

        flat_obs = obs.reshape(self.num_envs * 3, *OBS_SHAPE)
        flat_legal = legal_mask.reshape(self.num_envs * 3, 4)
        flat_kind = self.assignments.reshape(-1)
        flat_deterministic = self.deterministic_assignments.reshape(-1)
        flat_h = self.actor_h.reshape(-1, self.hidden_size)
        flat_c = self.actor_c.reshape(-1, self.hidden_size)
        live_slots = legal_mask.any(dim=-1)
        flat_live_slots = live_slots.reshape(-1)
        actions = random_legal_actions(flat_legal)

        if self.weights[2] > 0.0:
            hungry = hungry_legal_actions(flat_obs, flat_legal)
            actions = torch.where(flat_kind == POOL_HUNGRY, hungry, actions)
        if self.weights[1] > 0.0:
            self.best_selection.zero_()
            self.best_selection[:, 1:4] = (
                (self.assignments == POOL_BEST) & live_slots
            )
            best_cuda = env.best_actions(self.best_selection)[:, 1:4].reshape(
                -1
            )
            best_model = actions_cuda_to_model(best_cuda)
            actions = torch.where(flat_kind == POOL_BEST, best_model, actions)
        if self.native_heuristics:
            assignment_profiles = self.native_profile_lut[
                (-self.assignments).clamp(
                    min=0,
                    max=self.native_profile_lut.numel() - 1,
                )
            ]
            self.native_profile_ids.fill_(-1)
            self.native_profile_ids[:, 1:4] = torch.where(
                live_slots,
                assignment_profiles,
                -1,
            )
            heuristic_cuda = env.heuristic_actions(
                self.native_profile_ids
            )[:, 1:4].reshape(-1)
            heuristic_model = actions_cuda_to_model(heuristic_cuda)
            is_log_heuristic = assignment_profiles.reshape(-1) >= 0
            actions = torch.where(is_log_heuristic, heuristic_model, actions)

        # Copy the tiny assignment vector once instead of running CUDA
        # ``unique`` plus one dynamically-shaped ``nonzero`` per model. Those
        # operations each synchronize the host and turn a pool of small frozen
        # policies into a serial launch/synchronization workload.
        checkpoint_groups: list[tuple[int, torch.Tensor]] = []
        if self.checkpoints_enabled and self.models:
            # Dead snakes have an all-false legal mask and cannot act again
            # before the whole environment resets.  True duels start with two
            # such opponent slots, so excluding them here avoids hundreds of
            # pointless recurrent-policy rows per simulator step.
            if self.assignment_stream is not None:
                # Assignment updates happened before the learner forward. A
                # dedicated stream can therefore copy this tiny vector while
                # the main stream is still evaluating the learner policy.
                with torch.cuda.stream(self.assignment_stream):
                    self.assignment_stream.wait_event(self.assignment_ready)
                    live_kind = torch.where(
                        flat_legal.any(dim=-1),
                        flat_kind,
                        torch.full_like(flat_kind, POOL_RANDOM),
                    )
                    self.assignments_cpu.copy_(
                        live_kind.reshape(self.num_envs, 3),
                        non_blocking=True,
                    )
                self.assignment_stream.synchronize()
                flat_kind_cpu = self.assignments_cpu.reshape(-1)
            else:
                live_kind = torch.where(
                    flat_live_slots,
                    flat_kind,
                    torch.full_like(flat_kind, POOL_RANDOM),
                )
                flat_kind_cpu = live_kind.detach().to(device="cpu")
            present_checkpoint_ids = [
                int(checkpoint_id)
                for checkpoint_id in torch.unique(flat_kind_cpu).tolist()
                if int(checkpoint_id) >= 0
            ]
            for checkpoint_id in sorted(present_checkpoint_ids):
                indices_cpu = (
                    (flat_kind_cpu == checkpoint_id)
                    .nonzero(as_tuple=False)
                    .squeeze(-1)
                )
                if indices_cpu.numel():
                    checkpoint_groups.append(
                        (checkpoint_id, indices_cpu.to(device=self.device))
                    )
            # Retired policies must survive while an older episode still uses
            # them, but checking the already-copied CPU assignments avoids a
            # second device synchronization.
            present_set = set(present_checkpoint_ids)
            for checkpoint_id in list(self.retired_ids):
                if checkpoint_id not in present_set:
                    self.models.pop(checkpoint_id, None)
                    self.retired_ids.remove(checkpoint_id)

        def run_frozen(checkpoint_id: int, indices: torch.Tensor) -> None:
            frozen = self.models[checkpoint_id]
            checkpoint_obs = flat_obs.index_select(0, indices)
            with torch.autocast(
                device_type="cuda",
                dtype=self.inference_amp_dtype or torch.float16,
                enabled=self.inference_amp_dtype is not None,
            ):
                logits, next_h, next_c = frozen(
                    checkpoint_obs,
                    flat_h.index_select(0, indices),
                    flat_c.index_select(0, indices),
                )
            # Match the configured learner/evaluation deployment behavior. The
            # backend mask has hidden geometry and would make the same weights
            # behave like a different policy during self-play.
            selected_legal, _ = training_action_mask(
                checkpoint_obs,
                flat_legal.index_select(0, indices),
                self.action_mask_mode,
            )
            distribution = masked_categorical(logits.float(), selected_legal)
            sampled = distribution.sample()
            greedy = distribution.logits.argmax(dim=-1)
            sampled = torch.where(
                flat_deterministic.index_select(0, indices),
                greedy,
                sampled,
            )
            actions.index_copy_(0, indices, sampled)
            flat_h.index_copy_(0, indices, next_h.float())
            flat_c.index_copy_(0, indices, next_c.float())

        if self.parallel_inference and len(checkpoint_groups) > 1:
            current_stream = torch.cuda.current_stream(self.device)
            while len(self.inference_streams) < len(checkpoint_groups):
                self.inference_streams.append(torch.cuda.Stream(device=self.device))
            used_streams = self.inference_streams[: len(checkpoint_groups)]
            for stream, (checkpoint_id, indices) in zip(
                used_streams, checkpoint_groups
            ):
                stream.wait_stream(current_stream)
                with torch.cuda.stream(stream):
                    run_frozen(checkpoint_id, indices)
            for stream in used_streams:
                current_stream.wait_stream(stream)
        else:
            for checkpoint_id, indices in checkpoint_groups:
                run_frozen(checkpoint_id, indices)
        return actions.reshape(self.num_envs, 3)

    def assignment_summary(self) -> str:
        values = self.assignments.reshape(-1)
        parts = []
        for label, kind in self.scripted_opponents:
            parts.append(f"{label}={int((values == kind).sum().item())}")
        parts.append(f"frozen={int((values >= 0).sum().item())}")
        parts.append(
            "greedy_frozen="
            f"{int(((values >= 0) & self.deterministic_assignments.reshape(-1)).sum().item())}"
        )
        focus_id = self.focus_checkpoint_id()
        if focus_id is not None:
            parts.append(f"focus={int((values == focus_id).sum().item())}")
        if self.checkpoints_enabled:
            parts.append(f"active_models={len(self.active_ids)}")
        parts.append(f"copy_lineups={int(self.copy_rows.sum().item())}")
        return " ".join(parts)


EVAL_MODEL_A = 0
EVAL_MODEL_B = 1
EVAL_FILLER = 2


def _evaluation_seat_types(game_index: int, layout: str) -> tuple[int, int, int, int]:
    if layout == "copies":
        return (
            (EVAL_MODEL_A, EVAL_MODEL_B, EVAL_MODEL_A, EVAL_MODEL_B)
            if game_index % 2 == 0
            else (EVAL_MODEL_B, EVAL_MODEL_A, EVAL_MODEL_B, EVAL_MODEL_A)
        )
    if layout == "duel":
        a_seat = game_index % 4
        # Cycle all three relative B positions over blocks of four games.
        b_seat = (a_seat + 1 + (game_index // 4) % 3) % 4
        seats = [EVAL_FILLER] * 4
        seats[a_seat] = EVAL_MODEL_A
        seats[b_seat] = EVAL_MODEL_B
        return tuple(seats)
    if layout == "solo-pair":
        # Alternate one A vs three B with three A vs one B. Across each pair
        # of games equally strong policies still score 0.5, while every game
        # resembles deployment more closely than the 2-vs-2 copies layout.
        solo_seat = (game_index // 2) % 4
        solo_type = EVAL_MODEL_A if game_index % 2 == 0 else EVAL_MODEL_B
        other_type = (
            EVAL_MODEL_B if solo_type == EVAL_MODEL_A else EVAL_MODEL_A
        )
        seats = [other_type] * 4
        seats[solo_seat] = solo_type
        return tuple(seats)
    if layout == "true-duel":
        # The native environment starts only seats zero and one in this layout.
        # Alternating their types removes the otherwise fixed first-seat bias.
        return (
            (EVAL_MODEL_A, EVAL_MODEL_B, EVAL_FILLER, EVAL_FILLER)
            if game_index % 2 == 0
            else (EVAL_MODEL_B, EVAL_MODEL_A, EVAL_FILLER, EVAL_FILLER)
        )
    raise ValueError(f"unsupported comparison layout: {layout}")


def _evaluation_policy_actions(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    deterministic: bool,
) -> torch.Tensor:
    distribution = masked_categorical(logits, legal_mask)
    if deterministic:
        return distribution.logits.argmax(dim=-1)
    return distribution.sample()


@torch.inference_mode()
def compare_policies_cuda(
    model_a: ActorCritic,
    model_b: ActorCritic | FrozenActor | None,
    *,
    games: int = 512,
    num_envs: int = 256,
    max_turns: int = 1000,
    seed: int = 123,
    layout: str = "copies",
    fill: str = "random",
    model_b_kind: str = "model",
    deterministic: bool = True,
    action_mask_mode: str = "observable",
    device: torch.device | str = "cuda",
) -> dict[str, float]:
    """Compare recurrent policies in parallel, entirely through CUDA rollouts.

    ``copies`` uses two seats per policy. ``duel`` rotates one A and one B
    through all seats and fills the remainder with scripted agents.
    ``solo-pair`` alternates one A vs three B with three A vs one B, preserving
    a 0.5 equal-policy baseline while resembling deployment. ``true-duel``
    starts only one A and one B and alternates their two live seats. Games
    continue after seat zero dies and end only at the true terminal state or
    ``max_turns``.
    """

    if games <= 0:
        raise ValueError("games must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if layout not in ("copies", "duel", "solo-pair", "true-duel"):
        raise ValueError("layout must be copies, duel, solo-pair, or true-duel")
    if fill not in ("random", "hungry"):
        raise ValueError("fill must be 'random' or 'hungry'")
    valid_model_b_kinds = ("model", *(label for label, _ in SCRIPTED_OPPONENTS))
    if model_b_kind not in valid_model_b_kinds:
        raise ValueError(
            "model_b_kind must be one of " + ", ".join(valid_model_b_kinds)
        )
    if model_b_kind == "model" and model_b is None:
        raise ValueError("model_b is required when model_b_kind='model'")

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("compare_policies_cuda requires a CUDA device")
    batch_envs = min(num_envs, games)
    torch.manual_seed(seed)
    model_a = model_a.to(device).eval()
    if model_b is not None:
        model_b = model_b.to(device).eval()
    env = hisss.CudaBlackoutTorchVecEnv(
        batch_envs,
        seed=seed,
        device=str(device),
        duel_probability=1.0 if layout == "true-duel" else 0.0,
    )

    obs_all = legal_all = None
    try:
        obs_all, legal_all = env.reset_all()
        validate_cuda_all_seat_reset(
            obs_all,
            legal_all,
            context="policy comparison",
            allow_duels=layout == "true-duel",
        )
        legal_all = legal_cuda_to_model(legal_all)
        seat_types = torch.tensor(
            [_evaluation_seat_types(i, layout) for i in range(batch_envs)],
            dtype=torch.long,
            device=device,
        )
        active = torch.ones(batch_envs, dtype=torch.bool, device=device)
        next_game = batch_envs
        completed = 0
        a_wins = b_wins = fill_wins = draws = 0
        a_deaths = b_deaths = 0
        turn_sum = 0

        a_h, a_c = model_a.initial_actor_state(batch_envs * 4, device)
        if model_b_kind == "model":
            b_h, b_c = model_b.initial_actor_state(batch_envs * 4, device)
        else:
            b_h = b_c = None

        while completed < games:
            flat_obs = obs_all.reshape(batch_envs * 4, *OBS_SHAPE)
            flat_server_legal = legal_all.reshape(batch_envs * 4, 4)
            # This mirrors the configured deployment mask instead of silently
            # using privileged full-state geometry for compared policies.
            flat_policy_legal, _ = training_action_mask(
                flat_obs,
                flat_server_legal,
                action_mask_mode,
            )

            logits_a, a_h_next, a_c_next = model_a.actor_forward(
                flat_obs, a_h, a_c
            )
            actions_a = _evaluation_policy_actions(
                logits_a, flat_policy_legal, deterministic
            )
            if model_b_kind == "model":
                logits_b, b_h_next, b_c_next = model_b.actor_forward(
                    flat_obs, b_h, b_c
                )
                actions_b = _evaluation_policy_actions(
                    logits_b, flat_policy_legal, deterministic
                )
            elif model_b_kind == "hungry":
                actions_b = hungry_legal_actions(flat_obs, flat_server_legal)
            elif model_b_kind == "random":
                actions_b = random_legal_actions(flat_server_legal)
            elif model_b_kind == "best":
                best_selection = seat_types == EVAL_MODEL_B
                best_cuda = env.best_actions(best_selection).reshape(-1)
                actions_b = actions_cuda_to_model(best_cuda)
            else:
                profile_ids = torch.where(
                    seat_types == EVAL_MODEL_B,
                    NATIVE_PROFILE_BY_LABEL[model_b_kind],
                    -1,
                ).to(torch.int32)
                heuristic_cuda = env.heuristic_actions(profile_ids).reshape(-1)
                actions_b = actions_cuda_to_model(heuristic_cuda)
            if fill == "hungry":
                filler_actions = hungry_legal_actions(flat_obs, flat_server_legal)
            else:
                filler_actions = random_legal_actions(flat_server_legal)

            flat_types = seat_types.reshape(-1)
            model_actions = torch.where(
                flat_types == EVAL_MODEL_A,
                actions_a,
                torch.where(
                    flat_types == EVAL_MODEL_B,
                    actions_b,
                    filler_actions,
                ),
            ).reshape(batch_envs, 4)
            step = env.step_all_eval(
                actions_model_to_cuda(model_actions), max_turns=max_turns
            )

            not_done = (~step.done).float().view(batch_envs, 1, 1)
            a_h = a_h_next.reshape(batch_envs, 4, -1) * not_done
            a_c = a_c_next.reshape(batch_envs, 4, -1) * not_done
            a_h = a_h.reshape(batch_envs * 4, -1)
            a_c = a_c.reshape(batch_envs * 4, -1)
            if model_b_kind == "model":
                b_h = b_h_next.reshape(batch_envs, 4, -1) * not_done
                b_c = b_c_next.reshape(batch_envs, 4, -1) * not_done
                b_h = b_h.reshape(batch_envs * 4, -1)
                b_c = b_c.reshape(batch_envs * 4, -1)

            finished = (step.done & active).nonzero(as_tuple=False).squeeze(-1)
            if finished.numel():
                finished_list = finished.tolist()
                winners = step.winner.index_select(0, finished).tolist()
                turns = step.turns.index_select(0, finished).tolist()
                alive_rows = step.alive.index_select(0, finished).cpu()
                type_rows = seat_types.index_select(0, finished).cpu()

                for offset, env_index in enumerate(finished_list):
                    row_types = type_rows[offset]
                    row_alive = alive_rows[offset]
                    winner = int(winners[offset])
                    turn_sum += int(turns[offset])
                    a_deaths += int(
                        ((row_types == EVAL_MODEL_A) & ~row_alive).sum().item()
                    )
                    b_deaths += int(
                        ((row_types == EVAL_MODEL_B) & ~row_alive).sum().item()
                    )
                    if winner < 0:
                        draws += 1
                    else:
                        winner_type = int(row_types[winner].item())
                        if winner_type == EVAL_MODEL_A:
                            a_wins += 1
                        elif winner_type == EVAL_MODEL_B:
                            b_wins += 1
                        else:
                            fill_wins += 1
                    completed += 1

                    if next_game < games:
                        seat_types[env_index] = torch.tensor(
                            _evaluation_seat_types(next_game, layout),
                            dtype=torch.long,
                            device=device,
                        )
                        next_game += 1
                    else:
                        active[env_index] = False

            obs_all = step.obs
            legal_all = legal_cuda_to_model(step.legal_mask)

        total = float(games)
        return {
            "games": float(games),
            "a_wins": float(a_wins),
            "b_wins": float(b_wins),
            "fill_wins": float(fill_wins),
            "draws": float(draws),
            "a_deaths": float(a_deaths),
            "b_deaths": float(b_deaths),
            "a_win_rate": a_wins / total,
            "b_win_rate": b_wins / total,
            "fill_win_rate": fill_wins / total,
            "draw_rate": draws / total,
            "avg_turns": turn_sum / total,
        }
    finally:
        env.close()


@torch.no_grad()
def collect_rollout(
    env: hisss.CudaBlackoutTorchVecEnv,
    model: ActorCritic,
    obs: torch.Tensor,
    legal_mask: torch.Tensor,
    actor_h: torch.Tensor,
    actor_c: torch.Tensor,
    critic_h: torch.Tensor,
    critic_c: torch.Tensor,
    alive_count: torch.Tensor,
    n_steps: int,
    seq_len: int,
    obs_storage_dtype: torch.dtype,
    gamma: float,
    gae_lambda: float,
    action_mask_mode: str,
    potential_length_coef: float,
    potential_health_coef: float,
    potential_mobility_coef: float,
    reward_scheme: str = "kill",
    inference_amp_dtype: torch.dtype | None = None,
    learner_observable_masks_fn: Callable[
        [torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ] = observable_action_masks,
):
    device = obs.device
    num_envs = obs.shape[0]
    use_action_mask = action_mask_mode != "none"
    n_chunks = (n_steps + seq_len - 1) // seq_len

    obs_buf = torch.empty(
        (n_steps, num_envs, *OBS_SHAPE),
        dtype=obs_storage_dtype,
        device=device,
    )
    legal_buf = torch.empty((n_steps, num_envs, 4), dtype=torch.bool, device=device)
    action_buf = torch.empty((n_steps, num_envs), dtype=torch.int64, device=device)
    logprob_buf = torch.empty((n_steps, num_envs), device=device)
    reward_buf = torch.empty((n_steps, num_envs), device=device)
    env_reward_buf = torch.empty_like(reward_buf)
    done_buf = torch.empty((n_steps, num_envs), dtype=torch.bool, device=device)
    value_buf = torch.empty((n_steps, num_envs), device=device)
    illegal_buf = torch.empty((n_steps, num_envs), dtype=torch.bool, device=device)
    state_shape = (n_chunks, num_envs, model.hidden_size)
    actor_h0_buf = torch.empty(state_shape, device=device)
    actor_c0_buf = torch.empty(state_shape, device=device)
    critic_h0_buf = torch.empty(state_shape, device=device)
    critic_c0_buf = torch.empty(state_shape, device=device)

    need_potential = any(
        coefficient != 0.0
        for coefficient in (
            potential_length_coef,
            potential_health_coef,
            potential_mobility_coef,
        )
    )
    policy_legal_mask, use_action_mask, mobility_mask = rollout_action_masks(
        obs,
        legal_mask,
        action_mask_mode,
        need_mobility_mask=potential_mobility_coef != 0.0,
        observable_masks_fn=learner_observable_masks_fn,
    )
    current_potential = (
        observation_potential(
            obs,
            potential_length_coef,
            potential_health_coef,
            potential_mobility_coef,
            mobility_mask,
        )
        if need_potential
        else None
    )

    for t in range(n_steps):
        if t % seq_len == 0:
            chunk_idx = t // seq_len
            actor_h0_buf[chunk_idx].copy_(actor_h)
            actor_c0_buf[chunk_idx].copy_(actor_c)
            critic_h0_buf[chunk_idx].copy_(critic_h)
            critic_c0_buf[chunk_idx].copy_(critic_c)
        obs_buf[t].copy_(obs)
        legal_buf[t].copy_(policy_legal_mask)
        with torch.autocast(
            device_type="cuda",
            dtype=inference_amp_dtype or torch.float16,
            enabled=inference_amp_dtype is not None,
        ):
            (
                logits,
                values,
                actor_h_next,
                actor_c_next,
                critic_h_next,
                critic_c_next,
            ) = model(obs, actor_h, actor_c, critic_h, critic_c)
        logits = logits.float()
        values = values.float()
        log_probs = policy_log_probs(logits, policy_legal_mask, use_action_mask)
        actions, logprobs = sample_policy_actions(log_probs)

        step = env.step(actions_model_to_cuda(actions))
        next_legal_mask = legal_cuda_to_model(step.legal_mask)
        (
            next_policy_legal_mask,
            _,
            next_mobility_mask,
        ) = rollout_action_masks(
            step.obs,
            next_legal_mask,
            action_mask_mode,
            need_mobility_mask=potential_mobility_coef != 0.0,
            observable_masks_fn=learner_observable_masks_fn,
        )
        next_potential = (
            observation_potential(
                step.obs,
                potential_length_coef,
                potential_health_coef,
                potential_mobility_coef,
                next_mobility_mask,
            )
            if need_potential
            else None
        )
        environment_reward, alive_count = environment_reward_for_scheme(
            step.rewards, step.done, reward_scheme, alive_count
        )
        shaped_reward = potential_shaped_reward_from_values(
            environment_reward,
            current_potential,
            next_potential,
            step.done,
            gamma,
        )
        action_buf[t].copy_(actions)
        logprob_buf[t].copy_(logprobs)
        reward_buf[t].copy_(shaped_reward)
        env_reward_buf[t].copy_(environment_reward)
        done_buf[t].copy_(step.done)
        value_buf[t].copy_(values)
        illegal_buf[t].copy_(~legal_mask.gather(1, actions.unsqueeze(1)).squeeze(1))

        obs = step.obs
        legal_mask = next_legal_mask
        policy_legal_mask = next_policy_legal_mask
        current_potential = next_potential
        not_done = (~step.done).float().unsqueeze(-1)
        actor_h = actor_h_next * not_done
        actor_c = actor_c_next * not_done
        critic_h = critic_h_next * not_done
        critic_c = critic_c_next * not_done

    with torch.autocast(
        device_type="cuda",
        dtype=inference_amp_dtype or torch.float16,
        enabled=inference_amp_dtype is not None,
    ):
        _, next_values, *_ = model(
            obs, actor_h, actor_c, critic_h, critic_c
        )
    next_values = next_values.float()
    advantages, returns = compute_gae(
        reward_buf, done_buf, value_buf, next_values, gamma, gae_lambda
    )

    rollout = Rollout(
        obs=obs_buf,
        legal_mask=legal_buf,
        actions=action_buf,
        logprobs=logprob_buf,
        rewards=reward_buf,
        dones=done_buf,
        values=value_buf,
        advantages=advantages,
        returns=returns,
        env_rewards=env_reward_buf,
        illegal_actions=illegal_buf,
        actor_h0=actor_h0_buf,
        actor_c0=actor_c0_buf,
        critic_h0=critic_h0_buf,
        critic_c0=critic_c0_buf,
        seq_len=seq_len,
        use_action_mask=use_action_mask,
    )
    return (
        rollout,
        obs,
        legal_mask,
        actor_h,
        actor_c,
        critic_h,
        critic_c,
        alive_count,
    )


@torch.no_grad()
def collect_pool_rollout(
    env: hisss.CudaBlackoutTorchVecEnv,
    model: ActorCritic,
    opponent_pool: CudaOpponentPool,
    obs_all: torch.Tensor,
    legal_all: torch.Tensor,
    actor_h: torch.Tensor,
    actor_c: torch.Tensor,
    critic_h: torch.Tensor,
    critic_c: torch.Tensor,
    alive_count: torch.Tensor,
    n_steps: int,
    seq_len: int,
    obs_storage_dtype: torch.dtype,
    gamma: float,
    gae_lambda: float,
    action_mask_mode: str,
    potential_length_coef: float,
    potential_health_coef: float,
    potential_mobility_coef: float,
    reward_scheme: str = "kill",
    inference_amp_dtype: torch.dtype | None = None,
    learner_observable_masks_fn: Callable[
        [torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ] = observable_action_masks,
):
    device = obs_all.device
    num_envs = obs_all.shape[0]
    use_action_mask = action_mask_mode != "none"
    n_chunks = (n_steps + seq_len - 1) // seq_len

    obs_buf = torch.empty(
        (n_steps, num_envs, *OBS_SHAPE),
        dtype=obs_storage_dtype,
        device=device,
    )
    legal_buf = torch.empty((n_steps, num_envs, 4), dtype=torch.bool, device=device)
    action_buf = torch.empty((n_steps, num_envs), dtype=torch.int64, device=device)
    logprob_buf = torch.empty((n_steps, num_envs), device=device)
    reward_buf = torch.empty((n_steps, num_envs), device=device)
    env_reward_buf = torch.empty_like(reward_buf)
    done_buf = torch.empty((n_steps, num_envs), dtype=torch.bool, device=device)
    value_buf = torch.empty((n_steps, num_envs), device=device)
    illegal_buf = torch.empty((n_steps, num_envs), dtype=torch.bool, device=device)
    state_shape = (n_chunks, num_envs, model.hidden_size)
    actor_h0_buf = torch.empty(state_shape, device=device)
    actor_c0_buf = torch.empty(state_shape, device=device)
    critic_h0_buf = torch.empty(state_shape, device=device)
    critic_c0_buf = torch.empty(state_shape, device=device)

    obs = obs_all[:, 0]
    legal_mask = legal_all[:, 0]
    need_potential = any(
        coefficient != 0.0
        for coefficient in (
            potential_length_coef,
            potential_health_coef,
            potential_mobility_coef,
        )
    )
    policy_legal_mask, use_action_mask, mobility_mask = rollout_action_masks(
        obs,
        legal_mask,
        action_mask_mode,
        need_mobility_mask=potential_mobility_coef != 0.0,
        observable_masks_fn=learner_observable_masks_fn,
    )
    current_potential = (
        observation_potential(
            obs,
            potential_length_coef,
            potential_health_coef,
            potential_mobility_coef,
            mobility_mask,
        )
        if need_potential
        else None
    )
    all_actions = torch.empty((num_envs, 4), dtype=torch.long, device=device)

    for t in range(n_steps):
        if t % seq_len == 0:
            chunk_idx = t // seq_len
            actor_h0_buf[chunk_idx].copy_(actor_h)
            actor_c0_buf[chunk_idx].copy_(actor_c)
            critic_h0_buf[chunk_idx].copy_(critic_h)
            critic_c0_buf[chunk_idx].copy_(critic_c)
        obs_buf[t].copy_(obs)
        legal_buf[t].copy_(policy_legal_mask)

        with torch.autocast(
            device_type="cuda",
            dtype=inference_amp_dtype or torch.float16,
            enabled=inference_amp_dtype is not None,
        ):
            (
                logits,
                values,
                actor_h_next,
                actor_c_next,
                critic_h_next,
                critic_c_next,
            ) = model(obs, actor_h, actor_c, critic_h, critic_c)
        logits = logits.float()
        values = values.float()
        log_probs = policy_log_probs(logits, policy_legal_mask, use_action_mask)
        actions, logprobs = sample_policy_actions(log_probs)

        opponent_actions = opponent_pool.actions(
            env, obs_all[:, 1:4], legal_all[:, 1:4]
        )

        all_actions[:, 0] = actions
        all_actions[:, 1:4] = opponent_actions
        step = env.step_all(actions_model_to_cuda(all_actions))
        next_legal_all = legal_cuda_to_model(step.legal_mask)
        next_agent_obs = step.obs[:, 0]
        next_agent_legal = next_legal_all[:, 0]
        (
            next_policy_legal_mask,
            _,
            next_mobility_mask,
        ) = rollout_action_masks(
            next_agent_obs,
            next_agent_legal,
            action_mask_mode,
            need_mobility_mask=potential_mobility_coef != 0.0,
            observable_masks_fn=learner_observable_masks_fn,
        )
        next_potential = (
            observation_potential(
                next_agent_obs,
                potential_length_coef,
                potential_health_coef,
                potential_mobility_coef,
                next_mobility_mask,
            )
            if need_potential
            else None
        )
        reset_alive_count = next_legal_all.any(dim=-1).sum(dim=-1)
        environment_reward, alive_count = environment_reward_for_scheme(
            step.rewards,
            step.done,
            reward_scheme,
            alive_count,
            reset_alive_count,
        )
        shaped_reward = potential_shaped_reward_from_values(
            environment_reward,
            current_potential,
            next_potential,
            step.done,
            gamma,
        )

        action_buf[t].copy_(actions)
        logprob_buf[t].copy_(logprobs)
        reward_buf[t].copy_(shaped_reward)
        env_reward_buf[t].copy_(environment_reward)
        done_buf[t].copy_(step.done)
        value_buf[t].copy_(values)
        illegal_buf[t].copy_(~legal_mask.gather(1, actions.unsqueeze(1)).squeeze(1))

        obs_all = step.obs
        legal_all = next_legal_all
        obs = next_agent_obs
        legal_mask = next_agent_legal
        policy_legal_mask = next_policy_legal_mask
        current_potential = next_potential
        not_done = (~step.done).float().unsqueeze(-1)
        actor_h = actor_h_next * not_done
        actor_c = actor_c_next * not_done
        critic_h = critic_h_next * not_done
        critic_c = critic_c_next * not_done
        opponent_pool.resample(
            step.done,
            duel_rows=reset_alive_count == 2,
        )

    obs = obs_all[:, 0]
    with torch.autocast(
        device_type="cuda",
        dtype=inference_amp_dtype or torch.float16,
        enabled=inference_amp_dtype is not None,
    ):
        _, next_values, *_ = model(
            obs, actor_h, actor_c, critic_h, critic_c
        )
    next_values = next_values.float()
    advantages, returns = compute_gae(
        reward_buf, done_buf, value_buf, next_values, gamma, gae_lambda
    )

    rollout = Rollout(
        obs=obs_buf,
        legal_mask=legal_buf,
        actions=action_buf,
        logprobs=logprob_buf,
        rewards=reward_buf,
        dones=done_buf,
        values=value_buf,
        advantages=advantages,
        returns=returns,
        env_rewards=env_reward_buf,
        illegal_actions=illegal_buf,
        actor_h0=actor_h0_buf,
        actor_c0=actor_c0_buf,
        critic_h0=critic_h0_buf,
        critic_c0=critic_c0_buf,
        seq_len=seq_len,
        use_action_mask=use_action_mask,
    )
    return (
        rollout,
        obs_all,
        legal_all,
        actor_h,
        actor_c,
        critic_h,
        critic_c,
        alive_count,
    )


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    batch_size: int,
    n_epochs: int,
    clip_range: float,
    clip_range_vf: float | None,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    target_kl: float | None,
    training_amp_dtype: torch.dtype | None = None,
    grad_scaler=None,
    ppo_loss_fn: Callable | None = None,
    kl_stop_mode: str = "minibatch",
    actor_max_grad_norm: float | None = None,
    critic_max_grad_norm: float | None = None,
    actor_reference_parameters: list[torch.Tensor] | None = None,
    actor_reference_l2_coef: float = 0.0,
):
    if kl_stop_mode not in ("minibatch", "epoch"):
        raise ValueError("kl_stop_mode must be 'minibatch' or 'epoch'")
    if actor_reference_l2_coef < 0.0:
        raise ValueError("actor_reference_l2_coef must be non-negative")
    n_steps, num_envs = rollout.actions.shape
    seq_len = rollout.seq_len
    envs_per_batch = max(1, batch_size // seq_len)
    advantages = (rollout.advantages - rollout.advantages.mean()) / (
        rollout.advantages.std() + 1e-8
    )
    # Dones are tiny. Keeping one CPU copy lets us construct recurrent episode
    # fragments without synchronizing the full CUDA stream in every minibatch.
    dones_cpu = rollout.dones.detach().to(device="cpu")
    train_metric_sums = torch.zeros(4, device=rollout.obs.device)
    grad_norm_sum = torch.zeros((), device=rollout.obs.device)
    actor_grad_norm_sum = torch.zeros((), device=rollout.obs.device)
    critic_grad_norm_sum = torch.zeros((), device=rollout.obs.device)
    finite_grad_count = torch.zeros((), device=rollout.obs.device)
    nonfinite_grad_count = torch.zeros((), device=rollout.obs.device)
    metric_sums = torch.zeros(2, device=rollout.obs.device)
    reference_loss_sum = torch.zeros((), device=rollout.obs.device)
    update_count = 0
    optimizer_step_count = 0
    completed_epoch_count = 0
    early_stop = False
    chunks = math.ceil(n_steps / seq_len)
    minibatches_per_chunk = math.ceil(num_envs / envs_per_batch)
    planned_minibatches = n_epochs * chunks * minibatches_per_chunk
    # A scheduled Python float would become a Dynamo value guard. With a new
    # value every update, the full-graph compiled loss would recompile until it
    # hits torch._dynamo.config.recompile_limit and aborts. A scalar tensor is a
    # regular graph input, so its value can change without changing the graph.
    ent_coef_tensor = rollout.values.new_tensor(ent_coef)
    actor_reference_l2_coef_tensor = rollout.values.new_tensor(
        actor_reference_l2_coef
    )
    separate_branch_clipping = (
        actor_max_grad_norm is not None or critic_max_grad_norm is not None
    )
    actor_parameters, critic_parameters = actor_critic_parameter_groups(model)

    for _ in range(n_epochs):
        if early_stop:
            break
        epoch_kl_sum = torch.zeros((), device=rollout.obs.device)
        epoch_update_count = 0
        for seq_start in range(0, n_steps, seq_len):
            if early_stop:
                break
            seq_stop = min(seq_start + seq_len, n_steps)
            chunk_idx = seq_start // seq_len
            seq_perm_cpu = torch.randperm(num_envs)
            for env_start in range(0, num_envs, envs_per_batch):
                env_mb_cpu = seq_perm_cpu[env_start : env_start + envs_per_batch]
                env_mb = env_mb_cpu.to(device=rollout.obs.device)
                obs_mb = rollout.obs[seq_start:seq_stop, env_mb]
                if training_amp_dtype is None:
                    obs_mb = obs_mb.float()
                legal_mb = rollout.legal_mask[seq_start:seq_stop, env_mb]
                actions_mb = rollout.actions[seq_start:seq_stop, env_mb]
                old_logprobs_mb = rollout.logprobs[seq_start:seq_stop, env_mb]
                old_values_mb = rollout.values[seq_start:seq_stop, env_mb]
                returns_mb = rollout.returns[seq_start:seq_stop, env_mb]
                advantages_mb = advantages[seq_start:seq_stop, env_mb]
                dones_mb = rollout.dones[seq_start:seq_stop, env_mb]
                actor_h = rollout.actor_h0[chunk_idx, env_mb].detach()
                actor_c = rollout.actor_c0[chunk_idx, env_mb].detach()
                critic_h = rollout.critic_h0[chunk_idx, env_mb].detach()
                critic_c = rollout.critic_c0[chunk_idx, env_mb].detach()
                episode_layout = _episode_segments(
                    dones_cpu[seq_start:seq_stop, env_mb_cpu],
                    output_device=rollout.obs.device,
                )

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=training_amp_dtype or torch.float16,
                    enabled=training_amp_dtype is not None,
                ):
                    logits, values = forward_sequence(
                        model,
                        obs_mb,
                        dones_mb,
                        actor_h,
                        actor_c,
                        critic_h,
                        critic_c,
                        episode_layout,
                    )
                loss_fn = ppo_loss_components if ppo_loss_fn is None else ppo_loss_fn
                (
                    loss,
                    policy_loss,
                    value_loss,
                    entropy,
                    approx_kl,
                    clipfrac,
                ) = loss_fn(
                    logits,
                    values,
                    legal_mb,
                    actions_mb,
                    old_logprobs_mb,
                    old_values_mb,
                    returns_mb,
                    advantages_mb,
                    rollout.use_action_mask,
                    clip_range,
                    clip_range_vf,
                    ent_coef_tensor,
                    vf_coef,
                )
                reference_loss = (
                    actor_reference_l2_loss(
                        actor_parameters,
                        actor_reference_parameters,
                        actor_reference_l2_coef_tensor,
                    )
                    if actor_reference_l2_coef > 0.0
                    and actor_reference_parameters
                    else loss.new_zeros(())
                )
                loss = loss + reference_loss

                with torch.no_grad():
                    metric_sums.add_(torch.stack((approx_kl, clipfrac)))
                    epoch_kl_sum.add_(approx_kl)
                    update_count += 1
                    epoch_update_count += 1

                if target_kl is not None and kl_stop_mode == "minibatch":
                    if float(approx_kl.item()) > 1.5 * target_kl:
                        early_stop = True
                        break

                if grad_scaler is not None and grad_scaler.is_enabled():
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if separate_branch_clipping:
                    actor_grad_norm = nn.utils.clip_grad_norm_(
                        actor_parameters,
                        actor_max_grad_norm or max_grad_norm,
                    )
                    critic_grad_norm = nn.utils.clip_grad_norm_(
                        critic_parameters,
                        critic_max_grad_norm or max_grad_norm,
                    )
                    grad_norm = torch.sqrt(
                        actor_grad_norm.square() + critic_grad_norm.square()
                    )
                else:
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    actor_grad_norm = grad_norm.new_zeros(())
                    critic_grad_norm = grad_norm.new_zeros(())
                if grad_scaler is not None and grad_scaler.is_enabled():
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                with torch.no_grad():
                    train_metric_sums.add_(
                        torch.stack(
                            (
                                loss.detach(),
                                policy_loss.detach(),
                                value_loss.detach(),
                                entropy.detach(),
                            )
                        )
                    )
                    reference_loss_sum.add_(reference_loss.detach())
                    finite_grad = torch.isfinite(grad_norm)
                    grad_norm_sum.add_(
                        torch.where(finite_grad, grad_norm.detach(), 0.0)
                    )
                    actor_grad_norm_sum.add_(
                        torch.where(
                            torch.isfinite(actor_grad_norm),
                            actor_grad_norm.detach(),
                            0.0,
                        )
                    )
                    critic_grad_norm_sum.add_(
                        torch.where(
                            torch.isfinite(critic_grad_norm),
                            critic_grad_norm.detach(),
                            0.0,
                        )
                    )
                    finite_grad_count.add_(finite_grad.float())
                    nonfinite_grad_count.add_((~finite_grad).float())
                    optimizer_step_count += 1
        if epoch_update_count == chunks * minibatches_per_chunk:
            completed_epoch_count += 1
        # A single synchronization per epoch replaces one synchronization per
        # minibatch. Checking the epoch mean is also less sensitive to one
        # unusually difficult recurrent minibatch than the previous rule.
        if (
            target_kl is not None
            and kl_stop_mode == "epoch"
            and epoch_update_count
        ):
            epoch_kl = float((epoch_kl_sum / epoch_update_count).item())
            if epoch_kl > 1.5 * target_kl:
                early_stop = True

    with torch.no_grad():
        return_var = rollout.returns.var(unbiased=False)
        explained_variance = torch.where(
            return_var > 1e-8,
            1.0
            - (rollout.returns - rollout.values).var(unbiased=False)
            / return_var.clamp_min(1e-8),
            torch.zeros_like(return_var),
        )
    if optimizer_step_count == 0:
        return {
            "explained_variance": float(explained_variance.item()),
            "early_stop": float(early_stop),
            "minibatches_attempted": float(update_count),
            "optimizer_steps": 0.0,
            "optimizer_step_fraction": 0.0,
            "epochs_completed": 0.0,
            "nonfinite_grad_count": 0.0,
            "actor_reference_loss": 0.0,
        }
    averaged_train_metrics = train_metric_sums / optimizer_step_count
    averaged_metrics = metric_sums / max(update_count, 1)
    metric_values = torch.stack(
        (
            *averaged_train_metrics,
            grad_norm_sum / finite_grad_count.clamp_min(1.0),
            actor_grad_norm_sum / finite_grad_count.clamp_min(1.0),
            critic_grad_norm_sum / finite_grad_count.clamp_min(1.0),
            averaged_metrics[0],
            averaged_metrics[1],
            explained_variance,
            reference_loss_sum / optimizer_step_count,
        )
    ).tolist()
    nonfinite_steps = float(nonfinite_grad_count.item())
    actual_optimizer_steps = optimizer_step_count - nonfinite_steps
    return {
        "loss": metric_values[0],
        "policy_loss": metric_values[1],
        "value_loss": metric_values[2],
        "entropy": metric_values[3],
        "grad_norm": metric_values[4],
        "actor_grad_norm": metric_values[5],
        "critic_grad_norm": metric_values[6],
        "approx_kl": metric_values[7],
        "clipfrac": metric_values[8],
        "explained_variance": metric_values[9],
        "actor_reference_loss": metric_values[10],
        "early_stop": float(early_stop),
        "minibatches_attempted": float(update_count),
        "optimizer_steps": actual_optimizer_steps,
        "optimizer_step_fraction": actual_optimizer_steps / planned_minibatches,
        "epochs_completed": float(completed_epoch_count),
        "nonfinite_grad_count": nonfinite_steps,
    }


def save_training_checkpoint(
    path: str,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    update: int,
    args,
    grad_scaler=None,
):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "activation": "tanh",
            "separate_critic_lstm": True,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "grad_scaler_state_dict": (
                grad_scaler.state_dict()
                if grad_scaler is not None and grad_scaler.is_enabled()
                else None
            ),
            "total_steps": total_steps,
            "update": update,
            "args": vars(args),
        },
        tmp,
    )
    os.replace(tmp, target)


def save_league_champion(
    path: str,
    model: ActorCritic,
    total_steps: int,
    update: int,
) -> Path:
    """Atomically persist the deployable champion without optimizer state."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "activation": "tanh",
            "separate_critic_lstm": True,
            "league_champion": True,
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "total_steps": total_steps,
            "update": update,
        },
        tmp,
    )
    os.replace(tmp, target)
    return target


def save_periodic_training_checkpoint(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    directory: str,
    prefix: str,
    total_steps: int,
    update: int,
    args,
    grad_scaler=None,
    max_checkpoints: int = 0,
) -> Path:
    """Atomically save a resumable checkpoint and apply retention."""

    target_dir = Path(directory).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = Path(prefix).stem or "ppo_bs_lstm_cuda"
    target = target_dir / (
        f"{safe_prefix}_steps_{total_steps:012d}_u{update:08d}.pt"
    )
    save_training_checkpoint(
        str(target),
        model,
        optimizer,
        total_steps,
        update,
        args,
        grad_scaler,
    )

    if max_checkpoints > 0:
        checkpoints = sorted(
            target_dir.glob(f"{safe_prefix}_steps_*_u*.pt"),
            key=lambda path: path.stat().st_mtime,
        )
        for stale in checkpoints[:-max_checkpoints]:
            stale.unlink(missing_ok=True)
    return target


def load_training_checkpoint(
    path: str,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_scaler=None,
) -> tuple[int, int]:
    payload = torch.load(path, map_location=device, weights_only=False)
    total_steps = int(payload.get("total_steps", 0))
    saved_update = payload.get("update")
    if saved_update is None:
        saved_args = payload.get("args", {})
        steps_per_update = int(saved_args.get("num_envs", 0)) * int(
            saved_args.get("n_steps", 0)
        )
        saved_update = total_steps // steps_per_update if steps_per_update else 0
    completed_updates = int(saved_update)
    legacy = load_model_state(
        model,
        checkpoint_model_state(payload),
        allow_legacy=True,
    )
    if legacy:
        print(
            "[resume] upgraded legacy shared LSTM into actor+critic LSTMs; "
            "optimizer state was not restored and heads now use Tanh"
        )
        return total_steps, completed_updates
    if isinstance(payload, dict) and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    scaler_state = (
        payload.get("grad_scaler_state_dict")
        if isinstance(payload, dict)
        else None
    )
    if (
        scaler_state
        and grad_scaler is not None
        and grad_scaler.is_enabled()
    ):
        grad_scaler.load_state_dict(scaler_state)
    return total_steps, completed_updates


def load_warm_start(
    path: str,
    model: ActorCritic,
    device: torch.device,
) -> tuple[int, int, str, float | None]:
    """Load policy weights while deliberately starting a fresh optimizer run."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".zip":
        imported = load_sb3_recurrent_ppo(str(source), device=device)
        model.load_state_dict(imported.state_dict(), strict=True)
        total_steps, update = checkpoint_counters_from_name(source)
        return total_steps, update, "SB3 zip", None

    payload = torch.load(source, map_location=device, weights_only=False)
    legacy = load_model_state(
        model,
        checkpoint_model_state(payload),
        allow_legacy=True,
    )
    total_steps = (
        int(payload.get("total_steps", 0) or 0)
        if isinstance(payload, dict)
        else 0
    )
    update = int(payload.get("update", 0) or 0) if isinstance(payload, dict) else 0
    optimizer_groups = (
        payload.get("optimizer_state_dict", {}).get("param_groups", [])
        if isinstance(payload, dict)
        else []
    )
    source_lr = (
        float(optimizer_groups[0]["lr"])
        if optimizer_groups and "lr" in optimizer_groups[0]
        else None
    )
    source_kind = "legacy Torch checkpoint" if legacy else "Torch checkpoint"
    return total_steps, update, source_kind, source_lr


def create_tensorboard_writer(args, total_steps: int, model: nn.Module):
    """Create a resumable TensorBoard writer when logging is enabled."""

    if not args.tensorboard_log_dir:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard logging requires the 'tensorboard' package; "
            "install the project dependencies first"
        ) from exc

    log_dir = Path(args.tensorboard_log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(
        log_dir=str(log_dir),
        purge_step=total_steps + 1 if args.resume_path else None,
        max_queue=args.tensorboard_max_queue,
        flush_secs=args.tensorboard_flush_secs,
    )
    config = json.dumps(vars(args), indent=2, sort_keys=True)
    writer.add_text("run/configuration", f"```json\n{config}\n```", total_steps)
    writer.add_scalar(
        "system/model_parameters",
        sum(parameter.numel() for parameter in model.parameters()),
        total_steps,
    )
    print(f"[tensorboard] logging to {log_dir.resolve()}")
    return writer


def log_tensorboard_update(
    writer,
    *,
    update: int,
    total_steps: int,
    update_steps: int,
    rollout: Rollout,
    stats: dict[str, float],
    learning_rate: float,
    entropy_coefficient: float,
    rollout_seconds: float,
    ppo_seconds: float,
    cumulative_steps_per_second: float,
    device: torch.device,
    grad_scaler,
    opponent_pool: CudaOpponentPool | None,
) -> None:
    """Write one synchronized, low-overhead training snapshot."""

    if writer is None:
        return
    update_seconds = max(rollout_seconds + ppo_seconds, 1e-9)
    rollout_values = torch.stack(
        (
            rollout.env_rewards.mean(),
            rollout.rewards.mean(),
            rollout.dones.float().mean(),
            rollout.illegal_actions.float().mean(),
            rollout.advantages.mean(),
            rollout.advantages.std(unbiased=False),
            rollout.returns.mean(),
            rollout.values.mean(),
        )
    ).tolist()
    rollout_tags = (
        "rollout/env_reward_mean",
        "rollout/shaped_reward_mean",
        "rollout/done_fraction",
        "rollout/illegal_action_fraction",
        "rollout/advantage_mean",
        "rollout/advantage_std",
        "rollout/return_mean",
        "rollout/value_mean",
    )
    for tag, value in zip(rollout_tags, rollout_values):
        writer.add_scalar(tag, value, total_steps)

    for key in (
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "actor_reference_loss",
        "approx_kl",
        "clipfrac",
        "grad_norm",
        "actor_grad_norm",
        "critic_grad_norm",
        "explained_variance",
        "early_stop",
        "minibatches_attempted",
        "optimizer_steps",
        "optimizer_step_fraction",
        "epochs_completed",
        "nonfinite_grad_count",
    ):
        writer.add_scalar(f"train/{key}", stats.get(key, 0.0), total_steps)

    writer.add_scalar("charts/update", update, total_steps)
    writer.add_scalar("charts/learning_rate", learning_rate, total_steps)
    writer.add_scalar(
        "charts/entropy_coefficient", entropy_coefficient, total_steps
    )
    writer.add_scalar(
        "charts/steps_per_second",
        update_steps / update_seconds,
        total_steps,
    )
    writer.add_scalar(
        "charts/cumulative_steps_per_second",
        cumulative_steps_per_second,
        total_steps,
    )
    writer.add_scalar("time/rollout_seconds", rollout_seconds, total_steps)
    writer.add_scalar("time/ppo_seconds", ppo_seconds, total_steps)
    writer.add_scalar("time/update_seconds", update_seconds, total_steps)
    if grad_scaler is not None and grad_scaler.is_enabled():
        writer.add_scalar("train/amp_scale", grad_scaler.get_scale(), total_steps)

    gib = float(1 << 30)
    writer.add_scalar(
        "system/cuda_memory_allocated_gib",
        torch.cuda.memory_allocated(device) / gib,
        total_steps,
    )
    writer.add_scalar(
        "system/cuda_peak_memory_allocated_gib",
        torch.cuda.max_memory_allocated(device) / gib,
        total_steps,
    )
    writer.add_scalar(
        "system/cuda_memory_reserved_gib",
        torch.cuda.memory_reserved(device) / gib,
        total_steps,
    )

    if opponent_pool is not None:
        assignments = opponent_pool.assignments.reshape(-1)
        pool_metric_tags = [
            "selfplay/frozen_assignments",
            "selfplay/best_assignments",
            "selfplay/hungry_assignments",
            "selfplay/random_assignments",
            "selfplay/greedy_frozen_assignments",
            "selfplay/duel_environments",
            "selfplay/copy_lineup_environments",
        ]
        pool_metric_tensors = [
            (assignments >= 0).sum(),
            (assignments == POOL_BEST).sum(),
            (assignments == POOL_HUNGRY).sum(),
            (assignments == POOL_RANDOM).sum(),
            (
                (assignments >= 0)
                & opponent_pool.deterministic_assignments.reshape(-1)
            ).sum(),
            opponent_pool.duel_rows.sum(),
            opponent_pool.copy_rows.sum(),
        ]
        for label, opponent_id, _ in LOG_DERIVED_HEURISTICS:
            pool_metric_tags.append(f"selfplay/{label}_assignments")
            pool_metric_tensors.append((assignments == opponent_id).sum())
        focus_id = opponent_pool.focus_checkpoint_id()
        if focus_id is not None:
            pool_metric_tags.append("selfplay/focus_assignments")
            pool_metric_tensors.append((assignments == focus_id).sum())
        pinned_metrics: list[tuple[str, int]] = []
        for checkpoint_id in sorted(opponent_pool.pinned_ids):
            label = opponent_pool.checkpoint_labels.get(checkpoint_id)
            if label is None:
                continue
            pinned_metrics.append((label, checkpoint_id))
            pool_metric_tags.append(f"selfplay/anchor_assignments/{label}")
            pool_metric_tensors.append((assignments == checkpoint_id).sum())

        # One device-to-host synchronization covers all assignment counts.
        # Passing CUDA scalars to SummaryWriter one by one previously forced a
        # separate synchronization for the focus and every pinned opponent.
        pool_metric_values = torch.stack(pool_metric_tensors).tolist()
        for tag, value in zip(pool_metric_tags, pool_metric_values):
            writer.add_scalar(tag, value, total_steps)
        writer.add_scalar(
            "selfplay/checkpoint_count",
            opponent_pool.checkpoint_count,
            total_steps,
        )
        writer.add_scalar(
            "selfplay/active_model_count",
            len(opponent_pool.active_ids),
            total_steps,
        )
        writer.add_scalar(
            "selfplay/pinned_anchor_count",
            len(opponent_pool.pinned_ids),
            total_steps,
        )
        for label, checkpoint_id in pinned_metrics:
            writer.add_scalar(
                f"selfplay/anchor_weight/{label}",
                opponent_pool.anchor_sampling_weight(checkpoint_id),
                total_steps,
            )
            if label in opponent_pool.anchor_scores:
                writer.add_scalar(
                    f"selfplay/anchor_score/{label}",
                    opponent_pool.anchor_scores[label],
                    total_steps,
                )
        if opponent_pool.nash_weights:
            writer.add_scalar(
                "selfplay/nash/value", opponent_pool.nash_value, total_steps
            )
            writer.add_scalar(
                "selfplay/nash/matrix_rows",
                len(opponent_pool.nash_rows),
                total_steps,
            )
            for key, weight in opponent_pool.nash_weights.items():
                writer.add_scalar(
                    f"selfplay/nash_weight/{key.replace(':', '/')}",
                    weight,
                    total_steps,
                )
            for key, weight in opponent_pool.effective_training_weights().items():
                writer.add_scalar(
                    f"selfplay/training_weight/{key.replace(':', '/')}",
                    weight,
                    total_steps,
                )


def should_run_evaluation(
    local_update: int,
    completed_updates: int,
    eval_interval: int,
) -> bool:
    """Schedule bootstrap probes only for a genuinely new training run."""

    if eval_interval <= 0:
        return False
    bootstrap_probe = (
        completed_updates == 0 and local_update in (1, 10, 20)
    )
    return bootstrap_probe or local_update % eval_interval == 0


def evaluate_against_history(
    model: ActorCritic,
    opponent_pool: CudaOpponentPool,
    args,
    *,
    update: int,
    total_steps: int,
    writer,
    device: torch.device,
) -> list[dict[str, float | str]]:
    """Evaluate without perturbing the training policy mode or RNG streams."""

    candidates = opponent_pool.evaluation_candidates(args.eval_opponents)
    if not candidates:
        return []
    was_training = model.training
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device)
    results = []
    try:
        evaluation_round = update // max(1, args.eval_interval)
        seed_base = args.eval_seed + evaluation_round * args.eval_seed_stride
        for offset, (kind, label, checkpoint_id, historical_model) in enumerate(
            candidates
        ):
            comparison_seed = seed_base + offset
            comparison_layout = (
                "true-duel"
                if kind == "heuristic" and label == "snake25_duelist"
                else args.eval_layout
            )
            comparison = compare_policies_cuda(
                model,
                historical_model,
                games=args.eval_games,
                num_envs=args.eval_num_envs,
                max_turns=args.eval_max_turns,
                seed=comparison_seed,
                layout=comparison_layout,
                model_b_kind=label if kind == "heuristic" else "model",
                deterministic=True,
                action_mask_mode=args.agent_action_mask,
                device=device,
            )
            opponent_steps = opponent_pool.checkpoint_steps.get(checkpoint_id, 0)
            age_steps = max(0, total_steps - opponent_steps)
            result = {
                "kind": kind,
                "label": label,
                "current_win_rate": comparison["a_win_rate"],
                "opponent_win_rate": comparison["b_win_rate"],
                "draw_rate": comparison["draw_rate"],
                "opponent_steps": float(opponent_steps),
                "opponent_age_steps": float(age_steps),
                "opponent_key": opponent_pool.opponent_key(checkpoint_id),
                "layout": comparison_layout,
            }
            results.append(result)
            if writer is not None:
                prefix = {
                    "anchor": f"eval/anchors/{label}",
                    "history": f"eval/history/{label}",
                    "heuristic": f"eval/heuristics/{label}",
                }[kind]
                for key, value in result.items():
                    if isinstance(value, float):
                        writer.add_scalar(f"{prefix}/{key}", value, total_steps)
            print(
                f"[eval] update={update} steps={total_steps} "
                f"layout={comparison_layout} seed={comparison_seed} "
                f"opponent={kind}:{label} "
                f"opponent_steps={opponent_steps} age_steps={age_steps} "
                f"current_win_rate={comparison['a_win_rate']:.3f} "
                f"opponent_win_rate={comparison['b_win_rate']:.3f} "
                f"draw_rate={comparison['draw_rate']:.3f}",
                flush=True,
            )
        if writer is not None and results:
            anchor_results = [
                result for result in results if result["kind"] == "anchor"
            ]
            history_results = [
                result for result in results if result["kind"] == "history"
            ]
            heuristic_results = [
                result for result in results if result["kind"] == "heuristic"
            ]
            if anchor_results:
                anchor_rates = [
                    float(result["current_win_rate"])
                    for result in anchor_results
                ]
                writer.add_scalar(
                    "eval/anchors/mean_current_win_rate",
                    sum(anchor_rates) / len(anchor_rates),
                    total_steps,
                )
                writer.add_scalar(
                    "eval/anchors/worst_current_win_rate",
                    min(anchor_rates),
                    total_steps,
                )
                writer.add_scalar(
                    "eval/anchors/selection_score",
                    min(anchor_rates),
                    total_steps,
                )
            if history_results:
                history_rates = [
                    float(result["current_win_rate"])
                    for result in history_results
                ]
                writer.add_scalar(
                    "eval/history/mean_current_win_rate",
                    sum(history_rates) / len(history_rates),
                    total_steps,
                )
                writer.add_scalar(
                    "eval/history/worst_current_win_rate",
                    min(history_rates),
                    total_steps,
                )
            if heuristic_results:
                heuristic_rates = [
                    float(result["current_win_rate"])
                    for result in heuristic_results
                ]
                writer.add_scalar(
                    "eval/heuristics/mean_current_win_rate",
                    sum(heuristic_rates) / len(heuristic_rates),
                    total_steps,
                )
                writer.add_scalar(
                    "eval/heuristics/worst_current_win_rate",
                    min(heuristic_rates),
                    total_steps,
                )
            writer.add_scalar(
                "eval/worst_current_win_rate",
                min(float(result["current_win_rate"]) for result in results),
                total_steps,
            )
        fixed_neural_results = [
            result
            for result in results
            if result["kind"] == "anchor"
            and result["label"] != LEAGUE_CHAMPION_LABEL
        ]
        heuristic_results = [
            result for result in results if result["kind"] == "heuristic"
        ]

        def match_scores(items):
            return [
                float(item["current_win_rate"])
                + 0.5 * float(item["draw_rate"])
                for item in items
            ]

        neural_scores = match_scores(fixed_neural_results)
        heuristic_scores = match_scores(heuristic_results)
        if neural_scores:
            summary = (
                f"[eval-summary] update={update} fixed_neural_mean="
                f"{sum(neural_scores) / len(neural_scores):.3f} "
                f"fixed_neural_worst={min(neural_scores):.3f}"
            )
            if heuristic_scores:
                summary += (
                    f" heuristic_mean="
                    f"{sum(heuristic_scores) / len(heuristic_scores):.3f} "
                    f"heuristic_worst={min(heuristic_scores):.3f}"
                )
            print(summary, flush=True)
            if writer is not None:
                writer.add_scalar(
                    "eval/general/fixed_neural_mean_match_score",
                    sum(neural_scores) / len(neural_scores),
                    total_steps,
                )
                writer.add_scalar(
                    "eval/general/fixed_neural_worst_match_score",
                    min(neural_scores),
                    total_steps,
                )
                if heuristic_scores:
                    writer.add_scalar(
                        "eval/general/heuristic_mean_match_score",
                        sum(heuristic_scores) / len(heuristic_scores),
                        total_steps,
                    )
                    writer.add_scalar(
                        "eval/general/heuristic_worst_match_score",
                        min(heuristic_scores),
                        total_steps,
                    )
    finally:
        model.train(was_training)
        torch.random.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, device)
    return results


def evaluate_champion_anchor_baseline(
    champion: ActorCritic,
    opponent_pool: CudaOpponentPool,
    args,
    *,
    update: int,
    total_steps: int,
    writer,
    device: torch.device,
) -> dict[str, float]:
    """Measure the champion once on the candidate's fixed neural seed suite."""

    pinned = sorted(
        (
            opponent_pool.checkpoint_labels.get(
                checkpoint_id, f"anchor_{checkpoint_id}"
            ),
            checkpoint_id,
            opponent_pool.models[checkpoint_id],
        )
        for checkpoint_id in opponent_pool.selectable_ids
        if checkpoint_id in opponent_pool.pinned_ids
    )
    was_training = champion.training
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device)
    scores: dict[str, float] = {}
    try:
        evaluation_round = update // max(1, args.eval_interval)
        seed_base = args.eval_seed + evaluation_round * args.eval_seed_stride
        for offset, (label, _, historical_model) in enumerate(pinned):
            if label == LEAGUE_CHAMPION_LABEL:
                continue
            comparison_seed = seed_base + offset
            comparison = compare_policies_cuda(
                champion,
                historical_model,
                games=args.eval_games,
                num_envs=args.eval_num_envs,
                max_turns=args.eval_max_turns,
                seed=comparison_seed,
                layout=args.eval_layout,
                deterministic=True,
                action_mask_mode=args.agent_action_mask,
                device=device,
            )
            score = comparison_score(comparison)
            scores[label] = score
            print(
                f"[champion-benchmark] update={update} steps={total_steps} "
                f"layout={args.eval_layout} seed={comparison_seed} "
                f"opponent=anchor:{label} score={score:.3f}",
                flush=True,
            )
            if writer is not None:
                writer.add_scalar(
                    f"league/champion_anchor_baseline/{label}",
                    score,
                    total_steps,
                )
    finally:
        champion.train(was_training)
        torch.random.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, device)
    if not scores:
        raise RuntimeError("champion fixed-anchor benchmark has no opponents")
    print(
        f"[champion-benchmark] fixed_neural_mean="
        f"{sum(scores.values()) / len(scores):.3f} "
        f"fixed_neural_worst={min(scores.values()):.3f}",
        flush=True,
    )
    return scores


def evaluate_league_candidate(
    candidate: ActorCritic,
    champion: ActorCritic,
    args,
    *,
    update: int,
    total_steps: int,
    device: torch.device,
) -> list[dict[str, float]]:
    """Run the multi-seed champion match without changing training RNG state."""

    candidate_was_training = candidate.training
    champion_was_training = champion.training
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device)
    comparisons: list[dict[str, float]] = []
    try:
        evaluation_round = update // max(1, args.eval_interval)
        seed_offset = evaluation_round * args.league_promotion_seed_stride
        for layout in args.league_promotion_layouts:
            for seed in args.league_promotion_seeds:
                effective_seed = seed + seed_offset
                comparison = compare_policies_cuda(
                    candidate,
                    champion,
                    games=args.league_promotion_games,
                    num_envs=args.eval_num_envs,
                    max_turns=args.eval_max_turns,
                    seed=effective_seed,
                    layout=layout,
                    deterministic=True,
                    action_mask_mode=args.agent_action_mask,
                    device=device,
                )
                comparisons.append(comparison)
                print(
                    f"[league] update={update} steps={total_steps} "
                    f"layout={layout} seed={effective_seed} "
                    f"candidate_score={comparison_score(comparison):.3f} "
                    f"candidate_win_rate={comparison['a_win_rate']:.3f} "
                    f"champion_win_rate={comparison['b_win_rate']:.3f} "
                    f"draw_rate={comparison['draw_rate']:.3f}",
                    flush=True,
                )
    finally:
        candidate.train(candidate_was_training)
        champion.train(champion_was_training)
        torch.random.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, device)
    return comparisons


def main():
    parser = argparse.ArgumentParser(
        description="CUDA recurrent PPO with episode-stable pool self-play"
    )
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument(
        "--stop-after-update",
        type=int,
        default=None,
        help="absolute update at which to stop; useful for interruption-safe resumes",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--rollout-obs-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="mixed precision for CUDA inference and PPO updates; float32 disables AMP",
    )
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("cosine", "cosine-restarts", "constant"),
        default="cosine",
        help="monotonic cosine is safe for long runs; cosine-restarts is legacy",
    )
    parser.add_argument("--lr-floor", type=float, default=1e-5)
    parser.add_argument(
        "--lr-decay-updates",
        type=int,
        default=None,
        help=(
            "updates over which cosine reaches --lr-floor; defaults to the "
            "full run and remains at the floor afterwards"
        ),
    )
    parser.add_argument(
        "--continuation-lr",
        type=float,
        default=None,
        help=(
            "explicit LR at resume/warm-start; by default the saved LR is reused "
            "and never raised"
        ),
    )
    parser.add_argument(
        "--schedule-origin-update",
        type=int,
        default=None,
        help=(
            "absolute source update that counts as phase-local update zero; "
            "use together with explicit LR/entropy decay lengths when a new "
            "phase starts from an existing checkpoint"
        ),
    )
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument(
        "--kl-stop-mode",
        choices=("minibatch", "epoch"),
        default="minibatch",
        help=(
            "minibatch is the quality/stability default; epoch reduces CUDA "
            "synchronizations but reacts later to destructive policy updates"
        ),
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument(
        "--clip-range-vf",
        type=float,
        default=None,
        help="optional value clipping; default matches SB3/train.py (disabled)",
    )
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument(
        "--ent-coef-final",
        type=float,
        default=None,
        help="final entropy coefficient for cosine decay; omit to keep it constant",
    )
    parser.add_argument(
        "--ent-decay-updates",
        type=int,
        default=None,
        help="updates over which --ent-coef decays to --ent-coef-final",
    )
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--actor-max-grad-norm",
        type=float,
        default=None,
        help=(
            "optional actor-only gradient clip; enables independent actor/critic "
            "clipping and otherwise falls back to --max-grad-norm"
        ),
    )
    parser.add_argument(
        "--critic-max-grad-norm",
        type=float,
        default=None,
        help=(
            "optional critic-only gradient clip; enables independent actor/critic "
            "clipping and otherwise falls back to --max-grad-norm"
        ),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save-path", default="ppo_bs_lstm_cuda.pt")
    parser.add_argument("--sb3-save-path", default="ppo_bs_lstm_cuda.zip")
    parser.add_argument(
        "--sb3-checkpoint-interval",
        type=int,
        default=250,
        help="updates between intermediate ppo.py-compatible ZIP exports; 0 disables",
    )
    parser.add_argument(
        "--sb3-checkpoint-dir",
        default="models/ppo_bs_lstm_cuda_variants",
    )
    parser.add_argument(
        "--sb3-checkpoint-prefix",
        default=None,
        help="filename prefix; defaults to the stem of --sb3-save-path",
    )
    parser.add_argument(
        "--max-sb3-checkpoints",
        type=int,
        default=0,
        help=(
            "maximum periodic ZIP/PT checkpoint pairs to retain; "
            "0 keeps every export"
        ),
    )
    parser.add_argument("--resume-path", default=None)
    parser.add_argument(
        "--warm-start-path",
        default=None,
        help=(
            "initialize policy weights from a Torch .pt or SB3 .zip model; "
            "optimizer and AMP scaler start fresh, while counters and a safe LR are preserved"
        ),
    )
    parser.add_argument(
        "--actor-reference-path",
        default=None,
        help=(
            "immutable Torch/SB3 policy used for actor-only L2-SP "
            "regularization; requires --actor-reference-l2-coef"
        ),
    )
    parser.add_argument(
        "--actor-reference-l2-coef",
        type=float,
        default=0.0,
        help=(
            "strength of the cumulative actor-drift penalty around "
            "--actor-reference-path; 0 disables it"
        ),
    )
    parser.add_argument(
        "--critic-warmup-updates",
        type=int,
        default=0,
        help=(
            "freeze the actor (cnn, actor_lstm, actor head) for the first N "
            "updates after a warm start so the privileged critic can adapt to "
            "its zero-initialised unfogged channels first; 0 disables"
        ),
    )
    parser.add_argument("--potential-length-coef", type=float, default=0.05)
    parser.add_argument("--potential-health-coef", type=float, default=0.02)
    parser.add_argument(
        "--reward-scheme",
        choices=("kill", "tournament"),
        default="kill",
        help=(
            "kill keeps the legacy incremental rank reward; tournament gives "
            "2/1/0/0 points for first/second/third/fourth"
        ),
    )
    parser.add_argument(
        "--potential-mobility-coef",
        type=float,
        default=0.0,
        help="potential coefficient for observable safe-move count",
    )
    parser.add_argument(
        "--opponent-mode",
        choices=("curriculum", "selfplay", "random"),
        default="curriculum",
    )
    parser.add_argument(
        "--training-duel-probability",
        type=float,
        default=0.0,
        help=(
            "fraction of self-play episodes that start as true two-snake "
            "duels; the remainder keep all four snakes"
        ),
    )
    parser.add_argument(
        "--selfplay-update-interval",
        type=int,
        default=10,
        help="updates between immutable opponent snapshots",
    )
    parser.add_argument(
        "--selfplay-start-update",
        type=int,
        default=20,
        help="curriculum warm-up before frozen checkpoints enter the pool",
    )
    parser.add_argument("--pool-dir", default="models/selfplay_pool_cuda_v2")
    parser.add_argument(
        "--pool-anchor",
        dest="pool_anchors",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "permanent external Torch/SB3 opponent; repeat for multiple named "
            "anchors (each gets its own eval/anchors/LABEL TensorBoard series)"
        ),
    )
    parser.add_argument("--max-pool-checkpoints", type=int, default=50)
    parser.add_argument(
        "--pool-checkpoint-weight",
        type=float,
        default=0.65,
        help="free Nash-distributed mass once a payoff matrix is available",
    )
    parser.add_argument(
        "--pool-best-weight",
        type=float,
        default=0.15,
        help="Best-agent category weight and guaranteed floor under Nash",
    )
    parser.add_argument(
        "--pool-hungry-weight",
        type=float,
        default=0.15,
        help="Hungry-agent category weight and guaranteed floor under Nash",
    )
    parser.add_argument(
        "--pool-random-weight",
        type=float,
        default=0.05,
        help="Random-agent category weight and guaranteed floor under Nash",
    )
    parser.add_argument(
        "--pool-real-heuristic-weight",
        type=float,
        default=0.0,
        help=(
            "combined guaranteed weight split evenly over all native CUDA "
            "profiles; mutually exclusive with --pool-heuristic-weight"
        ),
    )
    parser.add_argument(
        "--pool-heuristic-weight",
        dest="pool_heuristic_weights",
        action="append",
        default=[],
        metavar="LABEL=WEIGHT",
        help=(
            "exact guaranteed weight for one native CUDA profile; repeat for "
            "the desired profiles, with unspecified profiles disabled"
        ),
    )
    parser.add_argument("--pool-latest-probability", type=float, default=0.5)
    parser.add_argument(
        "--pool-anchor-probability",
        type=float,
        default=0.25,
        help="within neural-opponent samples, probability of a pinned source policy",
    )
    parser.add_argument(
        "--pool-anchor-target",
        dest="pool_anchor_targets",
        action="append",
        default=[],
        metavar="LABEL=SCORE",
        help=(
            "desired match score against a fixed anchor; evaluated anchors below "
            "their target are sampled more often (repeat per anchor)"
        ),
    )
    parser.add_argument(
        "--pool-anchor-priority-exponent",
        type=float,
        default=2.0,
        help="PFSP emphasis on anchors furthest below their target",
    )
    parser.add_argument(
        "--pool-anchor-min-weight",
        type=float,
        default=0.05,
        help="non-zero PFSP weight retained after an anchor reaches its target",
    )
    parser.add_argument(
        "--pool-anchor-score-ema",
        type=float,
        default=0.5,
        help="new-evaluation fraction in smoothed anchor match scores",
    )
    parser.add_argument(
        "--pool-anchor-uniform-floor",
        type=float,
        default=0.0,
        help=(
            "total normalized training mass reserved equally for active fixed "
            "anchors except the league champion and dedicated duel policies; "
            "taken from checkpoint Nash mass"
        ),
    )
    parser.add_argument(
        "--pool-nash-history-size",
        type=int,
        default=16,
        help="learner evaluation rows retained in the empirical payoff matrix",
    )
    parser.add_argument(
        "--pool-nash-iterations",
        type=int,
        default=2000,
        help="multiplicative-weights iterations used to solve the Nash mixture",
    )
    parser.add_argument(
        "--pool-nash-exploration",
        type=float,
        default=0.05,
        help="uniform mass mixed into the adversarial Nash opponent strategy",
    )
    parser.add_argument(
        "--pool-nash-score-half-life-updates",
        type=int,
        default=0,
        help=(
            "decay stale payoff scores toward 0.5 over this many updates; "
            "0 preserves scores indefinitely"
        ),
    )
    parser.add_argument(
        "--pool-champion-min-weight",
        type=float,
        default=0.0,
        help=(
            "guaranteed total sampling mass for the current league champion; "
            "taken from the free Nash checkpoint share"
        ),
    )
    parser.add_argument(
        "--pool-deduplicate-champion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "exclude a fixed anchor from training while its provenance is "
            "identical to the current league champion; evaluation and later "
            "retention remain enabled"
        ),
    )
    parser.add_argument(
        "--pool-min-action-disagreement",
        type=float,
        default=0.0,
        help=(
            "minimum greedy-action disagreement from every population member; "
            "similar self-play snapshots are discarded"
        ),
    )
    parser.add_argument(
        "--pool-diversity-probe-size",
        type=int,
        default=512,
        help="rollout observations used for behavioral snapshot deduplication",
    )
    parser.add_argument(
        "--pool-focus-label",
        default=None,
        help="pinned checkpoint label sampled especially often",
    )
    parser.add_argument(
        "--pool-focus-probability",
        type=float,
        default=0.0,
        help="probability that a neural-opponent draw uses --pool-focus-label",
    )
    parser.add_argument(
        "--pool-deterministic-probability",
        type=float,
        default=0.0,
        help="fraction of frozen opponents acting greedily for a full episode",
    )
    parser.add_argument(
        "--pool-copies-probability",
        type=float,
        default=0.0,
        help=(
            "fraction of four-player episodes where one sampled opponent "
            "policy fills all three opponent seats; this aligns Nash/PFSP "
            "sampling with homogeneous policy evaluation"
        ),
    )
    parser.add_argument(
        "--pool-duel-opponent-label",
        default=None,
        help=(
            "native heuristic forced into the live opponent seat in a fraction "
            "of true duel episodes"
        ),
    )
    parser.add_argument(
        "--pool-duel-opponent-probability",
        type=float,
        default=0.0,
        help=(
            "conditional probability of using --pool-duel-opponent-label in "
            "a true duel"
        ),
    )
    parser.add_argument(
        "--pool-duel-opponent-weight",
        dest="pool_duel_opponent_weights",
        action="append",
        default=[],
        metavar="LABEL=PROBABILITY",
        help=(
            "conditional true-duel probability reserved for one enabled native "
            "profile, fixed neural anchor, or league champion; repeatable and "
            "must sum to <= 1"
        ),
    )
    parser.add_argument(
        "--pool-active-checkpoints",
        type=int,
        default=2,
        help=(
            "maximum neural opponents in the current rollout working set; "
            "fixed anchors rotate while mandatory opponents remain active"
        ),
    )
    parser.add_argument(
        "--pool-active-rotation-interval",
        type=int,
        default=1,
        help=(
            "updates between neural working-set rotations; larger values avoid "
            "temporarily evaluating both old and new sets in ongoing episodes"
        ),
    )
    parser.add_argument(
        "--pool-parallel-inference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "run independent frozen opponent policies on separate CUDA streams; "
            "defaults to on for GPUs with at least 100 SMs"
        ),
    )
    parser.add_argument(
        "--pool-load-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=250,
        help="updates between fixed historical-policy evaluations; 0 disables",
    )
    parser.add_argument("--eval-games", type=int, default=256)
    parser.add_argument("--eval-num-envs", type=int, default=128)
    parser.add_argument("--eval-max-turns", type=int, default=1000)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=123,
        help="fixed comparison seed so checkpoint scores use paired games",
    )
    parser.add_argument(
        "--eval-seed-stride",
        type=int,
        default=0,
        help=(
            "per-evaluation-round seed increment; 0 keeps the historical fixed "
            "paired suite"
        ),
    )
    parser.add_argument(
        "--eval-layout",
        choices=("copies", "duel", "solo-pair", "true-duel"),
        default="copies",
        help="seat layout for anchor, history, and heuristic evaluations",
    )
    parser.add_argument(
        "--eval-opponents",
        type=int,
        default=4,
        help=(
            "history policies per evaluation: latest, strongest Nash support, "
            "and least-recently measured population members"
        ),
    )
    parser.add_argument(
        "--league-champion-path",
        default=None,
        help=(
            "stable Torch path for the promoted champion; enables champion-gated "
            "league training"
        ),
    )
    parser.add_argument(
        "--league-champion-zip-path",
        default=None,
        help=(
            "ppo.py-compatible champion export; defaults to the Torch champion "
            "path with a .zip suffix"
        ),
    )
    parser.add_argument(
        "--league-initial-champion-path",
        default=None,
        help="initial Torch/SB3 champion used when the stable champion does not exist",
    )
    parser.add_argument(
        "--league-promotion-games",
        type=int,
        default=768,
        help="paired candidate-vs-champion games per promotion seed",
    )
    parser.add_argument(
        "--league-promotion-seed",
        dest="league_promotion_seeds",
        type=int,
        action="append",
        default=[],
        help="comparison seed for champion promotion; repeat to aggregate seeds",
    )
    parser.add_argument(
        "--league-promotion-seed-stride",
        type=int,
        default=0,
        help="per-promotion-round increment applied to every configured seed",
    )
    parser.add_argument(
        "--league-promotion-layout",
        dest="league_promotion_layouts",
        choices=("copies", "solo-pair", "true-duel"),
        action="append",
        default=[],
        help=(
            "symmetric candidate/champion seat layout; repeat to aggregate "
            "2-vs-2, deployment-like 1-vs-3/3-vs-1, and true 1-vs-1 games"
        ),
    )
    parser.add_argument(
        "--league-promotion-threshold",
        type=float,
        default=0.525,
        help="minimum candidate match score (wins + half draws) for promotion",
    )
    parser.add_argument(
        "--league-general-improvement",
        type=float,
        default=None,
        help=(
            "required mean candidate-minus-champion match-score improvement "
            "over the same fixed neural anchors; unset disables the relative gate"
        ),
    )
    parser.add_argument(
        "--league-max-anchor-regression",
        type=float,
        default=None,
        help=(
            "largest allowed candidate regression against any fixed neural "
            "anchor relative to the champion; unset disables the relative gate"
        ),
    )
    parser.add_argument(
        "--league-min-promotion-interval",
        type=int,
        default=250,
        help="minimum updates between champion replacements",
    )
    parser.add_argument(
        "--league-guard-label",
        dest="league_guard_labels",
        action="append",
        default=[],
        help="fixed pool anchor that a promoted candidate must continue to beat",
    )
    parser.add_argument(
        "--league-min-guard-score",
        type=float,
        default=0.75,
        help="minimum match score against every promotion guard anchor",
    )
    parser.add_argument(
        "--league-guard-score",
        dest="league_guard_scores",
        action="append",
        default=[],
        metavar="LABEL=SCORE",
        help=(
            "per-anchor or native-heuristic promotion floor; also makes LABEL "
            "a guard and overrides --league-min-guard-score for it"
        ),
    )
    parser.add_argument(
        "--fused-adam",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="cuDNN autotuning (usually slower here because episode batches vary)",
    )
    parser.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use NHWC memory layout for faster Tensor-Core convolutions",
    )
    parser.add_argument(
        "--compile-training-cnn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="compile the learner CNN (startup cost, beneficial for long runs)",
    )
    parser.add_argument(
        "--compile-rollout-ops",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "fuse the fixed-shape learner action mask during rollouts; useful "
            "on launch-bound high-end CUDA GPUs"
        ),
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead"),
        default="default",
        help=(
            "default avoids private CUDA-Graph pools and is safer on 8 GiB GPUs; "
            "reduce-overhead uses more VRAM"
        ),
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        default=None,
        help="event-file directory; omit or pass an empty string to disable",
    )
    parser.add_argument("--tensorboard-flush-secs", type=int, default=30)
    parser.add_argument("--tensorboard-max-queue", type=int, default=10)
    parser.add_argument(
        "--tensorboard-update-interval",
        type=int,
        default=1,
        help="write TensorBoard metrics every N updates",
    )
    parser.add_argument(
        "--timing-log-interval",
        type=int,
        default=10,
        help=(
            "updates between wall-clock phase diagnostics; 0 disables. "
            "The RTX 5090 launcher uses 1 to expose recurring host stalls."
        ),
    )
    parser.add_argument(
        "--agent-action-mask",
        choices=("observable", "observable-hard", "none", "server"),
        default="observable",
        help=(
            "observable is the conservative legacy mask; observable-hard lets "
            "PPO choose among risky head contests; server leaks hidden geometry"
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if args.batch_size < args.seq_len:
        raise ValueError("--batch-size must be at least --seq-len")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive")
    if args.actor_max_grad_norm is not None and args.actor_max_grad_norm <= 0.0:
        raise ValueError("--actor-max-grad-norm must be positive")
    if args.critic_max_grad_norm is not None and args.critic_max_grad_norm <= 0.0:
        raise ValueError("--critic-max-grad-norm must be positive")
    if args.stop_after_update is not None and args.stop_after_update <= 0:
        raise ValueError("--stop-after-update must be positive")
    if args.schedule_origin_update is not None and args.schedule_origin_update < 0:
        raise ValueError("--schedule-origin-update must be non-negative")
    if args.lr_decay_updates is not None and args.lr_decay_updates <= 0:
        raise ValueError("--lr-decay-updates must be positive")
    if args.ent_coef < 0.0:
        raise ValueError("--ent-coef must be non-negative")
    if args.ent_coef_final is not None and args.ent_coef_final < 0.0:
        raise ValueError("--ent-coef-final must be non-negative")
    if args.ent_coef_final is not None and args.ent_coef_final > args.ent_coef:
        raise ValueError("--ent-coef-final cannot exceed --ent-coef")
    if args.ent_decay_updates is not None and args.ent_decay_updates <= 0:
        raise ValueError("--ent-decay-updates must be positive")
    if args.potential_mobility_coef < 0.0:
        raise ValueError("--potential-mobility-coef must be non-negative")
    if not 0.0 <= args.training_duel_probability <= 1.0:
        raise ValueError("--training-duel-probability must be in [0, 1]")
    if (
        args.training_duel_probability > 0.0
        and args.opponent_mode not in ("curriculum", "selfplay")
    ):
        raise ValueError(
            "--training-duel-probability requires curriculum or selfplay opponents"
        )
    if args.pool_real_heuristic_weight < 0.0:
        raise ValueError("--pool-real-heuristic-weight must be non-negative")
    args.pool_heuristic_weight_map = parse_heuristic_weights(
        args.pool_heuristic_weights
    )
    if args.pool_real_heuristic_weight and args.pool_heuristic_weight_map:
        raise ValueError(
            "--pool-real-heuristic-weight and --pool-heuristic-weight are "
            "mutually exclusive"
        )
    if not 0.0 <= args.pool_anchor_probability <= 1.0:
        raise ValueError("--pool-anchor-probability must be in [0, 1]")
    if args.pool_anchor_priority_exponent <= 0.0:
        raise ValueError("--pool-anchor-priority-exponent must be positive")
    if not 0.0 < args.pool_anchor_min_weight <= 1.0:
        raise ValueError("--pool-anchor-min-weight must be in (0, 1]")
    if not 0.0 < args.pool_anchor_score_ema <= 1.0:
        raise ValueError("--pool-anchor-score-ema must be in (0, 1]")
    if args.pool_anchor_uniform_floor < 0.0:
        raise ValueError("--pool-anchor-uniform-floor must be non-negative")
    if args.pool_active_rotation_interval <= 0:
        raise ValueError("--pool-active-rotation-interval must be positive")
    if args.pool_nash_history_size <= 0:
        raise ValueError("--pool-nash-history-size must be positive")
    if args.pool_nash_iterations <= 0:
        raise ValueError("--pool-nash-iterations must be positive")
    if not 0.0 <= args.pool_nash_exploration < 1.0:
        raise ValueError("--pool-nash-exploration must be in [0, 1)")
    if args.pool_nash_score_half_life_updates < 0:
        raise ValueError("--pool-nash-score-half-life-updates must be non-negative")
    if args.pool_champion_min_weight < 0.0:
        raise ValueError("--pool-champion-min-weight must be non-negative")
    if not 0.0 <= args.pool_min_action_disagreement <= 1.0:
        raise ValueError("--pool-min-action-disagreement must be in [0, 1]")
    if args.pool_diversity_probe_size <= 0:
        raise ValueError("--pool-diversity-probe-size must be positive")
    if not 0.0 <= args.pool_focus_probability <= 1.0:
        raise ValueError("--pool-focus-probability must be in [0, 1]")
    if not 0.0 <= args.pool_deterministic_probability <= 1.0:
        raise ValueError("--pool-deterministic-probability must be in [0, 1]")
    if not 0.0 <= args.pool_copies_probability <= 1.0:
        raise ValueError("--pool-copies-probability must be in [0, 1]")
    if not 0.0 <= args.pool_duel_opponent_probability <= 1.0:
        raise ValueError("--pool-duel-opponent-probability must be in [0, 1]")
    if args.pool_duel_opponent_probability and not args.pool_duel_opponent_label:
        raise ValueError(
            "--pool-duel-opponent-probability requires --pool-duel-opponent-label"
        )
    args.pool_duel_opponent_weight_map = parse_labeled_scores(
        args.pool_duel_opponent_weights,
        option_name="pool duel opponent weight",
    )
    if args.pool_duel_opponent_weight_map and (
        args.pool_duel_opponent_label is not None
        or args.pool_duel_opponent_probability != 0.0
    ):
        raise ValueError(
            "--pool-duel-opponent-weight is mutually exclusive with the "
            "singular duel opponent options"
        )
    if sum(args.pool_duel_opponent_weight_map.values()) > 1.0 + 1e-9:
        raise ValueError("--pool-duel-opponent-weight values must sum to <= 1")
    if args.selfplay_update_interval <= 0:
        raise ValueError("--selfplay-update-interval must be positive")
    if args.eval_interval < 0:
        raise ValueError("--eval-interval must be non-negative")
    if args.eval_seed_stride < 0:
        raise ValueError("--eval-seed-stride must be non-negative")
    if args.league_promotion_seed_stride < 0:
        raise ValueError("--league-promotion-seed-stride must be non-negative")
    if args.sb3_checkpoint_interval < 0:
        raise ValueError("--sb3-checkpoint-interval must be non-negative")
    if args.max_sb3_checkpoints < 0:
        raise ValueError("--max-sb3-checkpoints must be non-negative")
    if args.eval_interval and (
        args.eval_games <= 0
        or args.eval_num_envs <= 0
        or args.eval_max_turns <= 0
        or args.eval_opponents < 0
    ):
        raise ValueError(
            "evaluation sizes must be positive and --eval-opponents non-negative "
            "when evaluation is enabled"
        )
    if args.league_champion_path:
        if args.opponent_mode not in ("curriculum", "selfplay"):
            raise ValueError("league training requires a self-play opponent pool")
        if not args.eval_interval:
            raise ValueError("league training requires --eval-interval > 0")
        if args.league_promotion_games <= 0:
            raise ValueError("--league-promotion-games must be positive")
        if args.league_min_promotion_interval <= 0:
            raise ValueError("--league-min-promotion-interval must be positive")
        if not 0.5 <= args.league_promotion_threshold <= 1.0:
            raise ValueError("--league-promotion-threshold must be in [0.5, 1]")
        relative_gate_values = (
            args.league_general_improvement,
            args.league_max_anchor_regression,
        )
        if any(value is not None for value in relative_gate_values) and not all(
            value is not None for value in relative_gate_values
        ):
            raise ValueError(
                "--league-general-improvement and "
                "--league-max-anchor-regression must be configured together"
            )
        if (
            args.league_general_improvement is not None
            and args.league_general_improvement < 0.0
        ):
            raise ValueError("--league-general-improvement must be non-negative")
        if (
            args.league_max_anchor_regression is not None
            and args.league_max_anchor_regression < 0.0
        ):
            raise ValueError("--league-max-anchor-regression must be non-negative")
        if (
            args.league_general_improvement is not None
            and args.eval_seed_stride != 0
        ):
            raise ValueError(
                "the relative league generalization gate requires "
                "--eval-seed-stride 0"
            )
        if not 0.0 <= args.league_min_guard_score <= 1.0:
            raise ValueError("--league-min-guard-score must be in [0, 1]")
        if len(set(args.league_guard_labels)) != len(args.league_guard_labels):
            raise ValueError("--league-guard-label values must be unique")
        if not args.league_promotion_seeds:
            args.league_promotion_seeds = [args.eval_seed]
        if not args.league_promotion_layouts:
            args.league_promotion_layouts = ["copies"]
        champion_path = Path(args.league_champion_path).expanduser()
        args.league_champion_path = str(champion_path)
        if args.league_champion_zip_path is None:
            args.league_champion_zip_path = str(champion_path.with_suffix(".zip"))
        else:
            args.league_champion_zip_path = str(
                Path(args.league_champion_zip_path).expanduser()
            )
        if args.league_initial_champion_path:
            initial_champion = Path(args.league_initial_champion_path).expanduser()
            if not initial_champion.is_file():
                raise FileNotFoundError(
                    f"--league-initial-champion-path does not exist: {initial_champion}"
                )
            args.league_initial_champion_path = str(initial_champion)
    elif (
        args.league_initial_champion_path
        or args.league_champion_zip_path
        or args.league_guard_labels
        or args.league_guard_scores
        or args.league_general_improvement is not None
        or args.league_max_anchor_regression is not None
        or args.pool_champion_min_weight
    ):
        raise ValueError(
            "--league-champion-path is required for the other league options"
        )
    if args.tensorboard_flush_secs <= 0:
        raise ValueError("--tensorboard-flush-secs must be positive")
    if args.tensorboard_max_queue <= 0:
        raise ValueError("--tensorboard-max-queue must be positive")
    if args.tensorboard_update_interval <= 0:
        raise ValueError("--tensorboard-update-interval must be positive")
    if args.timing_log_interval < 0:
        raise ValueError("--timing-log-interval must be non-negative")
    if args.resume_path and args.warm_start_path:
        raise ValueError("--resume-path and --warm-start-path are mutually exclusive")
    if (
        not math.isfinite(args.actor_reference_l2_coef)
        or args.actor_reference_l2_coef < 0.0
    ):
        raise ValueError("--actor-reference-l2-coef must be finite and non-negative")
    if args.actor_reference_l2_coef > 0.0 and not args.actor_reference_path:
        raise ValueError(
            "--actor-reference-path is required when actor reference L2 is enabled"
        )
    if args.actor_reference_path and args.actor_reference_l2_coef == 0.0:
        raise ValueError(
            "--actor-reference-l2-coef must be positive when an actor reference is set"
        )
    anchor_specs = parse_anchor_specs(args.pool_anchors)
    anchor_labels = {label for label, _ in anchor_specs}
    args.pool_anchor_target_map = parse_labeled_scores(
        args.pool_anchor_targets,
        option_name="pool anchor target",
    )
    args.league_guard_score_map = parse_labeled_scores(
        args.league_guard_scores,
        option_name="league guard",
    )
    if LEAGUE_CHAMPION_LABEL in anchor_labels:
        raise ValueError(f"pool anchor label {LEAGUE_CHAMPION_LABEL!r} is reserved")
    unknown_targets = sorted(set(args.pool_anchor_target_map) - anchor_labels)
    if unknown_targets:
        raise ValueError(
            "pool anchor targets must name fixed pool anchors: "
            + ", ".join(unknown_targets)
        )
    valid_focus_labels = set(anchor_labels)
    if args.league_champion_path:
        valid_focus_labels.add(LEAGUE_CHAMPION_LABEL)
    if args.pool_focus_label and args.pool_focus_label not in valid_focus_labels:
        raise ValueError(
            "--pool-focus-label must name a fixed pool anchor or "
            f"{LEAGUE_CHAMPION_LABEL!r} when league training is enabled"
        )
    all_guard_labels = list(
        dict.fromkeys(
            [*args.league_guard_labels, *args.league_guard_score_map]
        )
    )
    args.league_guard_labels = all_guard_labels
    enabled_heuristic_guards = {"best", "hungry", "random"}
    if args.pool_real_heuristic_weight > 0.0:
        enabled_heuristic_guards.update(NATIVE_PROFILE_BY_LABEL)
    else:
        enabled_heuristic_guards.update(
            label
            for label, weight in args.pool_heuristic_weight_map.items()
            if weight > 0.0
        )
    valid_duel_opponent_labels = anchor_labels | (
        enabled_heuristic_guards - {"best", "hungry", "random"}
    )
    if args.league_champion_path:
        valid_duel_opponent_labels.add(LEAGUE_CHAMPION_LABEL)
    unknown_duel_opponents = sorted(
        set(args.pool_duel_opponent_weight_map) - valid_duel_opponent_labels
    )
    if unknown_duel_opponents:
        raise ValueError(
            "duel opponent weights must name fixed anchors, the configured "
            "league champion, or enabled native heuristics: "
            + ", ".join(unknown_duel_opponents)
        )
    valid_guard_labels = anchor_labels | enabled_heuristic_guards
    missing_guards = [
        label for label in all_guard_labels if label not in valid_guard_labels
    ]
    if missing_guards:
        raise ValueError(
            "league guard labels must name fixed pool anchors or enabled "
            "scripted opponents: "
            + ", ".join(missing_guards)
        )
    pinned_count = len(anchor_specs) + int(bool(args.league_champion_path))
    if pinned_count >= args.max_pool_checkpoints:
        raise ValueError(
            "--max-pool-checkpoints must leave room for at least one self-play snapshot"
        )
    for path_attribute in (
        "resume_path",
        "warm_start_path",
        "actor_reference_path",
    ):
        raw_path = getattr(args, path_attribute)
        if raw_path:
            checkpoint_path = Path(raw_path).expanduser()
            if not checkpoint_path.is_file():
                option = "--" + path_attribute.replace("_", "-")
                raise FileNotFoundError(f"{option} does not exist: {checkpoint_path}")
            setattr(args, path_attribute, str(checkpoint_path))
    if not args.sb3_checkpoint_prefix:
        args.sb3_checkpoint_prefix = Path(
            args.sb3_save_path or args.save_path
        ).stem
    device = torch.device("cuda")
    device_properties = torch.cuda.get_device_properties(device)
    if args.pool_parallel_inference is None:
        args.pool_parallel_inference = (
            device_properties.multi_processor_count >= 100
        )
    print(
        f"[cuda] device={device_properties.name} "
        f"cc={device_properties.major}.{device_properties.minor} "
        f"sms={device_properties.multi_processor_count} "
        f"vram={device_properties.total_memory / (1 << 30):.1f}GiB "
        f"parallel_opponents={args.pool_parallel_inference} "
        f"torch_compile={args.compile_training_cnn} "
        f"torch={torch.__version__} runtime_cuda={torch.version.cuda} "
        f"cudnn={torch.backends.cudnn.version()}",
        flush=True,
    )
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")
    obs_storage_dtype = rollout_obs_dtype(args.rollout_obs_dtype)
    compute_dtype = amp_dtype(args.amp_dtype)
    if compute_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this CUDA device does not support bfloat16 AMP")
    action_mask_mode = args.agent_action_mask

    env = hisss.CudaBlackoutTorchVecEnv(
        args.num_envs,
        seed=args.seed,
        device="cuda",
        obs_dtype=environment_obs_dtype(obs_storage_dtype, compute_dtype),
        duel_probability=args.training_duel_probability,
    )

    def make_model() -> ActorCritic:
        result = ActorCritic().to(device)
        if args.channels_last:
            result.enable_channels_last()
        return result

    model = make_model()
    compiled_ppo_loss = None
    learner_observable_masks_fn = observable_action_masks
    if args.compile_training_cnn:
        model.enable_compiled_training_cnn(args.compile_mode)
        learner_compile_mode = compile_mode_without_cudagraphs(args.compile_mode)
        compiled_ppo_loss = torch.compile(
            ppo_loss_components,
            mode=learner_compile_mode,
            dynamic=False,
            fullgraph=True,
        )
        print(
            f"[cuda] torch.compile learner mode={learner_compile_mode} enabled "
            "for Autograd CNNs and PPO loss (CUDA Graphs disabled); "
            f"rollout mode={args.compile_mode}; the first update includes "
            "one-time compilation"
        )
    if args.compile_rollout_ops:
        learner_observable_masks_fn = compile_observable_action_masks(
            args.compile_mode
        )
        rollout_compile_mode = (
            "default"
            if args.compile_mode == "reduce-overhead"
            else args.compile_mode
        )
        print(
            f"[cuda] torch.compile mode={rollout_compile_mode} enabled for "
            "fixed-shape learner rollout masks (CUDA Graphs disabled; "
            f"revision={ROLLOUT_MASK_REVISION})"
        )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        eps=1e-5,
        fused=args.fused_adam,
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=compute_dtype == torch.float16
    )
    total_steps = 0
    completed_updates = 0
    continuation_source = args.resume_path or args.warm_start_path
    schedule_start_lr = args.lr
    if args.resume_path:
        total_steps, completed_updates = load_training_checkpoint(
            args.resume_path, model, optimizer, device, grad_scaler
        )
        saved_lr = float(optimizer.param_groups[0]["lr"])
        schedule_start_lr = min(args.lr, saved_lr)
        print(
            f"[resume] loaded {args.resume_path}: steps={total_steps} "
            f"completed_updates={completed_updates} saved_lr={saved_lr:.3g}"
        )
    elif args.warm_start_path:
        source_steps, source_update, source_kind, source_lr = load_warm_start(
            args.warm_start_path,
            model,
            device,
        )
        # Preserve provenance and avoid colliding with the source run's pool
        # filenames. The schedule itself still uses local updates.
        total_steps = source_steps
        completed_updates = source_update
        schedule_start_lr = min(
            args.lr,
            source_lr if source_lr is not None else args.lr_floor,
        )
        print(
            f"[warm-start] loaded {source_kind} weights from "
            f"{args.warm_start_path}: source_steps={source_steps} "
            f"source_update={source_update} source_lr={source_lr}; optimizer and "
            "AMP scaler start fresh, counters are preserved"
        )
    if continuation_source and args.continuation_lr is not None:
        if args.continuation_lr <= 0.0:
            raise ValueError("--continuation-lr must be positive")
        schedule_start_lr = args.continuation_lr
    if args.stop_after_update is not None:
        remaining_updates = max(0, args.stop_after_update - completed_updates)
        args.updates = min(args.updates, remaining_updates)
        print(
            f"[schedule] stop_after_update={args.stop_after_update} "
            f"remaining_updates={args.updates}"
        )
    set_optimizer_lr(optimizer, schedule_start_lr)
    args.effective_start_lr = schedule_start_lr
    if continuation_source:
        print(
            f"[continuation] monotonic schedule starts at {schedule_start_lr:.3g}; "
            "the LR will not be raised above this value"
        )
    actor_reference_parameters: list[torch.Tensor] | None = None
    if args.actor_reference_path:
        reference_model = make_model()
        _, reference_update, reference_kind, _ = load_warm_start(
            args.actor_reference_path,
            reference_model,
            device,
        )
        learner_actor_parameters, _ = actor_critic_parameter_groups(model)
        reference_actor_parameters, _ = actor_critic_parameter_groups(
            reference_model
        )
        if len(learner_actor_parameters) != len(reference_actor_parameters):
            raise RuntimeError(
                "actor reference architecture does not match the learner"
            )
        actor_reference_parameters = [
            parameter.detach().clone()
            for parameter in reference_actor_parameters
        ]
        with torch.no_grad():
            initial_reference_distance = torch.sqrt(
                sum(
                    (parameter.float() - reference.float()).square().sum()
                    for parameter, reference in zip(
                        learner_actor_parameters,
                        actor_reference_parameters,
                    )
                )
            ).item()
        del reference_model
        print(
            f"[actor-reference] loaded {reference_kind} update="
            f"{reference_update} coef={args.actor_reference_l2_coef:.4g} "
            f"initial_l2={initial_reference_distance:.6f}",
            flush=True,
        )
    critic_warmup_remaining = 0
    if args.critic_warmup_updates > 0:
        if not continuation_source:
            raise ValueError(
                "--critic-warmup-updates requires --warm-start-path or "
                "--resume-path; from scratch there is no actor worth freezing"
            )
        critic_warmup_remaining = args.critic_warmup_updates
        set_actor_frozen(model, True)
        print(
            f"[critic-warmup] actor frozen for the first "
            f"{critic_warmup_remaining} updates"
        )
    champion_model: ActorCritic | None = None
    champion_total_steps = 0
    champion_update = 0
    if args.league_champion_path:
        stable_champion = Path(args.league_champion_path)
        champion_source = (
            stable_champion
            if stable_champion.is_file()
            else (
                Path(args.league_initial_champion_path)
                if args.league_initial_champion_path
                else None
            )
        )
        champion_model = make_model()
        if champion_source is not None:
            (
                champion_total_steps,
                champion_update,
                champion_source_kind,
                _,
            ) = load_warm_start(str(champion_source), champion_model, device)
            print(
                f"[league] loaded {champion_source_kind} champion from "
                f"{champion_source}: steps={champion_total_steps} "
                f"update={champion_update}",
                flush=True,
            )
        else:
            champion_model.load_state_dict(model.state_dict(), strict=True)
            champion_total_steps = total_steps
            champion_update = completed_updates
            print("[league] initialized champion from the learner", flush=True)
        champion_model.eval()
        for parameter in champion_model.parameters():
            parameter.requires_grad_(False)
        if not stable_champion.is_file():
            save_league_champion(
                args.league_champion_path,
                champion_model,
                champion_total_steps,
                champion_update,
            )
        champion_zip = Path(args.league_champion_zip_path)
        # Re-export on startup as well, so an interrupted prior promotion can
        # never leave the Torch and SB3 champion files out of sync.
        export_league_champion(champion_model, str(champion_zip))
        print(
            f"[league] deployable champion: {stable_champion} / {champion_zip}",
            flush=True,
        )
    writer = create_tensorboard_writer(args, total_steps, model)
    rollout_start_event = torch.cuda.Event(enable_timing=True) if writer else None
    rollout_end_event = torch.cuda.Event(enable_timing=True) if writer else None
    ppo_start_event = torch.cuda.Event(enable_timing=True) if writer else None
    ppo_end_event = torch.cuda.Event(enable_timing=True) if writer else None

    actor_h, actor_c, critic_h, critic_c = model.initial_state(
        args.num_envs, device
    )
    alive_count = torch.full(
        (args.num_envs,), 4, dtype=torch.long, device=device
    )
    opponent_pool = None
    obs = legal_mask = obs_all = legal_all = None
    if args.opponent_mode in ("curriculum", "selfplay"):
        opponent_pool = CudaOpponentPool(
            model_factory=make_model,
            num_envs=args.num_envs,
            hidden_size=model.hidden_size,
            device=device,
            pool_dir=args.pool_dir,
            max_checkpoints=args.max_pool_checkpoints,
            checkpoint_weight=args.pool_checkpoint_weight,
            best_weight=args.pool_best_weight,
            hungry_weight=args.pool_hungry_weight,
            random_weight=args.pool_random_weight,
            latest_probability=args.pool_latest_probability,
            seed=args.seed + 1,
            load_existing=args.pool_load_existing,
            real_heuristic_weight=args.pool_real_heuristic_weight,
            heuristic_weights=(
                args.pool_heuristic_weight_map
                if args.pool_heuristic_weights
                else None
            ),
            inference_amp_dtype=compute_dtype,
            active_checkpoint_limit=args.pool_active_checkpoints,
            active_rotation_interval=args.pool_active_rotation_interval,
            anchor_probability=args.pool_anchor_probability,
            focus_label=args.pool_focus_label,
            focus_probability=args.pool_focus_probability,
            deterministic_probability=args.pool_deterministic_probability,
            copies_probability=args.pool_copies_probability,
            anchor_targets=args.pool_anchor_target_map,
            anchor_priority_exponent=args.pool_anchor_priority_exponent,
            anchor_min_weight=args.pool_anchor_min_weight,
            anchor_score_ema=args.pool_anchor_score_ema,
            anchor_uniform_floor=args.pool_anchor_uniform_floor,
            nash_history_size=args.pool_nash_history_size,
            nash_iterations=args.pool_nash_iterations,
            nash_exploration=args.pool_nash_exploration,
            nash_score_half_life_updates=args.pool_nash_score_half_life_updates,
            champion_min_weight=args.pool_champion_min_weight,
            deduplicate_champion=args.pool_deduplicate_champion,
            action_mask_mode=(
                args.agent_action_mask
                if args.agent_action_mask in ("observable", "observable-hard")
                else "observable"
            ),
            parallel_inference=args.pool_parallel_inference,
            min_action_disagreement=args.pool_min_action_disagreement,
            diversity_probe_size=args.pool_diversity_probe_size,
            max_snapshot_update=completed_updates if args.resume_path else None,
            duel_opponent_label=args.pool_duel_opponent_label,
            duel_opponent_probability=args.pool_duel_opponent_probability,
            duel_opponent_weights=args.pool_duel_opponent_weight_map,
        )
        for anchor_label, anchor_path in anchor_specs:
            opponent_pool.add_external_anchor(anchor_label, anchor_path)
        if champion_model is not None:
            opponent_pool.replace_anchor(
                LEAGUE_CHAMPION_LABEL,
                champion_model,
                champion_update,
                champion_total_steps,
            )
        if continuation_source:
            if champion_model is None:
                opponent_pool.add_anchor(
                    model,
                    completed_updates,
                    total_steps,
                    label=f"continuation_u{completed_updates:08d}",
                )
            opponent_pool.enable_checkpoint_sampling()
            opponent_pool.begin_rollout(completed_updates)
            opponent_pool.resample(
                torch.ones(args.num_envs, dtype=torch.bool, device=device)
            )
        obs_all, legal_all = env.reset_all()
        validate_cuda_all_seat_reset(
            obs_all,
            legal_all,
            context="self-play training",
            allow_duels=args.training_duel_probability > 0.0,
        )
        legal_all = legal_cuda_to_model(legal_all)
        reset_alive_count = legal_all.any(dim=-1).sum(dim=-1)
        alive_count.copy_(reset_alive_count)
        opponent_pool.resample(
            torch.ones(args.num_envs, dtype=torch.bool, device=device),
            duel_rows=reset_alive_count == 2,
        )
    else:
        obs, legal_mask = env.reset()
        legal_mask = legal_cuda_to_model(legal_mask)
    t0 = time.perf_counter()
    final_update = completed_updates
    periodic_export_count = 0
    schedule_update_offset = phase_schedule_offset(
        completed_updates,
        args.schedule_origin_update,
        resumed=bool(args.resume_path),
    )
    if args.schedule_origin_update is not None:
        print(
            f"[schedule] phase_origin_update={args.schedule_origin_update} "
            f"phase_completed_updates={schedule_update_offset} "
            f"lr_decay_updates={args.lr_decay_updates or args.updates} "
            f"ent_decay_updates={args.ent_decay_updates or args.updates}",
            flush=True,
        )
    last_champion_promotion_local_update = 0
    champion_anchor_baseline: dict[str, float] | None = None

    try:
        for local_update in range(1, args.updates + 1):
            update_wall_start = time.perf_counter()
            update = completed_updates + local_update
            final_update = update
            if critic_warmup_remaining and local_update > critic_warmup_remaining:
                set_actor_frozen(model, False)
                critic_warmup_remaining = 0
                print("[critic-warmup] warmup complete, actor unfrozen")
            cur_lr = lr_for_update(
                args,
                local_update,
                schedule_start_lr,
                update_offset=schedule_update_offset,
            )
            set_optimizer_lr(optimizer, cur_lr)
            if writer is not None:
                torch.cuda.reset_peak_memory_stats(device)
                rollout_start_event.record()

            rollout_wall_start = time.perf_counter()
            if args.opponent_mode in ("curriculum", "selfplay"):
                pool_enabled = (
                    args.opponent_mode == "selfplay"
                    or update >= args.selfplay_start_update
                )
                if pool_enabled and not opponent_pool.checkpoints_enabled:
                    if opponent_pool.checkpoint_count == 0:
                        opponent_pool.add_checkpoint(
                            model,
                            update=max(0, update - 1),
                            total_steps=total_steps,
                        )
                    opponent_pool.enable_checkpoint_sampling()
                if opponent_pool.checkpoints_enabled:
                    opponent_pool.begin_rollout(update)

                (
                    rollout,
                    obs_all,
                    legal_all,
                    actor_h,
                    actor_c,
                    critic_h,
                    critic_c,
                    alive_count,
                ) = collect_pool_rollout(
                    env,
                    model,
                    opponent_pool,
                    obs_all,
                    legal_all,
                    actor_h,
                    actor_c,
                    critic_h,
                    critic_c,
                    alive_count,
                    args.n_steps,
                    args.seq_len,
                    obs_storage_dtype,
                    args.gamma,
                    args.gae_lambda,
                    action_mask_mode,
                    args.potential_length_coef,
                    args.potential_health_coef,
                    args.potential_mobility_coef,
                    args.reward_scheme,
                    compute_dtype,
                    learner_observable_masks_fn,
                )
            else:
                (
                    rollout,
                    obs,
                    legal_mask,
                    actor_h,
                    actor_c,
                    critic_h,
                    critic_c,
                    alive_count,
                ) = collect_rollout(
                    env,
                    model,
                    obs,
                    legal_mask,
                    actor_h,
                    actor_c,
                    critic_h,
                    critic_c,
                    alive_count,
                    args.n_steps,
                    args.seq_len,
                    obs_storage_dtype,
                    args.gamma,
                    args.gae_lambda,
                    action_mask_mode,
                    args.potential_length_coef,
                    args.potential_health_coef,
                    args.potential_mobility_coef,
                    args.reward_scheme,
                    compute_dtype,
                    learner_observable_masks_fn,
                )
            rollout_wall_seconds = time.perf_counter() - rollout_wall_start
            if writer is not None:
                rollout_end_event.record()
                ppo_start_event.record()
            cur_ent_coef = entropy_coef_for_update(
                args,
                local_update,
                update_offset=schedule_update_offset,
            )
            ppo_wall_start = time.perf_counter()
            stats = ppo_update(
                model,
                optimizer,
                rollout,
                args.batch_size,
                args.epochs,
                args.clip_range,
                args.clip_range_vf,
                cur_ent_coef,
                args.vf_coef,
                args.max_grad_norm,
                args.target_kl,
                compute_dtype,
                grad_scaler,
                compiled_ppo_loss,
                args.kl_stop_mode,
                args.actor_max_grad_norm,
                args.critic_max_grad_norm,
                actor_reference_parameters,
                args.actor_reference_l2_coef,
            )
            ppo_wall_seconds = time.perf_counter() - ppo_wall_start
            if writer is not None:
                ppo_end_event.record()
                ppo_end_event.synchronize()
                rollout_seconds = (
                    rollout_start_event.elapsed_time(rollout_end_event) / 1000.0
                )
                ppo_seconds = ppo_start_event.elapsed_time(ppo_end_event) / 1000.0
            total_steps += args.num_envs * args.n_steps
            snapshot_wall_seconds = 0.0
            if (
                opponent_pool is not None
                and opponent_pool.checkpoints_enabled
                and update - opponent_pool.last_snapshot_attempt_update
                >= args.selfplay_update_interval
            ):
                snapshot_wall_start = time.perf_counter()
                opponent_pool.add_checkpoint(
                    model,
                    update,
                    total_steps,
                    probe_obs=rollout.obs,
                )
                snapshot_wall_seconds = (
                    time.perf_counter() - snapshot_wall_start
                )

            active_training_seconds = (
                rollout_seconds + ppo_seconds
                if writer is not None
                else rollout_wall_seconds + ppo_wall_seconds
            )
            active_steps_per_second = (
                args.num_envs
                * args.n_steps
                / max(active_training_seconds, 1e-9)
            )
            elapsed = max(time.perf_counter() - t0, 1e-6)
            run_steps = local_update * args.num_envs * args.n_steps
            cumulative_steps_per_second = run_steps / elapsed
            tensorboard_wall_start = time.perf_counter()
            if (
                writer is not None
                and (
                    local_update == 1
                    or update % args.tensorboard_update_interval == 0
                )
            ):
                log_tensorboard_update(
                    writer,
                    update=update,
                    total_steps=total_steps,
                    update_steps=args.num_envs * args.n_steps,
                    rollout=rollout,
                    stats=stats,
                    learning_rate=cur_lr,
                    entropy_coefficient=cur_ent_coef,
                    rollout_seconds=rollout_seconds,
                    ppo_seconds=ppo_seconds,
                    cumulative_steps_per_second=cumulative_steps_per_second,
                    device=device,
                    grad_scaler=grad_scaler,
                    opponent_pool=opponent_pool,
                )
            tensorboard_wall_seconds = (
                time.perf_counter() - tensorboard_wall_start
            )

            update_wall_seconds = time.perf_counter() - update_wall_start
            if (
                args.timing_log_interval
                and (
                    local_update == 1
                    or update % args.timing_log_interval == 0
                )
            ):
                unaccounted_wall_seconds = max(
                    0.0,
                    update_wall_seconds
                    - rollout_wall_seconds
                    - ppo_wall_seconds
                    - snapshot_wall_seconds
                    - tensorboard_wall_seconds,
                )
                print(
                    f"[timing] update={update} "
                    f"rollout_wall_s={rollout_wall_seconds:.3f} "
                    f"ppo_wall_s={ppo_wall_seconds:.3f} "
                    f"snapshot_wall_s={snapshot_wall_seconds:.3f} "
                    f"tensorboard_wall_s={tensorboard_wall_seconds:.3f} "
                    f"other_wall_s={unaccounted_wall_seconds:.3f} "
                    f"update_wall_s={update_wall_seconds:.3f}",
                    flush=True,
                )

            if local_update == 1 or update % 10 == 0:
                pool_summary = (
                    f" pool={opponent_pool.checkpoint_count}/"
                    f"{opponent_pool.max_checkpoints} "
                    f"{opponent_pool.assignment_summary()}"
                    if opponent_pool is not None
                    else ""
                )
                timing_summary = (
                    f" train_steps/s={active_steps_per_second:.0f}"
                    f" rollout_s={rollout_seconds:.3f} ppo_s={ppo_seconds:.3f}"
                    if writer is not None
                    else ""
                )
                print(
                    f"update={update} steps={total_steps} "
                    f"steps/s={active_steps_per_second:.0f} "
                    f"cumulative_steps/s={cumulative_steps_per_second:.0f} "
                    f"env_reward_mean={rollout.env_rewards.mean().item():.4f} "
                    f"shaped_reward_mean={rollout.rewards.mean().item():.4f} "
                    f"done_frac={rollout.dones.float().mean().item():.4f} "
                    f"illegal_frac={rollout.illegal_actions.float().mean().item():.4f} "
                    f"loss={stats.get('loss', 0.0):.4f} "
                    f"value_loss={stats.get('value_loss', 0.0):.4f} "
                    f"explained_var={stats.get('explained_variance', 0.0):.3f} "
                    f"entropy={stats.get('entropy', 0.0):.4f} "
                    f"ent_coef={cur_ent_coef:.4f} "
                    f"anchor_l2={stats.get('actor_reference_loss', 0.0):.6f} "
                    f"kl={stats.get('approx_kl', 0.0):.5f} "
                    f"clipfrac={stats.get('clipfrac', 0.0):.3f} "
                    f"lr={cur_lr:.2e} "
                    f"opt_fraction={stats.get('optimizer_step_fraction', 0.0):.2f} "
                    f"early_stop={int(stats.get('early_stop', 0.0))}"
                    f"{timing_summary}"
                    f"{pool_summary}",
                    flush=True,
                )

            # During ``rollout = collect_*()`` Python keeps the previous value
            # alive until the right-hand side returns. Releasing it here avoids
            # overlapping two large rollout buffers on the next update.
            del rollout

            if (
                opponent_pool is not None
                and should_run_evaluation(
                    local_update,
                    completed_updates,
                    args.eval_interval,
                )
            ):
                evaluation_wall_start = time.perf_counter()
                evaluation_results = evaluate_against_history(
                    model,
                    opponent_pool,
                    args,
                    update=update,
                    total_steps=total_steps,
                    writer=writer,
                    device=device,
                )
                anchor_priorities = opponent_pool.update_anchor_priorities(
                    evaluation_results
                )
                if anchor_priorities:
                    priority_summary = " ".join(
                        f"{label}=score:{score:.3f}/weight:{weight:.3f}"
                        for label, (score, weight) in sorted(
                            anchor_priorities.items()
                        )
                    )
                    print(f"[pfsp] {priority_summary}", flush=True)
                    if writer is not None:
                        for label, (score, weight) in anchor_priorities.items():
                            writer.add_scalar(
                                f"selfplay/anchor_score/{label}", score, total_steps
                            )
                            writer.add_scalar(
                                f"selfplay/anchor_weight/{label}", weight, total_steps
                            )
                nash_weights = opponent_pool.update_nash_distribution(
                    evaluation_results,
                    update=update,
                )
                if nash_weights:
                    strongest = sorted(
                        nash_weights.items(), key=lambda item: item[1], reverse=True
                    )[:6]
                    training_weights = opponent_pool.effective_training_weights()
                    heuristic_mix = " ".join(
                        f"{key}={training_weights.get(key, 0.0):.3f}"
                        for key in (
                            opponent_pool.opponent_key(opponent_id)
                            for _, opponent_id in opponent_pool.scripted_opponents
                        )
                    )
                    print(
                        f"[nash] rows={len(opponent_pool.nash_rows)} "
                        f"value={opponent_pool.nash_value:.3f} mix="
                        + " ".join(
                            f"{key}={weight:.3f}" for key, weight in strongest
                        )
                        + f" training_heuristics={heuristic_mix}",
                        flush=True,
                    )
                    if writer is not None:
                        writer.add_scalar(
                            "selfplay/nash/value",
                            opponent_pool.nash_value,
                            total_steps,
                        )
                        writer.add_scalar(
                            "selfplay/nash/matrix_rows",
                            len(opponent_pool.nash_rows),
                            total_steps,
                        )
                        for key, weight in nash_weights.items():
                            writer.add_scalar(
                                f"selfplay/nash_weight/{key.replace(':', '/')}",
                                weight,
                                total_steps,
                            )
                if (
                    champion_model is not None
                    and local_update % args.eval_interval == 0
                    and local_update - last_champion_promotion_local_update
                    >= args.league_min_promotion_interval
                ):
                    relative_gate_enabled = (
                        args.league_general_improvement is not None
                    )
                    if relative_gate_enabled and champion_anchor_baseline is None:
                        champion_anchor_baseline = (
                            evaluate_champion_anchor_baseline(
                                champion_model,
                                opponent_pool,
                                args,
                                update=update,
                                total_steps=total_steps,
                                writer=writer,
                                device=device,
                            )
                        )
                    comparisons = evaluate_league_candidate(
                        model,
                        champion_model,
                        args,
                        update=update,
                        total_steps=total_steps,
                        device=device,
                    )
                    (
                        promote,
                        direct_score,
                        guard_score,
                        promotion_reason,
                    ) = league_promotion_decision(
                        comparisons,
                        evaluation_results,
                        threshold=args.league_promotion_threshold,
                        guard_labels=args.league_guard_labels,
                        min_guard_score=args.league_min_guard_score,
                        guard_thresholds=args.league_guard_score_map,
                    )
                    general_mean_delta = None
                    general_worst_delta = None
                    if relative_gate_enabled:
                        (
                            general_ok,
                            general_mean_delta,
                            general_worst_delta,
                            general_reason,
                        ) = league_generalization_decision(
                            evaluation_results,
                            champion_anchor_baseline,
                            min_mean_improvement=(
                                args.league_general_improvement
                            ),
                            max_anchor_regression=(
                                args.league_max_anchor_regression
                            ),
                        )
                        print(
                            f"[league-general] update={update} "
                            f"mean_delta="
                            f"{f'{general_mean_delta:+.3f}' if general_mean_delta is not None else 'n/a'} "
                            f"worst_delta="
                            f"{f'{general_worst_delta:+.3f}' if general_worst_delta is not None else 'n/a'} "
                            f"passed={int(general_ok)}",
                            flush=True,
                        )
                        if promote and not general_ok:
                            promote = False
                            promotion_reason = general_reason
                    if writer is not None:
                        writer.add_scalar(
                            "league/candidate_score", direct_score, total_steps
                        )
                        writer.add_scalar(
                            "league/promoted", float(promote), total_steps
                        )
                        if guard_score is not None:
                            writer.add_scalar(
                                "league/worst_guard_score",
                                guard_score,
                                total_steps,
                            )
                        if general_mean_delta is not None:
                            writer.add_scalar(
                                "league/fixed_anchor_mean_delta",
                                general_mean_delta,
                                total_steps,
                            )
                        if general_worst_delta is not None:
                            writer.add_scalar(
                                "league/fixed_anchor_worst_delta",
                                general_worst_delta,
                                total_steps,
                            )
                    if promote:
                        if champion_anchor_baseline is not None:
                            candidate_anchor_scores = {
                                str(result["label"]): (
                                    float(result["current_win_rate"])
                                    + 0.5 * float(result["draw_rate"])
                                )
                                for result in evaluation_results
                                if result.get("kind") == "anchor"
                                and str(result.get("label"))
                                in champion_anchor_baseline
                            }
                            if set(candidate_anchor_scores) != set(
                                champion_anchor_baseline
                            ):
                                raise RuntimeError(
                                    "promoted candidate is missing fixed-anchor "
                                    "baseline scores"
                                )
                            champion_anchor_baseline = candidate_anchor_scores
                        champion_model.load_state_dict(
                            model.state_dict(), strict=True
                        )
                        champion_model.eval()
                        champion_total_steps = total_steps
                        champion_update = update
                        last_champion_promotion_local_update = local_update
                        save_league_champion(
                            args.league_champion_path,
                            champion_model,
                            champion_total_steps,
                            champion_update,
                        )
                        export_league_champion(
                            champion_model,
                            args.league_champion_zip_path,
                        )
                        opponent_pool.replace_anchor(
                            LEAGUE_CHAMPION_LABEL,
                            champion_model,
                            champion_update,
                            champion_total_steps,
                        )
                        # The just-promoted policy is now its own champion.
                        # Reusing the candidate's score against the previous
                        # champion would make the Nash column immediately stale.
                        champion_id = opponent_pool.pinned_id_for_label(
                            LEAGUE_CHAMPION_LABEL
                        )
                        if champion_id is None:
                            raise RuntimeError(
                                "promoted league champion is missing from the pool"
                            )
                        champion_key = opponent_pool.opponent_key(champion_id)
                        opponent_pool.nash_latest_scores[champion_key] = 0.5
                        opponent_pool.nash_score_updates[champion_key] = update
                        opponent_pool._save_nash_state(update)
                        print(
                            f"[league] PROMOTED update={update} steps={total_steps} "
                            f"direct_score={direct_score:.3f} "
                            f"guard_score="
                            f"{f'{guard_score:.3f}' if guard_score is not None else 'n/a'}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[league] kept champion update={champion_update}: "
                            f"{promotion_reason}",
                            flush=True,
                        )
                    if writer is not None:
                        writer.add_scalar(
                            "league/champion_update", champion_update, total_steps
                        )
                print(
                    f"[timing] update={update} evaluation_wall_s="
                    f"{time.perf_counter() - evaluation_wall_start:.3f}",
                    flush=True,
                )

            if (
                args.sb3_checkpoint_interval
                and local_update % args.sb3_checkpoint_interval == 0
            ):
                checkpoint_wall_start = time.perf_counter()
                training_checkpoint_path = save_periodic_training_checkpoint(
                    model,
                    optimizer,
                    args.sb3_checkpoint_dir,
                    args.sb3_checkpoint_prefix,
                    total_steps,
                    update,
                    args,
                    grad_scaler,
                    args.max_sb3_checkpoints,
                )
                export_path = export_periodic_sb3_checkpoint(
                    model,
                    args.sb3_checkpoint_dir,
                    args.sb3_checkpoint_prefix,
                    total_steps,
                    update,
                    args.max_sb3_checkpoints,
                )
                periodic_export_count += 1
                if writer is not None:
                    writer.add_scalar(
                        "checkpoints/periodic_sb3_export_count",
                        periodic_export_count,
                        total_steps,
                    )
                    writer.add_scalar(
                        "checkpoints/periodic_training_checkpoint_count",
                        periodic_export_count,
                        total_steps,
                    )
                print(
                    f"[checkpoint] saved resumable training state: "
                    f"{training_checkpoint_path}",
                    flush=True,
                )
                print(
                    f"[checkpoint] exported ppo.py-compatible variant: {export_path}",
                    flush=True,
                )
                print(
                    f"[timing] update={update} checkpoint_wall_s="
                    f"{time.perf_counter() - checkpoint_wall_start:.3f}",
                    flush=True,
                )

        save_training_checkpoint(
            args.save_path,
            model,
            optimizer,
            total_steps,
            final_update,
            args,
            grad_scaler,
        )
        print(f"Saved Torch checkpoint to {args.save_path}")
        if args.sb3_save_path:
            export_model = champion_model if champion_model is not None else model
            export_sb3_recurrent_ppo(export_model, args.sb3_save_path)
            export_kind = "league champion" if champion_model is not None else "policy"
            print(
                f"Saved ppo.py-compatible {export_kind} to {args.sb3_save_path}"
            )
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    main()
