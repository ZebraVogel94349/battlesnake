"""Single source of truth shared by train.py / ppo.py / selfplay.py:

- which hisss encoder channels the model actually sees (KEPT_CHANNELS, to_model_obs)
- the network architecture (BattlesnakeCNN, policy_kwargs, build_model)
- LSTM state helpers for stateful inference (initial_lstm_states)

Training and inference both import from here, so the observation layout and the
policy skeleton can never silently diverge between the two paths — the main
error source of the old 15-channel setup, where the same policy_kwargs dict was
copy-pasted across three files.
"""
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import hisss

# The trimmed channel set (15 -> 9). Names are hisss encoder layer names, see
# docs/hisss_observations.md. Dropped: number_of_turns (the LSTM can count),
# distance_map (derived, learnable), 0_snake_body_as_one_hot (redundant with
# own decay), 1_snake_tail / 1_snake_health (fog-truncated and therefore
# unreliable when reconstructed from live JSON in ppo.py).
# Enemy body is the BINARY variant, not decay: the encoder computes decay from
# an enemy's TRUE length, which fog makes unknowable at inference (JSON length
# is a lower bound) — the decay values on visible cells are simply not
# reconstructable. Occupancy is. Own decay stays (own body is never fogged in
# the live JSON, verified against docs/log.json).
KEPT_CHANNELS = [
    "current_food",
    "board",                     # in-bounds vs out-of-bounds (wall/edge)
    "0_snake_body",              # own body, decay = position from head
    "0_snake_head",
    "0_snake_health",
    "0_snake_tail",              # disambiguates just-ate/tail-chase from the decay channel
    "1_snake_body_as_one_hot",   # all enemies compressed, binary occupancy
    "1_snake_head",
    "view_mask",                 # fog visibility
]
N_CHANNELS = len(KEPT_CHANNELS)
OBS_SHAPE = (N_CHANNELS, 29, 29)


def make_game_config():
    """The one game/encoder config both training and inference must use."""
    cfg = hisss.restricted_standard_config()
    cfg.ec.include_snake_length = False
    return cfg


_KEPT_IDX = None


def kept_indices() -> np.ndarray:
    """Encoder channel indices of KEPT_CHANNELS, resolved by name once."""
    global _KEPT_IDX
    if _KEPT_IDX is None:
        mapping = hisss.encoding_layer_indices(make_game_config())
        _KEPT_IDX = np.array([mapping[name] for name in KEPT_CHANNELS], dtype=np.int64)
    return _KEPT_IDX


def to_model_obs(obs_hwc: np.ndarray) -> np.ndarray:
    """Full encoder output (29,29,15) HWC -> model obs (9,29,29) CHW float32."""
    return np.transpose(obs_hwc[:, :, kept_indices()], (2, 0, 1)).astype(np.float32)


class BattlesnakeCNN(BaseFeaturesExtractor):
    """CNN sized for 29×29×N Battlesnake observations (N = N_CHANNELS)."""
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]  # channels-first

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),  # -> 32×29×29
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # -> 64×15×15
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),          # -> 64×8×8
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.zeros(1, n_channels, 29, 29)
            n_flatten = self.cnn(sample).shape[1]  # = 64 * 8 * 8 = 4096

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def policy_kwargs() -> dict:
    """Architecture of the recurrent policy: CNN extractor -> LSTM -> heads."""
    return dict(
        features_extractor_class=BattlesnakeCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,
        net_arch=dict(pi=[128, 128], vf=[128, 128]),  # separate heads after the LSTM
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,  # critic gets its own LSTM (sb3-contrib default)
    )


class _DummyEnv(gym.Env):
    """Shape-only env to build a model skeleton for weight loading."""
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=OBS_SHAPE, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        return np.zeros(OBS_SHAPE, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(OBS_SHAPE, dtype=np.float32), 0, False, False, {}


def build_model(env=None, device: str = "cpu", **kwargs):
    """Fresh RecurrentPPO with the shared architecture.

    Load weights afterwards via model.set_parameters(path) — never
    RecurrentPPO.load (full-zip unpickling segfaults, same as MaskablePPO.load).
    """
    from sb3_contrib import RecurrentPPO
    return RecurrentPPO(
        "CnnLstmPolicy",
        env if env is not None else _DummyEnv(),
        policy_kwargs=policy_kwargs(),
        device=device,
        **kwargs,
    )


def initial_lstm_states(policy) -> tuple[th.Tensor, th.Tensor]:
    """Zero (h, c) actor-LSTM state for a single-env forward pass."""
    n_layers = policy.lstm_actor.num_layers
    hidden = policy.lstm_actor.hidden_size
    dev = policy.device
    return (th.zeros(n_layers, 1, hidden, device=dev),
            th.zeros(n_layers, 1, hidden, device=dev))


def recurrent_logits(policy, obs_chw: np.ndarray, lstm_states, episode_start: bool):
    """One stateful actor forward: returns (action logits, new lstm_states).

    lstm_states=None means "start of episode" state (zeros). The LSTM consumes
    only the observation, so callers may pick any action from the logits
    (masking, sampling, argmax) without corrupting the hidden state.
    """
    obs_t, _ = policy.obs_to_tensor(obs_chw)
    with th.no_grad():
        states = lstm_states if lstm_states is not None else initial_lstm_states(policy)
        starts = th.tensor([1.0 if episode_start else 0.0],
                           dtype=th.float32, device=policy.device)
        dist, new_states = policy.get_distribution(obs_t, states, starts)
        logits = dist.distribution.logits.squeeze(0)
    return logits, new_states
