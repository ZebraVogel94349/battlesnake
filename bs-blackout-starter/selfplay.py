"""Checkpointed self-play: frozen past versions of our own model as opponents.

Frozen opponents live inside the training env workers and act directly on the
raw hisss observations (no JSON round-trip like the BaseAgent opponents), so a
move costs one small CPU forward. Policies are cached per process, so each
SubprocVecEnv worker loads every checkpoint exactly once (~5MB each).

LSTM note: the cached FrozenPolicy is shared between opponent slots, so the
recurrent hidden state must live on the per-episode FrozenOpponent instance
(train.py builds fresh opponents every reset(), which resets the state).
"""
import glob
import os
import random
import shutil

import numpy as np
import torch

from obs_config import build_model, recurrent_logits

# NEW dir for the from-scratch LSTM run: the old CNN pool zips are
# architecturally incompatible and must never be sampled.
# Overridable via env var so smoke tests can use a throwaway pool (leftover
# test snapshots in the real pool would auto-open the self-play gate).
# SubprocVecEnv workers inherit the env var, so all processes agree.
POOL_DIR = os.environ.get("BS_POOL_DIR", "models/selfplay_pool_lstm")
# From-scratch run: no seeds. The pool stays empty (env workers then play the
# phase-A heuristic mix) until the eval gate in train.py opens snapshotting.
POOL_SEEDS: dict[str, str] = {}
P_LATEST = 0.5  # chance to face the newest snapshot instead of a uniform pool sample


class FrozenPolicy:
    """A frozen checkpoint's policy net, loaded once per process via the cache."""
    _cache: dict = {}

    def __init__(self, path: str):
        model = build_model(device="cpu")
        # set_parameters, never .load (segfaults on unpickle)
        model.set_parameters(path, device="cpu")
        self.policy = model.policy
        self.policy.set_training_mode(False)

    @classmethod
    def get(cls, path: str) -> "FrozenPolicy":
        if path not in cls._cache:
            cls._cache[path] = FrozenPolicy(path)
        return cls._cache[path]

    def act(self, obs_chw: np.ndarray, mask: np.ndarray, lstm_states):
        """Sample a masked action, threading the LSTM state (None = episode
        start). Stochastic on purpose: deterministic frozen opponents would
        mirror each other into mutual head-collisions, exactly the
        Hungry-vs-Hungry artifact self-play is meant to remove."""
        logits, new_states = recurrent_logits(
            self.policy, obs_chw, lstm_states, episode_start=lstm_states is None)
        masked = torch.where(torch.as_tensor(mask, dtype=torch.bool),
                             logits, torch.full_like(logits, -1e9))
        return int(torch.distributions.Categorical(logits=masked).sample()), new_states


class FrozenOpponent:
    """Wrapper the training env puts into an opponent slot. Holds the
    per-episode LSTM state (fresh instance per reset() = fresh state)."""
    def __init__(self, path: str):
        self.path = path
        self._policy = FrozenPolicy.get(path)
        self._lstm_states = None

    def act(self, obs_chw: np.ndarray, mask: np.ndarray) -> int:
        action, self._lstm_states = self._policy.act(obs_chw, mask, self._lstm_states)
        return action


def ensure_pool_seeded(pool_dir: str = POOL_DIR) -> None:
    os.makedirs(pool_dir, exist_ok=True)
    for name, src in POOL_SEEDS.items():
        dst = os.path.join(pool_dir, name)
        if not os.path.exists(dst):
            if os.path.exists(src):
                shutil.copyfile(src, dst)
            else:
                print(f"Warning: pool seed source missing: {src}")


def sample_pool_path(pool_dir: str = POOL_DIR, p_latest: float = P_LATEST) -> str | None:
    """With p_latest pick the newest zip (freshest snapshot), else uniform."""
    paths = glob.glob(os.path.join(pool_dir, "*.zip"))
    if not paths:
        return None
    if random.random() < p_latest:
        return max(paths, key=os.path.getmtime)
    return random.choice(paths)
