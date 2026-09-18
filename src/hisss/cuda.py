import copy
import ctypes as ct
from dataclasses import dataclass

import numpy as np

from hisss.cpp.lib import CPP_LIB
from hisss.game.config import (
    BattleSnakeConfig,
    post_init_battlesnake_cfg,
    validate_battlesnake_cfg,
)


# Native profile ids accepted by CudaBlackoutTorchVecEnv.heuristic_actions().
BLACKOUT_HEURISTIC_FORAGER = 0
BLACKOUT_HEURISTIC_HUNTER = 1
BLACKOUT_HEURISTIC_TERRITORIAL = 2
BLACKOUT_HEURISTIC_EDGE_TRAPPER = 3
BLACKOUT_HEURISTIC_SURVIVOR = 4
BLACKOUT_HEURISTIC_SNAKE25 = 5
BLACKOUT_HEURISTIC_SNAKE25_INTERCEPTOR = 6
BLACKOUT_HEURISTIC_SNAKE25_DENIER = 7
BLACKOUT_HEURISTIC_SNAKE25_DUELIST = 8
BLACKOUT_HEURISTIC_COUNT = 9


@dataclass
class CudaRolloutResult:
    """Result arrays from CUDA random-agent Battlesnake rollouts."""

    turns_played: np.ndarray
    terminal: np.ndarray
    winner: np.ndarray
    alive: np.ndarray
    lengths: np.ndarray
    health: np.ndarray
    death_cause: np.ndarray
    death_turn: np.ndarray
    killer_id: np.ndarray


@dataclass
class CudaBlackoutStep:
    obs: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    legal_mask: np.ndarray


@dataclass
class CudaBlackoutTorchStep:
    obs: object
    rewards: object
    done: object
    legal_mask: object


@dataclass
class CudaBlackoutTorchAllStep:
    obs: object
    rewards: object
    done: object
    legal_mask: object


@dataclass
class CudaBlackoutTorchEvalStep:
    obs: object
    done: object
    legal_mask: object
    winner: object
    alive: object
    turns: object


def cuda_available() -> bool:
    """Return whether hisss was built with CUDA support and can see a CUDA device."""

    return CPP_LIB.cuda_available()


def cuda_last_error() -> str:
    """Return the latest CUDA backend error message."""

    return CPP_LIB.cuda_last_error()


class CudaBlackoutVecEnv:
    """Persistent CUDA batch environment for Battlesnake Blackout.

    This is the low-level building block for GPU PPO. It keeps all game states
    on the GPU between calls. Snake 0 is controlled by the provided actions;
    snakes 1-3 currently use the placeholder random agent on the GPU. With a
    non-zero ``duel_probability``, selected resets keep only snakes 0 and 1.
    """

    # 9 fogged policy channels + 5 privileged (unfogged) critic-only
    # channels; must match BLACKOUT_OBS_CHANNELS in battlesnake_cuda.cu.
    obs_shape = (14, 29, 29)

    def __init__(
        self,
        num_envs: int,
        seed: int = 0,
        duel_probability: float = 0.0,
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not 0.0 <= duel_probability <= 1.0:
            raise ValueError("duel_probability must be in [0, 1]")
        if not cuda_available():
            raise RuntimeError(cuda_last_error())
        self.num_envs = num_envs
        self.duel_probability = float(duel_probability)
        self._closed = False
        self._env_p = CPP_LIB.lib.cuda_blackout_vec_create_cpp(
            num_envs,
            seed & ((1 << 64) - 1),
            round(self.duel_probability * 1_000_000),
        )
        if not self._env_p:
            raise RuntimeError(cuda_last_error())

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self._check_open()
        obs = np.zeros((self.num_envs, *self.obs_shape), dtype=np.float32)
        legal = np.zeros((self.num_envs, 4), dtype=bool)
        code = CPP_LIB.lib.cuda_blackout_vec_reset_cpp(
            self._env_p,
            obs.ctypes.data_as(ct.POINTER(ct.c_float)),
            legal.ctypes.data_as(ct.POINTER(ct.c_bool)),
        )
        if code != 0:
            raise RuntimeError(cuda_last_error())
        return obs, legal

    def step(self, actions: np.ndarray) -> CudaBlackoutStep:
        self._check_open()
        action_arr = np.asarray(actions, dtype=ct.c_int)
        if action_arr.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape ({self.num_envs},)")
        obs = np.zeros((self.num_envs, *self.obs_shape), dtype=np.float32)
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        done = np.zeros((self.num_envs,), dtype=bool)
        legal = np.zeros((self.num_envs, 4), dtype=bool)
        code = CPP_LIB.lib.cuda_blackout_vec_step_cpp(
            self._env_p,
            action_arr.ctypes.data_as(ct.POINTER(ct.c_int)),
            obs.ctypes.data_as(ct.POINTER(ct.c_float)),
            rewards.ctypes.data_as(ct.POINTER(ct.c_float)),
            done.ctypes.data_as(ct.POINTER(ct.c_bool)),
            legal.ctypes.data_as(ct.POINTER(ct.c_bool)),
        )
        if code != 0:
            raise RuntimeError(cuda_last_error())
        return CudaBlackoutStep(obs=obs, rewards=rewards, done=done, legal_mask=legal)

    def close(self):
        if not self._closed:
            CPP_LIB.lib.cuda_blackout_vec_close_cpp(self._env_p)
            self._closed = True

    def _check_open(self):
        if self._closed:
            raise ValueError("Cannot use closed CudaBlackoutVecEnv")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CudaBlackoutTorchVecEnv:
    """Torch-CUDA tensor variant of :class:`CudaBlackoutVecEnv`.

    All returned tensors live on the requested CUDA device. The underlying C++
    API receives raw device pointers from ``tensor.data_ptr()``; observations,
    rewards, done flags, legal masks, and actions therefore avoid NumPy/CPU
    staging. This is intended for custom PPO rollout collection.
    """

    # 9 fogged policy channels + 5 privileged (unfogged) critic-only
    # channels; must match BLACKOUT_OBS_CHANNELS in battlesnake_cuda.cu.
    obs_shape = (14, 29, 29)

    def __init__(
        self,
        num_envs: int,
        seed: int = 0,
        device: str = "cuda",
        obs_dtype=None,
        duel_probability: float = 0.0,
    ):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("CudaBlackoutTorchVecEnv requires torch") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is not available")
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not 0.0 <= duel_probability <= 1.0:
            raise ValueError("duel_probability must be in [0, 1]")
        if not cuda_available():
            raise RuntimeError(cuda_last_error())
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CudaBlackoutTorchVecEnv requires a CUDA device")
        self.num_envs = num_envs
        self.duel_probability = float(duel_probability)
        if isinstance(obs_dtype, str):
            try:
                obs_dtype = {"float16": torch.float16, "float32": torch.float32}[
                    obs_dtype
                ]
            except KeyError as exc:
                raise ValueError(
                    "obs_dtype must be torch.float16 or torch.float32"
                ) from exc
        self.obs_dtype = torch.float32 if obs_dtype is None else obs_dtype
        if self.obs_dtype not in (torch.float16, torch.float32):
            raise ValueError("obs_dtype must be torch.float16 or torch.float32")
        self._closed = False
        self._env_p = CPP_LIB.lib.cuda_blackout_vec_create_cpp(
            num_envs,
            seed & ((1 << 64) - 1),
            round(self.duel_probability * 1_000_000),
        )
        if not self._env_p:
            raise RuntimeError(cuda_last_error())

    def reset(self):
        self._check_open()
        obs = self.torch.empty((self.num_envs, *self.obs_shape), dtype=self.torch.float32, device=self.device)
        legal = self.torch.empty((self.num_envs, 4), dtype=self.torch.bool, device=self.device)
        code = CPP_LIB.lib.cuda_blackout_vec_reset_device_cpp(
            self._env_p,
            obs.data_ptr(),
            legal.data_ptr(),
        )
        if code != 0:
            raise RuntimeError(cuda_last_error())
        return obs.to(self.obs_dtype), legal

    def reset_all(self):
        self._check_open()
        obs = self.torch.empty(
            (self.num_envs, 4, *self.obs_shape),
            dtype=self.obs_dtype,
            device=self.device,
        )
        legal = self.torch.empty((self.num_envs, 4, 4), dtype=self.torch.bool, device=self.device)
        stream_ptr = self.torch.cuda.current_stream(self.device).cuda_stream
        code = CPP_LIB.lib.cuda_blackout_vec_reset_all_device_async_cpp(
            self._env_p,
            obs.data_ptr(),
            legal.data_ptr(),
            self.obs_dtype == self.torch.float16,
            stream_ptr,
        )
        if code != 0:
            raise RuntimeError(cuda_last_error())
        return obs, legal

    def step(self, actions) -> CudaBlackoutTorchStep:
        self._check_open()
        action_tensor = self.torch.as_tensor(actions, dtype=self.torch.int32, device=self.device)
        if tuple(action_tensor.shape) != (self.num_envs,):
            raise ValueError(f"actions must have shape ({self.num_envs},)")
        if not action_tensor.is_contiguous():
            action_tensor = action_tensor.contiguous()
        obs = self.torch.empty((self.num_envs, *self.obs_shape), dtype=self.torch.float32, device=self.device)
        rewards = self.torch.empty((self.num_envs,), dtype=self.torch.float32, device=self.device)
        done = self.torch.empty((self.num_envs,), dtype=self.torch.bool, device=self.device)
        legal = self.torch.empty((self.num_envs, 4), dtype=self.torch.bool, device=self.device)
        code = CPP_LIB.lib.cuda_blackout_vec_step_device_cpp(
            self._env_p,
            action_tensor.data_ptr(),
            obs.data_ptr(),
            rewards.data_ptr(),
            done.data_ptr(),
            legal.data_ptr(),
        )
        if code != 0:
            raise RuntimeError(cuda_last_error())
        return CudaBlackoutTorchStep(
            obs=obs.to(self.obs_dtype),
            rewards=rewards,
            done=done,
            legal_mask=legal,
        )

    def step_all(self, actions) -> CudaBlackoutTorchAllStep:
        self._check_open()
        action_tensor = self.torch.as_tensor(actions, dtype=self.torch.int32, device=self.device)
        if tuple(action_tensor.shape) != (self.num_envs, 4):
            raise ValueError(f"actions must have shape ({self.num_envs}, 4)")
        if not action_tensor.is_contiguous():
            action_tensor = action_tensor.contiguous()
        obs = self.torch.empty(
            (self.num_envs, 4, *self.obs_shape),
            dtype=self.obs_dtype,
            device=self.device,
        )
        rewards = self.torch.empty((self.num_envs,), dtype=self.torch.float32, device=self.device)
        done = self.torch.empty((self.num_envs,), dtype=self.torch.bool, device=self.device)
        legal = self.torch.empty((self.num_envs, 4, 4), dtype=self.torch.bool, device=self.device)
        stream_ptr = self.torch.cuda.current_stream(self.device).cuda_stream
        code = CPP_LIB.lib.cuda_blackout_vec_step_all_device_async_cpp(
            self._env_p,
            action_tensor.data_ptr(),
            obs.data_ptr(),
            rewards.data_ptr(),
            done.data_ptr(),
            legal.data_ptr(),
            self.obs_dtype == self.torch.float16,
            stream_ptr,
        )
        if code != 0:
            raise RuntimeError(cuda_last_error())
        return CudaBlackoutTorchAllStep(
            obs=obs,
            rewards=rewards,
            done=done,
            legal_mask=legal,
        )

    def step_all_eval(self, actions, max_turns: int = 1000) -> CudaBlackoutTorchEvalStep:
        """Step all snakes until the true game end, for CUDA evaluation.

        Unlike :meth:`step_all`, an episode does not end merely because snake
        zero dies. Completed and horizon-truncated games are auto-reset, while
        ``winner``, ``alive``, and ``turns`` describe the game before reset.
        ``winner`` is ``-1`` for draws and horizon truncations.
        """

        self._check_open()
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        action_tensor = self.torch.as_tensor(
            actions, dtype=self.torch.int32, device=self.device
        )
        if tuple(action_tensor.shape) != (self.num_envs, 4):
            raise ValueError(f"actions must have shape ({self.num_envs}, 4)")
        if not action_tensor.is_contiguous():
            action_tensor = action_tensor.contiguous()

        obs = self.torch.empty(
            (self.num_envs, 4, *self.obs_shape),
            dtype=self.torch.float32,
            device=self.device,
        )
        done = self.torch.empty(
            (self.num_envs,), dtype=self.torch.bool, device=self.device
        )
        legal = self.torch.empty(
            (self.num_envs, 4, 4), dtype=self.torch.bool, device=self.device
        )
        winner = self.torch.empty(
            (self.num_envs,), dtype=self.torch.int32, device=self.device
        )
        alive = self.torch.empty(
            (self.num_envs, 4), dtype=self.torch.bool, device=self.device
        )
        turns = self.torch.empty(
            (self.num_envs,), dtype=self.torch.int32, device=self.device
        )
        code = CPP_LIB.lib.cuda_blackout_vec_step_all_eval_device_cpp(
            self._env_p,
            action_tensor.data_ptr(),
            max_turns,
            obs.data_ptr(),
            done.data_ptr(),
            legal.data_ptr(),
            winner.data_ptr(),
            alive.data_ptr(),
            turns.data_ptr(),
        )
        if code != 0:
            raise RuntimeError(cuda_last_error())
        return CudaBlackoutTorchEvalStep(
            obs=obs,
            done=done,
            legal_mask=legal,
            winner=winner,
            alive=alive,
            turns=turns,
        )

    def best_actions(self, selection_mask=None):
        """Return tactical scripted-agent actions for all four snakes.

        The result is an ``int32`` CUDA tensor shaped ``(num_envs, 4)`` in the
        native Hisss action order ``UP, RIGHT, DOWN, LEFT``.  Each live snake's
        action is legal in the current state.  The policy runs entirely in the
        CUDA backend and combines reachable space, open exits, food/health
        pressure, wall clearance, and head-to-head risk.  If ``selection_mask``
        is provided, it must be boolean with shape ``(num_envs, 4)``; only
        selected snakes are evaluated and unselected output entries are zero.
        """

        self._check_open()
        selection_ptr = 0
        if selection_mask is not None:
            selection = self.torch.as_tensor(
                selection_mask,
                dtype=self.torch.bool,
                device=self.device,
            )
            if tuple(selection.shape) != (self.num_envs, 4):
                raise ValueError(
                    f"selection_mask must have shape ({self.num_envs}, 4)"
                )
            if not selection.is_contiguous():
                selection = selection.contiguous()
            selection_ptr = selection.data_ptr()
        actions = self.torch.empty(
            (self.num_envs, 4),
            dtype=self.torch.int32,
            device=self.device,
        )
        stream_ptr = self.torch.cuda.current_stream(self.device).cuda_stream
        code = CPP_LIB.lib.cuda_blackout_vec_best_actions_device_async_cpp(
            self._env_p,
            selection_ptr,
            actions.data_ptr(),
            stream_ptr,
        )
        if code != 0:
            raise RuntimeError(cuda_last_error())
        return actions

    def heuristic_actions(self, profile_ids):
        """Evaluate log-derived scripted opponents in native CUDA.

        ``profile_ids`` is an ``int32`` tensor shaped ``(num_envs, 4)``. Values
        0..8 select one of the exported ``BLACKOUT_HEURISTIC_*`` profiles;
        -1 skips a seat and leaves its returned action at zero. The result uses
        native action order ``UP, RIGHT, DOWN, LEFT`` and never leaves the GPU.
        """

        self._check_open()
        profiles = self.torch.as_tensor(profile_ids, dtype=self.torch.int32)
        expected_shape = (self.num_envs, 4)
        if tuple(profiles.shape) != expected_shape:
            raise ValueError(f"profile_ids must have shape {expected_shape}")
        # CPU inputs can be checked without stalling the CUDA stream. Native
        # code treats invalid ids from already-device-resident hot-path inputs
        # as unselected, avoiding a synchronization on every rollout step.
        if profiles.device.type == "cpu" and bool(
            ((profiles < -1) | (profiles >= BLACKOUT_HEURISTIC_COUNT)).any()
        ):
            raise ValueError(
                "profile_ids entries must be -1 or in "
                f"[0, {BLACKOUT_HEURISTIC_COUNT - 1}]"
            )
        profiles = profiles.to(device=self.device)
        if not profiles.is_contiguous():
            profiles = profiles.contiguous()
        actions = self.torch.empty(
            expected_shape,
            dtype=self.torch.int32,
            device=self.device,
        )
        stream_ptr = self.torch.cuda.current_stream(self.device).cuda_stream
        code = CPP_LIB.lib.cuda_blackout_vec_heuristic_actions_device_async_cpp(
            self._env_p,
            profiles.data_ptr(),
            actions.data_ptr(),
            stream_ptr,
        )
        if code != 0:
            raise RuntimeError(cuda_last_error())
        return actions

    def close(self):
        if not self._closed:
            CPP_LIB.lib.cuda_blackout_vec_close_cpp(self._env_p)
            self._closed = True

    def _check_open(self):
        if self._closed:
            raise ValueError("Cannot use closed CudaBlackoutTorchVecEnv")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _null_int_p():
    return ct.cast(0, ct.POINTER(ct.c_int))


def _null_bool_p():
    return ct.cast(0, ct.POINTER(ct.c_bool))


def _int_p(arr: np.ndarray | None):
    if arr is None:
        return _null_int_p()
    return arr.ctypes.data_as(ct.POINTER(ct.c_int))


def _bool_p(arr: np.ndarray | None):
    if arr is None:
        return _null_bool_p()
    return arr.ctypes.data_as(ct.POINTER(ct.c_bool))


def run_random_rollouts_cuda(
    cfg: BattleSnakeConfig,
    num_games: int,
    max_turns: int,
    seed: int = 0,
) -> CudaRolloutResult:
    """Run complete Battlesnake games with a placeholder random agent on the GPU.

    The CUDA backend samples legal moves and applies the game rules inside one
    kernel launch. Python only prepares the initial configuration and receives
    final summary arrays.
    """

    if num_games <= 0:
        raise ValueError("num_games must be positive")
    if max_turns < 0:
        raise ValueError("max_turns must be non-negative")
    if not cuda_available():
        raise RuntimeError(cuda_last_error())

    cfg = copy.deepcopy(cfg)
    post_init_battlesnake_cfg(cfg)
    validate_battlesnake_cfg(cfg)

    spawn_snakes_randomly = cfg.init_snake_pos is None
    body_len_arr: np.ndarray | None
    snake_pos_arr: np.ndarray | None
    max_init_body_len: int

    if cfg.init_snake_pos is None:
        body_len_arr = None
        snake_pos_arr = None
        max_init_body_len = 0
    else:
        snake_pos: dict[int, list[tuple[int, int]]] = {}
        body_lengths: list[int] = []
        for s in range(cfg.num_players):
            cur_snake_pos = []
            for pos in cfg.init_snake_pos[s]:
                cur_snake_pos.append((pos[0], pos[1]))
            cur_snake_pos = list(dict.fromkeys(cur_snake_pos))
            snake_pos[s] = cur_snake_pos
            body_lengths.append(len(cur_snake_pos))
        max_init_body_len = max(1, max(body_lengths))
        body_len_arr = np.asarray(body_lengths, dtype=ct.c_int)
        snake_pos_arr = (
            np.zeros((cfg.num_players, max_init_body_len, 2), dtype=ct.c_int) - 1
        )
        for s in range(cfg.num_players):
            for i, pos in enumerate(snake_pos[s]):
                snake_pos_arr[s, i, 0] = pos[0]
                snake_pos_arr[s, i, 1] = pos[1]

    if cfg.init_food_pos is None:
        num_init_food = -1
        food_pos_arr = None
    elif not cfg.init_food_pos:
        num_init_food = -2
        food_pos_arr = None
    else:
        num_init_food = len(cfg.init_food_pos)
        food_pos_arr = np.asarray(cfg.init_food_pos, dtype=ct.c_int)

    snake_len_arr = np.asarray(cfg.init_snake_len, dtype=ct.c_int)
    snake_alive_arr = np.asarray(cfg.init_snakes_alive, dtype=bool)
    snake_health_arr = np.asarray(cfg.init_snake_health, dtype=ct.c_int)
    snake_max_health_arr = np.asarray(cfg.max_snake_health, dtype=ct.c_int)

    hazard_arr = np.zeros((cfg.h, cfg.w), dtype=bool)
    if cfg.init_hazards is not None:
        for hazard_tile in cfg.init_hazards:
            hazard_arr[hazard_tile[1], hazard_tile[0]] = True

    turns_played = np.zeros((num_games,), dtype=ct.c_int)
    terminal = np.zeros((num_games,), dtype=bool)
    winner = np.full((num_games,), -1, dtype=ct.c_int)
    alive = np.zeros((num_games, cfg.num_players), dtype=bool)
    lengths = np.zeros((num_games, cfg.num_players), dtype=ct.c_int)
    health = np.zeros((num_games, cfg.num_players), dtype=ct.c_int)
    death_cause = np.zeros((num_games, cfg.num_players), dtype=ct.c_int)
    death_turn = np.zeros((num_games, cfg.num_players), dtype=ct.c_int)
    killer_id = np.zeros((num_games, cfg.num_players), dtype=ct.c_int)

    result_code = CPP_LIB.lib.cuda_run_random_rollouts_cpp(
        cfg.w,
        cfg.h,
        cfg.num_players,
        cfg.min_food,
        cfg.food_spawn_chance,
        cfg.init_turns_played,
        spawn_snakes_randomly,
        _int_p(body_len_arr),
        max_init_body_len,
        _int_p(snake_pos_arr),
        num_init_food,
        _int_p(food_pos_arr),
        _null_int_p(),
        _bool_p(snake_alive_arr),
        _int_p(snake_health_arr),
        _int_p(snake_len_arr),
        _int_p(snake_max_health_arr),
        cfg.wrapped,
        cfg.royale,
        cfg.shrink_n_turns,
        cfg.hazard_damage,
        _bool_p(hazard_arr),
        num_games,
        max_turns,
        seed & ((1 << 64) - 1),
        _int_p(turns_played),
        _bool_p(terminal),
        _int_p(winner),
        _bool_p(alive),
        _int_p(lengths),
        _int_p(health),
        _int_p(death_cause),
        _int_p(death_turn),
        _int_p(killer_id),
    )
    if result_code != 0:
        raise RuntimeError(cuda_last_error())

    return CudaRolloutResult(
        turns_played=turns_played,
        terminal=terminal,
        winner=winner,
        alive=alive,
        lengths=lengths,
        health=health,
        death_cause=death_cause,
        death_turn=death_turn,
        killer_id=killer_id,
    )
