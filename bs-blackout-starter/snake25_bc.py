"""Behavioral cloning utilities for the real tournament Snake 25.

The game logs are recorded from our snake's fog-of-war perspective.  They do
not contain Snake 25's private observation.  This module therefore builds an
honest *proxy* observation: it is centred on Snake 25, while cells that were
not visible in the recorded request remain fogged.  The resulting policy is
used as a frozen opponent, never as a direct replacement for the learner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import hisss
from hisss.game.state import BattleSnakeState

from obs_config import KEPT_CHANNELS, OBS_SHAPE, make_game_config, to_model_obs


DATASET_SCHEMA_VERSION = 1
PROXY_OBSERVATION_VERSION = 1
MODEL_ACTION_BY_DELTA = {
    (0, 1): 0,   # up
    (0, -1): 1,  # down
    (-1, 0): 2,  # left
    (1, 0): 3,   # right
}


def _point(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, dict):
        return None
    try:
        x, y = int(value["x"]), int(value["y"])
    except (KeyError, TypeError, ValueError):
        return None
    return (x, y) if 0 <= x < 15 and 0 <= y < 15 else None


def snake_named(state: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Return the uniquely named snake in one logged request."""

    matches = [
        snake
        for snake in state.get("board", {}).get("snakes", [])
        if str(snake.get("name", "")).strip() == name
    ]
    return matches[0] if len(matches) == 1 else None


def inferred_model_action(
    current_state: dict[str, Any],
    next_state: dict[str, Any],
    snake_name: str = "25",
) -> int | None:
    """Infer Snake 25's action from two consecutive visible head positions."""

    current = snake_named(current_state, snake_name)
    following = snake_named(next_state, snake_name)
    if current is None or following is None:
        return None
    current_head = _point(current.get("head"))
    next_head = _point(following.get("head"))
    if current_head is None or next_head is None:
        return None
    delta = (next_head[0] - current_head[0], next_head[1] - current_head[1])
    return MODEL_ACTION_BY_DELTA.get(delta)


def alive_snake_count(state: dict[str, Any]) -> int:
    """Count live slots in logs that retain eliminated snakes for diagnostics."""

    return sum(
        snake.get("elimination_event") is None
        for snake in state.get("board", {}).get("snakes", [])
    )


def validation_game(game_id: str, validation_fraction: float) -> bool:
    """Stable game-level split, preventing adjacent turns leaking across sets."""

    digest = hashlib.blake2b(game_id.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") / float(1 << 64)
    return bucket < validation_fraction


@dataclass
class ProxyObservation:
    observation: np.ndarray
    coverage: float


class Snake25ProxyEncoder:
    """Reconstruct model observations centred on a visible opponent snake."""

    def __init__(self):
        self.cfg = make_game_config()
        self.env = hisss.BattleSnakeGame(self.cfg)
        self.env.reset()

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    @staticmethod
    def _phantom(head: tuple[int, int]) -> tuple[int, int]:
        return (0 if head[0] >= 8 else 14, 0 if head[1] >= 8 else 14)

    def encode(
        self,
        state: dict[str, Any],
        snake_name: str = "25",
    ) -> ProxyObservation | None:
        board = state.get("board", {})
        if int(board.get("width", 0)) != 15 or int(board.get("height", 0)) != 15:
            return None
        candidate = snake_named(state, snake_name)
        candidate_head = _point(candidate.get("head")) if candidate else None
        recorder_head = _point(state.get("you", {}).get("head"))
        if candidate is None or candidate_head is None or recorder_head is None:
            return None

        snakes = list(board.get("snakes", []))
        others = sorted(
            (snake for snake in snakes if snake.get("id") != candidate.get("id")),
            key=lambda snake: str(snake.get("id", "")),
        )
        ordered = [candidate, *others[:3]]
        phantom = self._phantom(candidate_head)
        snakes_alive: list[bool] = []
        snake_pos: dict[int, list[tuple[int, int]]] = {}
        snake_health: list[int] = []
        snake_len: list[int] = []

        for index in range(4):
            snake = ordered[index] if index < len(ordered) else None
            positions: list[tuple[int, int]] = []
            any_visible = False
            for raw_point in [] if snake is None else snake.get("body", []):
                point = _point(raw_point)
                if point is None:
                    positions.append(phantom)
                else:
                    positions.append(point)
                    any_visible = True
            if not positions:
                positions = [phantom]
            # Keeping four nominally live slots prevents the reconstructed
            # engine from returning a terminal blank observation.  Fog hides
            # every phantom, so this does not expose an alive-count channel.
            snakes_alive.append(True)
            snake_pos[index] = positions
            raw_health = snake.get("health") if snake else None
            snake_health.append(
                int(raw_health) if raw_health is not None and raw_health > 0 else 100
            )
            raw_length = snake.get("length") if snake else None
            visible_length = sum(_point(point) is not None for point in (
                [] if snake is None else snake.get("body", [])
            ))
            snake_len.append(max(1, int(raw_length or 0), visible_length))

        food_pos: list[list[int]] = []
        food_spawn_turns: list[int] = []
        turn = int(state.get("turn", 0))
        for food in board.get("food", []):
            point = _point(food)
            if point is None:
                continue
            food_pos.append([point[0], point[1]])
            food_spawn_turns.append(int(food.get("spawn_turn", -1)))

        reconstructed = BattleSnakeState(
            turn=turn,
            snakes_alive=snakes_alive,
            snake_pos=snake_pos,
            food_pos=food_pos,
            snake_health=snake_health,
            snake_len=snake_len,
            food_spawn_turns=food_spawn_turns,
        )
        self.env.set_state(reconstructed)
        if 0 not in self.env.players_at_turn():
            return None
        encoded_hwc = self.env.get_obs()[0]
        player_index = self.env.players_at_turn().index(0)
        obs = to_model_obs(encoded_hwc[player_index])

        radius = (
            state.get("game", {})
            .get("ruleset", {})
            .get("settings", {})
            .get("viewRadius", 5)
        )
        radius = 5 if radius is None else int(radius)
        coordinates = np.arange(29, dtype=np.int16)
        world_x = coordinates[:, None] - 14 + candidate_head[0]
        world_y = coordinates[None, :] - 14 + candidate_head[1]
        recorded_view = (
            np.abs(world_x - recorder_head[0])
            + np.abs(world_y - recorder_head[1])
            <= radius
        )
        candidate_view = obs[8] > 0.0
        trusted_view = candidate_view & recorded_view
        denominator = max(1, int(candidate_view.sum()))
        coverage = float(trusted_view.sum()) / denominator

        # Opponent/food geometry outside the request's original view is
        # unknown, not empty.  Preserve the board and private health channels,
        # while explicitly fogging all spatially uncertain channels.
        uncertain = ~recorded_view
        for channel in (2, 3, 5, 6, 7):
            obs[channel, uncertain] = 0.0

        new_food = np.zeros((29, 29), dtype=bool)
        for (food_x, food_y), spawn_turn in zip(food_pos, food_spawn_turns):
            if spawn_turn != turn:
                continue
            obs_x = food_x + 14 - candidate_head[0]
            obs_y = food_y + 14 - candidate_head[1]
            if 0 <= obs_x < 29 and 0 <= obs_y < 29:
                new_food[obs_x, obs_y] = True
        obs[0, uncertain & ~new_food] = 0.0
        obs[8] = trusted_view.astype(np.float32)
        return ProxyObservation(obs, coverage)


def iter_unique_logs(
    log_dir: Path,
    *,
    max_files: int | None = None,
) -> Iterable[tuple[Path, dict[str, Any]]]:
    paths = sorted(log_dir.glob("*.json"))
    if max_files is not None:
        paths = paths[-max_files:]
    seen_game_ids: set[str] = set()
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        game_id = str(payload.get("game_id") or path.stem)
        if game_id in seen_game_ids:
            continue
        seen_game_ids.add(game_id)
        yield path, payload


def build_dataset(
    log_dir: str | Path,
    *,
    snake_name: str = "25",
    sequence_length: int = 32,
    validation_fraction: float = 0.15,
    max_files: int | None = None,
) -> dict[str, Any]:
    """Extract consecutive visible Snake-25 actions into recurrent sequences."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    root = Path(log_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"game log directory does not exist: {root}")

    train_sequences: list[dict[str, Any]] = []
    validation_sequences: list[dict[str, Any]] = []
    game_count = transition_count = duel_transition_count = 0

    with Snake25ProxyEncoder() as encoder:
        for _, payload in iter_unique_logs(root, max_files=max_files):
            moves = payload.get("moves", [])
            if len(moves) < 2:
                continue
            game_id = str(payload.get("game_id") or "unknown")
            game_sequences: list[dict[str, Any]] = []
            run_obs: list[torch.Tensor] = []
            run_actions: list[int] = []
            run_duels: list[bool] = []
            run_coverage: list[float] = []

            def flush_run() -> None:
                nonlocal run_obs, run_actions, run_duels, run_coverage
                for start in range(0, len(run_actions), sequence_length):
                    stop = min(start + sequence_length, len(run_actions))
                    if stop <= start:
                        continue
                    game_sequences.append(
                        {
                            "game_id": game_id,
                            "obs": torch.stack(run_obs[start:stop]),
                            "actions": torch.tensor(
                                run_actions[start:stop], dtype=torch.int64
                            ),
                            "duel": torch.tensor(
                                run_duels[start:stop], dtype=torch.bool
                            ),
                            "coverage": torch.tensor(
                                run_coverage[start:stop], dtype=torch.float32
                            ),
                        }
                    )
                run_obs, run_actions, run_duels, run_coverage = [], [], [], []

            previous_turn: int | None = None
            for current_move, next_move in zip(moves, moves[1:]):
                current_state = current_move.get("state", {})
                next_state = next_move.get("state", {})
                current_turn = int(current_move.get("turn", current_state.get("turn", -1)))
                next_turn = int(next_move.get("turn", next_state.get("turn", -1)))
                action = (
                    inferred_model_action(current_state, next_state, snake_name)
                    if next_turn == current_turn + 1
                    else None
                )
                proxy = encoder.encode(current_state, snake_name) if action is not None else None
                if (
                    action is None
                    or proxy is None
                    or (previous_turn is not None and current_turn != previous_turn + 1)
                ):
                    flush_run()
                if action is not None and proxy is not None:
                    run_obs.append(torch.from_numpy(proxy.observation).to(torch.float16))
                    run_actions.append(action)
                    is_duel = alive_snake_count(current_state) == 2
                    run_duels.append(is_duel)
                    run_coverage.append(proxy.coverage)
                    transition_count += 1
                    duel_transition_count += int(is_duel)
                    previous_turn = current_turn
                else:
                    previous_turn = None
            flush_run()
            if not game_sequences:
                continue
            game_count += 1
            destination = (
                validation_sequences
                if validation_game(game_id, validation_fraction)
                else train_sequences
            )
            destination.extend(game_sequences)

    if not train_sequences or not validation_sequences:
        raise ValueError(
            "behavioral-cloning extraction needs both train and validation games"
        )
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "proxy_observation_version": PROXY_OBSERVATION_VERSION,
        "snake_name": snake_name,
        "obs_shape": OBS_SHAPE,
        "channels": list(KEPT_CHANNELS),
        "sequence_length": sequence_length,
        "validation_fraction": validation_fraction,
        "max_files": max_files,
        "source_log_dir": str(root.resolve()),
        "game_count": game_count,
        "transition_count": transition_count,
        "duel_transition_count": duel_transition_count,
        "train": train_sequences,
        "validation": validation_sequences,
    }


def save_dataset(dataset: dict[str, Any], path: str | Path) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(dataset, temporary)
    temporary.replace(target)
    return target


def load_dataset(path: str | Path) -> dict[str, Any]:
    dataset = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if dataset.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("incompatible Snake-25 BC dataset schema")
    if tuple(dataset.get("obs_shape", ())) != OBS_SHAPE:
        raise ValueError("Snake-25 BC dataset has the wrong observation shape")
    if not dataset.get("train") or not dataset.get("validation"):
        raise ValueError("Snake-25 BC dataset has an empty split")
    return dataset
