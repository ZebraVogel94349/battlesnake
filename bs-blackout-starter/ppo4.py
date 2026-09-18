from __future__ import annotations

import hashlib
import math
import os
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import hisss
import numpy as np
import torch

from battlesnake_types import BaseAgent, Direction, GameState, MoveAction, Point
from obs_config import OBS_SHAPE, build_model


MODEL_ACTIONS = (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT)
ACTION_DELTAS = tuple(direction.board_delta for direction in MODEL_ACTIONS)
MODEL_TO_ENGINE = (hisss.UP, hisss.DOWN, hisss.LEFT, hisss.RIGHT)
ENGINE_TO_MODEL = {
    hisss.UP: 0,
    hisss.DOWN: 1,
    hisss.LEFT: 2,
    hisss.RIGHT: 3,
}
OBS_CENTER = 14
MAX_HEALTH = 100
DEFAULT_MODEL_PATH = Path(__file__).with_name(
    "ppo_bs_lstm_cuda_v25_targeted_finish_champion.zip"
)
_DEFAULT_MODEL = object()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("", "0", "false", "no", "off")


def _valid_point(point: Point | None, width: int, height: int) -> bool:
    return point is not None and 0 <= point.x < width and 0 <= point.y < height


def _stable_body(
    body: Iterable[Point | None], width: int, height: int
) -> tuple[tuple[int, int], ...]:
    """Return the in-bounds body with duplicate coordinates removed stably.

    Hisss represents the initially stacked length-three snake with one physical
    coordinate.  Its native state restoration also removes duplicate cells, so
    doing the same here is required for observation parity.
    """

    result: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for point in body:
        if not _valid_point(point, width, height):
            continue
        cell = (point.x, point.y)
        if cell not in seen:
            seen.add(cell)
            result.append(cell)
    return tuple(result)


def encode_observation(game_state: GameState) -> np.ndarray:
    """Encode a Blackout API request directly into the actor's nine channels.

    Unlike the old reconstructed ``BattleSnakeGame``, this preserves food that
    spawned globally on the current turn.  Hisss' JSON exporter/the tournament
    server already applies fog to old food and enemy segments, so every valid
    coordinate present in the request is observable information.
    """

    width = game_state.board.width
    height = game_state.board.height
    if (width, height) != (15, 15):
        raise ValueError(f"expected a 15x15 Blackout board, got {width}x{height}")
    head = game_state.you.head
    if not _valid_point(head, width, height):
        return np.zeros(OBS_SHAPE, dtype=np.float32)

    offset_x = OBS_CENTER - head.x
    offset_y = OBS_CENTER - head.y
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)

    # Food supplied by Blackout is either in view or globally announced on its
    # spawn turn.  Both kinds must reach the recurrent policy.
    for food in game_state.board.food:
        if _valid_point(food, width, height):
            obs[0, food.x + offset_x, food.y + offset_y] = 1.0

    # The centered board channel is +1 in bounds and -1 outside.
    obs[1].fill(-1.0)
    obs[
        1,
        offset_x : offset_x + width,
        offset_y : offset_y + height,
    ] = 1.0

    own_body = _stable_body(game_state.you.body, width, height)
    own_length = game_state.you.length
    for index, (x, y) in enumerate(own_body):
        obs[2, x + offset_x, y + offset_y] = (own_length - index) / 10.0

    obs[3, OBS_CENTER, OBS_CENTER] = 1.0
    obs[4].fill((game_state.you.health or 0) / MAX_HEALTH)
    if own_body:
        tail_x, tail_y = own_body[-1]
        obs[5, tail_x + offset_x, tail_y + offset_y] = 1.0

    for snake in game_state.board.snakes:
        if snake.id == game_state.you.id:
            continue
        for x, y in _stable_body(snake.body, width, height):
            obs[6, x + offset_x, y + offset_y] = 1.0
        if _valid_point(snake.head, width, height):
            obs[7, snake.head.x + offset_x, snake.head.y + offset_y] = 1.0

    view_radius = game_state.game.ruleset.settings.viewRadius
    if view_radius is None:
        obs[8].fill(1.0)
    else:
        # Clamp only to the tensor boundary. A radius larger than the centered
        # observation still correctly fills the entire visible plane.
        for dx in range(-OBS_CENTER, OBS_CENTER + 1):
            remaining = view_radius - abs(dx)
            if remaining < 0:
                continue
            remaining = min(remaining, OBS_CENTER)
            obs[
                8,
                OBS_CENTER + dx,
                OBS_CENTER - remaining : OBS_CENTER + remaining + 1,
            ] = 1.0
    return obs


def _d4_action_permutations() -> np.ndarray:
    """Original-model action -> transformed-model action for Hisss' D4 order."""

    result = np.empty((8, 4), dtype=np.int64)
    for symmetry in range(8):
        flip = symmetry % 2 == 1
        rotations = symmetry // 2
        for model_action, engine_action in enumerate(MODEL_TO_ENGINE):
            transformed = (engine_action - rotations) % 4
            if flip:
                if transformed == hisss.UP:
                    transformed = hisss.DOWN
                elif transformed == hisss.DOWN:
                    transformed = hisss.UP
            result[symmetry, model_action] = ENGINE_TO_MODEL[transformed]
    return result


D4_ACTION_PERMUTATIONS = _d4_action_permutations()
D4_SYMMETRY_INDICES = {
    1: (0,),
    2: (0, 4),
    4: (0, 2, 4, 6),
    8: tuple(range(8)),
}


def transform_observations(obs: np.ndarray, count: int) -> np.ndarray:
    """Return identity, then the first ``count-1`` Hisss D4 transforms."""

    if count not in (1, 2, 4, 8):
        raise ValueError("symmetries must be one of 1, 2, 4, or 8")
    transformed = []
    for symmetry in D4_SYMMETRY_INDICES[count]:
        flip = symmetry % 2 == 1
        rotations = symmetry // 2
        cur = np.rot90(obs, k=rotations, axes=(-2, -1))
        if flip:
            cur = np.flip(cur, axis=-1)
        transformed.append(np.ascontiguousarray(cur))
    return np.stack(transformed)


@dataclass
class _Session:
    lstm_states: tuple[torch.Tensor, torch.Tensor] | None = None
    episode_start: bool = True
    last_turn: int = -1
    last_move: MoveAction | None = None


@dataclass(frozen=True)
class MoveGeometry:
    action: int
    target: tuple[int, int]
    area: int
    exits: int
    tail_reachable: bool
    survival_depth: int = 0
    search_complete: bool = True


class _SearchBudgetExhausted(Exception):
    pass


class PPOAgent4(BaseAgent):
    """Recurrent PPO inference with exact encoding and tactical safety layers.

    The neural policy still determines strategy. Geometry only removes known
    deaths and roots of short forced self-traps. Optional D4 test-time
    augmentation is batched in one policy call, with an independent recurrent
    state for every board symmetry.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None | object = _DEFAULT_MODEL,
        *,
        symmetries: int | None = None,
        safety_search: bool | None = None,
        search_horizon: int | None = None,
        search_node_budget: int | None = None,
        device: str | None = None,
    ):
        if model_path is _DEFAULT_MODEL:
            model_path = os.environ.get("PPO_MODEL_PATH", str(DEFAULT_MODEL_PATH))
        if symmetries is None:
            # Eight-way TTA is retained as an evaluated experiment, but the
            # current champion is materially stronger with its identity policy.
            symmetries = int(os.environ.get("PPO_SYMMETRIES", "1"))
        if symmetries not in (1, 2, 4, 8):
            raise ValueError("symmetries must be one of 1, 2, 4, or 8")
        if safety_search is None:
            safety_search = _env_bool("PPO_SAFETY_SEARCH", True)
        if search_horizon is None:
            search_horizon = int(os.environ.get("PPO_SEARCH_HORIZON", "10"))
        if search_node_budget is None:
            search_node_budget = int(os.environ.get("PPO_SEARCH_NODE_BUDGET", "6000"))
        if search_horizon < 1:
            raise ValueError("search_horizon must be positive")
        if search_node_budget < 1:
            raise ValueError("search_node_budget must be positive")

        self.symmetries = symmetries
        self.safety_search = safety_search
        self.search_horizon = search_horizon
        self.search_node_budget = search_node_budget
        self.device = device or os.environ.get("PPO_DEVICE", "cpu")
        self.model_path = None if model_path is None else Path(model_path).resolve()
        self.model = build_model(device=self.device)
        if self.model_path is not None:
            self.model.set_parameters(str(self.model_path), device=self.device)
        self.model.policy.set_training_mode(False)

        self._sessions: dict[tuple[str, str], _Session] = {}
        self._lock = threading.RLock()
        self._model_fingerprint = self._fingerprint(self.model_path)
        self._code_fingerprint = self._fingerprint(Path(__file__).resolve())

    @staticmethod
    def _fingerprint(path: Path | None) -> str:
        if path is None or not path.is_file():
            return "random"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()[:12]

    @staticmethod
    def _session_key(game_state: GameState) -> tuple[str, str]:
        return game_state.game.id, game_state.you.id

    def get_name(self):
        return "Der Snaketürke v4"

    def get_color(self):
        return "#b0e000"

    def get_author(self):
        return "Colin"

    def get_diagnostics(self) -> dict[str, object]:
        return {
            "agent": type(self).__name__,
            "model_sha256": self._model_fingerprint,
            "code_sha256": self._code_fingerprint,
            "symmetries": self.symmetries,
            "safety_search": self.safety_search,
            "search_horizon": self.search_horizon,
            "search_node_budget": self.search_node_budget,
        }

    def start(self, game_state: GameState):
        with self._lock:
            self._sessions[self._session_key(game_state)] = _Session()

    def end(self, game_state: GameState):
        with self._lock:
            self._sessions.pop(self._session_key(game_state), None)

    def _policy_logits(self, obs: np.ndarray, session: _Session) -> np.ndarray:
        batch = transform_observations(obs, self.symmetries)
        obs_tensor, _ = self.model.policy.obs_to_tensor(batch)
        policy = self.model.policy
        if session.lstm_states is None:
            layers = policy.lstm_actor.num_layers
            hidden = policy.lstm_actor.hidden_size
            states = (
                torch.zeros(
                    layers,
                    self.symmetries,
                    hidden,
                    device=policy.device,
                ),
                torch.zeros(
                    layers,
                    self.symmetries,
                    hidden,
                    device=policy.device,
                ),
            )
        else:
            states = session.lstm_states
        starts = torch.full(
            (self.symmetries,),
            1.0 if session.episode_start else 0.0,
            dtype=torch.float32,
            device=policy.device,
        )
        with torch.inference_mode():
            distribution, new_states = policy.get_distribution(obs_tensor, states, starts)
            transformed_log_probs = distribution.distribution.logits
            action_permutations = torch.as_tensor(
                D4_ACTION_PERMUTATIONS[list(D4_SYMMETRY_INDICES[self.symmetries])],
                dtype=torch.long,
                device=policy.device,
            )
            original_log_probs = transformed_log_probs.gather(1, action_permutations)
            # Average probabilities, not raw logits: every transformed row is a
            # separately normalized policy distribution.
            mixed = torch.logsumexp(original_log_probs, dim=0) - math.log(
                self.symmetries
            )
        session.lstm_states = new_states
        session.episode_start = False
        return mixed.detach().cpu().numpy()

    @staticmethod
    def _food_cells(game_state: GameState) -> set[tuple[int, int]]:
        width, height = game_state.board.width, game_state.board.height
        return {
            (food.x, food.y)
            for food in game_state.board.food
            if _valid_point(food, width, height)
        }

    @staticmethod
    def _hazard_cells(game_state: GameState) -> set[tuple[int, int]]:
        width, height = game_state.board.width, game_state.board.height
        return {
            (point.x, point.y)
            for point in game_state.board.hazards
            if _valid_point(point, width, height)
        }

    def _action_tiers(self, game_state: GameState) -> tuple[list[int], list[int]]:
        """Return certain-survival and non-losing-head-contest action tiers."""

        width, height = game_state.board.width, game_state.board.height
        head = game_state.you.head
        if not _valid_point(head, width, height):
            return list(range(4)), list(range(4))

        own_body = _stable_body(game_state.you.body, width, height)
        # A tail only vacates if the physical body has caught up with its target
        # length.  This fixes the post-food/initial-growth tail exception.
        tail_vacates = len(own_body) >= game_state.you.length
        own_blocked = set(own_body[:-1] if tail_vacates else own_body)

        enemy_blocked: set[tuple[int, int]] = set()
        contested: set[tuple[int, int]] = set()
        own_length = game_state.you.length
        fogged_opponent_lengths = (
            game_state.game.ruleset.settings.viewRadius is not None
        )
        for snake in game_state.board.snakes:
            if snake.id == game_state.you.id:
                continue
            enemy_blocked.update(_stable_body(snake.body, width, height))
            if not _valid_point(snake.head, width, height):
                continue
            has_hidden_body = any(
                not _valid_point(point, width, height) for point in snake.body
            )
            # Hisss' Blackout export reports the restricted physical body-list
            # length for opponents.  Even without a hidden sentinel this can be
            # smaller than logical length while initial/food growth is pending
            # (one visible cell versus logical length three).  Never classify a
            # fog-game head contest as a guaranteed win from that value alone.
            enemy_length_unknown = fogged_opponent_lengths or has_hidden_body
            if enemy_length_unknown or snake.length >= own_length:
                for dx, dy in ACTION_DELTAS:
                    target = (snake.head.x + dx, snake.head.y + dy)
                    if 0 <= target[0] < width and 0 <= target[1] < height:
                        contested.add(target)

        food = self._food_cells(game_state)
        hazards = self._hazard_cells(game_state)
        health = game_state.you.health or 0
        hazard_damage = game_state.game.ruleset.settings.hazardDamagePerTurn

        hard: list[int] = []
        soft: list[int] = []
        for action, (dx, dy) in enumerate(ACTION_DELTAS):
            target = (head.x + dx, head.y + dy)
            if not (0 <= target[0] < width and 0 <= target[1] < height):
                continue
            if target in own_blocked or target in enemy_blocked:
                continue
            eats = target in food
            if not eats and health <= 1:
                continue
            if not eats and target in hazards and health <= hazard_damage + 1:
                continue
            hard.append(action)
            if target not in contested:
                soft.append(action)
        return hard, soft

    @staticmethod
    def _flood(
        start: tuple[int, int],
        blocked: set[tuple[int, int]],
        width: int,
        height: int,
    ) -> set[tuple[int, int]]:
        if start in blocked:
            return set()
        seen = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ACTION_DELTAS:
                nxt = x + dx, y + dy
                if (
                    0 <= nxt[0] < width
                    and 0 <= nxt[1] < height
                    and nxt not in blocked
                    and nxt not in seen
                ):
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    def _move_geometry(self, game_state: GameState, action: int) -> MoveGeometry:
        width, height = game_state.board.width, game_state.board.height
        head = game_state.you.head
        dx, dy = ACTION_DELTAS[action]
        target = head.x + dx, head.y + dy
        own_body = _stable_body(game_state.you.body, width, height)
        foods = self._food_cells(game_state)

        new_body = (target,) + own_body
        if len(new_body) > game_state.you.length:
            new_body = new_body[:-1]
        new_length = game_state.you.length + int(target in foods)

        # The next tail is traversable only when it will actually vacate.
        tail_vacates_next = len(new_body) >= new_length
        own_blocked = set(new_body[1:-1] if tail_vacates_next else new_body[1:])
        enemy_blocked: set[tuple[int, int]] = set()
        for snake in game_state.board.snakes:
            if snake.id != game_state.you.id:
                enemy_blocked.update(_stable_body(snake.body, width, height))
        reachable = self._flood(target, own_blocked | enemy_blocked, width, height)
        exits = 0
        for ex, ey in ACTION_DELTAS:
            nxt = target[0] + ex, target[1] + ey
            if (
                0 <= nxt[0] < width
                and 0 <= nxt[1] < height
                and nxt not in own_blocked
                and nxt not in enemy_blocked
            ):
                exits += 1
        tail = new_body[-1] if new_body else target
        return MoveGeometry(
            action=action,
            target=target,
            area=len(reachable),
            exits=exits,
            tail_reachable=tail in reachable,
        )

    def _survival_depth(self, game_state: GameState, first_action: int) -> tuple[int, bool]:
        """Exact own-body DFS; unknown future opponents are deliberately omitted.

        The result is used only as a veto when another root provably survives
        farther.  Exhausting the node budget returns ``complete=False`` and is
        therefore never interpreted as proof of a trap.
        """

        width, height = game_state.board.width, game_state.board.height
        body = _stable_body(game_state.you.body, width, height)
        if not body:
            return self.search_horizon, True
        foods = frozenset(self._food_cells(game_state))
        hazards = frozenset(self._hazard_cells(game_state))
        hazard_damage = game_state.game.ruleset.settings.hazardDamagePerTurn
        nodes_left = [self.search_node_budget]
        memo: dict[
            tuple[tuple[tuple[int, int], ...], int, int, frozenset[tuple[int, int]], int],
            int,
        ] = {}

        def advance(
            cur_body: tuple[tuple[int, int], ...],
            target_length: int,
            health: int,
            cur_foods: frozenset[tuple[int, int]],
            action: int,
        ):
            hx, hy = cur_body[0]
            dx, dy = ACTION_DELTAS[action]
            target = hx + dx, hy + dy
            if not (0 <= target[0] < width and 0 <= target[1] < height):
                return None
            tail_vacates = len(cur_body) >= target_length
            blocked = set(cur_body[:-1] if tail_vacates else cur_body)
            if target in blocked:
                return None

            next_body = (target,) + cur_body
            if len(next_body) > target_length:
                next_body = next_body[:-1]
            eats = target in cur_foods
            next_health = MAX_HEALTH if eats else health - 1
            if not eats and target in hazards:
                next_health -= hazard_damage
            if next_health <= 0:
                return None
            next_length = target_length + int(eats)
            next_foods = cur_foods - {target} if eats else cur_foods
            return next_body, next_length, next_health, next_foods

        def dfs(
            cur_body: tuple[tuple[int, int], ...],
            target_length: int,
            health: int,
            cur_foods: frozenset[tuple[int, int]],
            remaining: int,
        ) -> int:
            if remaining == 0:
                return self.search_horizon
            nodes_left[0] -= 1
            if nodes_left[0] < 0:
                raise _SearchBudgetExhausted
            key = (cur_body, target_length, min(health, remaining + 1), cur_foods, remaining)
            if key in memo:
                return memo[key]

            candidates = []
            for action in range(4):
                nxt = advance(cur_body, target_length, health, cur_foods, action)
                if nxt is None:
                    continue
                next_body = nxt[0]
                hx, hy = next_body[0]
                blocked = set(next_body[1:-1])
                degree = sum(
                    0 <= hx + dx < width
                    and 0 <= hy + dy < height
                    and (hx + dx, hy + dy) not in blocked
                    for dx, dy in ACTION_DELTAS
                )
                candidates.append((degree, nxt))
            candidates.sort(key=lambda item: item[0], reverse=True)

            best = self.search_horizon - remaining
            for _, nxt in candidates:
                reached = dfs(*nxt, remaining - 1)
                best = max(best, reached)
                if best >= self.search_horizon:
                    break
            memo[key] = best
            return best

        first = advance(
            body,
            game_state.you.length,
            game_state.you.health or 0,
            foods,
            first_action,
        )
        if first is None:
            return 0, True
        try:
            return dfs(*first, self.search_horizon - 1), True
        except _SearchBudgetExhausted:
            return self.search_horizon, False

    def _safe_candidates(
        self,
        game_state: GameState,
        hard: list[int],
        soft: list[int],
        policy_action: int,
    ) -> tuple[list[int], dict[int, MoveGeometry]]:
        # Never prefer a roomy head-to-head loss over a smaller uncontested move.
        candidates = list(soft if soft else hard)
        geometry = {
            action: self._move_geometry(game_state, action) for action in candidates
        }
        if not self.safety_search or len(hard) <= 1:
            return candidates, geometry

        def search(action: int) -> MoveGeometry:
            info = geometry.get(action)
            if info is None:
                info = self._move_geometry(game_state, action)
                geometry[action] = info
            depth, complete = self._survival_depth(game_state, action)
            result = MoveGeometry(
                action=info.action,
                target=info.target,
                area=info.area,
                exits=info.exits,
                tail_reachable=info.tail_reachable,
                survival_depth=depth,
                search_complete=complete,
            )
            geometry[action] = result
            return result

        # Preserve the learned policy whenever it has a path to the requested
        # horizon. Budget exhaustion is UNKNOWN, never proof of a trap.
        policy_info = search(policy_action)
        if (
            not policy_info.search_complete
            or policy_info.survival_depth >= self.search_horizon
        ):
            return candidates, geometry

        for action in candidates:
            if action != policy_action:
                search(action)
        escapes = [
            action
            for action in candidates
            if not geometry[action].search_complete
            or geometry[action].survival_depth >= self.search_horizon
        ]
        if escapes:
            return escapes, geometry

        # If every uncontested move is a proven self-trap, a contested hard move
        # is preferable when it might survive. This fixes ppo3's soft-before-space
        # ordering without treating uncertain head-to-heads as guaranteed wins.
        if soft:
            hard_extras = [action for action in hard if action not in soft]
            for action in hard_extras:
                search(action)
            hard_escapes = [
                action
                for action in hard_extras
                if not geometry[action].search_complete
                or geometry[action].survival_depth >= self.search_horizon
            ]
            if hard_escapes:
                return hard_escapes, geometry
            candidates.extend(hard_extras)

        best_depth = max(geometry[action].survival_depth for action in candidates)
        # Do not replace one doomed strategic line with a barely longer doomed
        # line; require a material, completely demonstrated survival advantage.
        if best_depth >= policy_info.survival_depth + 4:
            return [
                action
                for action in candidates
                if geometry[action].survival_depth == best_depth
            ], geometry
        return list(soft if soft else hard), geometry

    @staticmethod
    def _argmax_legal(logits: np.ndarray, legal: list[int]) -> int:
        if not legal:
            return int(np.argmax(logits))
        return max(legal, key=lambda action: (float(logits[action]), -action))

    def move(self, game_state: GameState) -> MoveAction:
        with self._lock:
            key = self._session_key(game_state)
            session = self._sessions.setdefault(key, _Session())
            # A retried request must not advance recurrent state a second time.
            if game_state.turn == session.last_turn and session.last_move is not None:
                return session.last_move
            if game_state.turn < session.last_turn:
                raise ValueError(
                    f"out-of-order turn for {key}: {game_state.turn} < {session.last_turn}"
                )
            session.last_turn = game_state.turn

            obs = encode_observation(game_state)
            logits = self._policy_logits(obs, session)
            hard, soft = self._action_tiers(game_state)
            policy_tier = soft if soft else hard
            policy_action = self._argmax_legal(logits, policy_tier)
            candidates, _ = self._safe_candidates(
                game_state, hard, soft, policy_action
            )
            action = self._argmax_legal(logits, candidates)

            # Independent final invariant: no later ranking/search bug may emit
            # a move that the certain-death layer rejected.
            if hard and action not in hard:
                action = self._argmax_legal(logits, hard)
            session.last_move = MoveAction(move=MODEL_ACTIONS[action])
            return session.last_move


if __name__ == "__main__":
    from battlesnake_server import start_server

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <port>")
        raise SystemExit(1)

    start_server(agent=PPOAgent4(), port=int(sys.argv[1]))
