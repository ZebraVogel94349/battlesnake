from dataclasses import dataclass
import heapq
import itertools
import time
from typing import List, Tuple, Dict
import numpy as np

from battlesnake_types import GameState, MoveAction, Direction, BaseAgent, Point

# ---------------------------------------------------------
# Behavior Constants
# ---------------------------------------------------------
DEFAULT_START_DIRECTION = Direction.RIGHT
UNREACHABLE_DISTANCE = 9999999

BASE_STRATEGY_WEIGHT = 1.0

FOOD_MISSING_WEIGHT_MULTIPLIER = 0.25
EAT_HEALTH_LOW_THRESHOLD = 40
EAT_HEALTH_MEDIUM_THRESHOLD = 70
EAT_CLOSE_FOOD_DISTANCE = 2
EAT_NEAR_FOOD_DISTANCE = 5
EAT_HEALTH_LOW_MULTIPLIER = 4.5
EAT_HEALTH_MEDIUM_MULTIPLIER = 3.5
EAT_CLOSE_FOOD_MULTIPLIER = 3.0
EAT_NEAR_FOOD_MULTIPLIER = 1.3
EAT_CRITICAL_HEALTH_THRESHOLD = 25
EAT_CRITICAL_HEALTH_MULTIPLIER = 1.5
EAT_REACHABLE_SPACE_WEIGHT = 0.02
EAT_HEALTH_BIAS_BASE = 60
EAT_HEALTH_BIAS_SCALE = 40.0

CHILL_REACHABLE_SPACE_WEIGHT = 1.0
CHILL_OPEN_EXITS_WEIGHT = 1.8
CHILL_WALL_DISTANCE_WEIGHT = 0.5
CHILL_OPEN_SPACE_RATIO_THRESHOLD = 0.35
CHILL_OPEN_SPACE_MULTIPLIER = 1.4
CHILL_DENSE_SPACE_RATIO_THRESHOLD = 0.18
CHILL_DENSE_SPACE_MULTIPLIER = 1.6

AVOID_STUCK_REACHABLE_SPACE_WEIGHT = 1.2
AVOID_STUCK_OPEN_EXITS_WEIGHT = 2.0
AVOID_STUCK_WALL_DISTANCE_WEIGHT = 0.25
AVOID_STUCK_DEAD_END_THRESHOLD = 1
AVOID_STUCK_CORRIDOR_THRESHOLD = 2
AVOID_STUCK_DEAD_END_MULTIPLIER = 0.2
AVOID_STUCK_CORRIDOR_MULTIPLIER = 0.7
AVOID_STUCK_TIGHT_SPACE_RATIO = 0.22
AVOID_STUCK_TIGHT_SPACE_MULTIPLIER = 2.6
AVOID_STUCK_MEDIUM_SPACE_RATIO = 0.35
AVOID_STUCK_MEDIUM_SPACE_MULTIPLIER = 1.5

FLEE_CLOSE_DISTANCE = 2
FLEE_NEAR_DISTANCE = 5
FLEE_MEDIUM_DISTANCE = 8
FLEE_CLOSE_DISTANCE_MULTIPLIER = 3.0
FLEE_NEAR_DISTANCE_MULTIPLIER = 1.8
FLEE_MEDIUM_DISTANCE_MULTIPLIER = 1.2
FLEE_LONGER_SNAKE_MULTIPLIER_STEP = 0.3
FLEE_REACHABLE_SPACE_WEIGHT = 0.02
FLEE_WALL_DISTANCE_WEIGHT = 0.2

LOW_SPACE_BODY_FACTOR = 2
LOW_SPACE_SELF_BODY_MULTIPLIER = 2.5
MID_SPACE_SELF_BODY_MULTIPLIER = 1.5
HIGH_SPACE_RATIO = 0.35
HIGH_SPACE_CHILL_MULTIPLIER = 1.4
LOW_HEALTH_AGGRESSION_THRESHOLD = 25
LOW_HEALTH_CHILL_MULTIPLIER = 0.7

SAFE_MOVE_SPACE_WEIGHT = 1.0
SAFE_MOVE_WALL_DISTANCE_WEIGHT = 0.5
FALLBACK_OPEN_SPACE_WEIGHT = 1.0
FALLBACK_WALL_DISTANCE_WEIGHT = 0.5

TACTICAL_HEAD_RISK_MULTIPLIER = 0.08
TACTICAL_TIGHT_SPACE_MULTIPLIER = 0.18
TACTICAL_LOW_SPACE_MULTIPLIER = 0.55
TACTICAL_DEAD_END_MULTIPLIER = 0.32
TACTICAL_FOOD_ESCAPE_HEALTH = 35
TACTICAL_SAFE_SPACE_BONUS = 0.35
TACTICAL_MIN_SPACE_FLOOR = 8
TACTICAL_BOARD_SPACE_CAP_RATIO = 0.18
TACTICAL_RISKY_TAIL_MULTIPLIER = 0.25

FINAL_ESCAPE_OPEN_NEIGHBORS = 2
FINAL_SPACE_ADVANTAGE_MIN = 8
FINAL_SPACE_ADVANTAGE_RATIO = 1.45
FINAL_TIGHT_SPACE_LENGTH_MULTIPLIER = 1.25
FINAL_DEAD_END_SPACE_MULTIPLIER = 2.0

HEAD_TO_HEAD_SAFE_SCORE = 3.0
HEAD_TO_HEAD_RISK_SCORE = 0.02
HEAD_TO_HEAD_ATTACK_SCORE = 3.8
HEAD_TO_HEAD_SPACE_WEIGHT = 0.01

TAIL_CHASE_DISTANCE_WEIGHT = 1.0
TAIL_CHASE_REACHABLE_SPACE_WEIGHT = 0.025
TAIL_CHASE_NO_FOOD_MULTIPLIER = 1.7
TAIL_CHASE_LOW_SPACE_MULTIPLIER = 1.8
TAIL_CHASE_LOW_HEALTH_MULTIPLIER = 0.5

EXPLORE_STALE_CELL_WEIGHT = 1.0
EXPLORE_DISTANCE_DECAY = 0.55
EXPLORE_REACHABLE_SPACE_WEIGHT = 0.015
EXPLORE_HEALTH_THRESHOLD = 55
EXPLORE_UNKNOWN_RATIO_THRESHOLD = 0.12
EXPLORE_OLD_CELL_TURN_THRESHOLD = 8
EXPLORE_WEIGHT_MULTIPLIER = 1.8
EXPLORE_LOW_HEALTH_MULTIPLIER = 0.35

STARVE_URGENCY_HEALTH = 45
STARVE_MEDIUM_HEALTH = 70
STARVE_PATH_BLOCK_WEIGHT = 2.2
STARVE_FOOD_RACE_WEIGHT = 1.4
STARVE_CLOSE_FOOD_DISTANCE = 4
STARVE_URGENCY_MULTIPLIER = 2.6
STARVE_MEDIUM_MULTIPLIER = 1.4
STARVE_WEIGHT_MULTIPLIER = 1.2
STARVE_LOW_HEALTH_MULTIPLIER = 0.45

GROWTH_SAFE_SPACE_RATIO = 0.28
GROWTH_FOOD_MULTIPLIER = 1.45
GROWTH_OPEN_SPACE_CHILL_MULTIPLIER = 0.85

STRATEGY_COMMITMENT_MIN_TURNS = 5
STRATEGY_COMMITMENT_MULTIPLIER = 1.65
STRATEGY_SWITCH_MARGIN = 1.35

FOOD_DISCIPLINE_MAX_DISTANCE = 5
FOOD_DISCIPLINE_DIRECT_MULTIPLIER = 5.5
FOOD_DISCIPLINE_PATH_MULTIPLIER = 2.4
FOOD_DISCIPLINE_PROGRESS_MULTIPLIER = 1.35
FOOD_DISCIPLINE_BYPASS_PENALTY = 0.35
FOOD_DISCIPLINE_ADJACENT_BYPASS_PENALTY = 0.12
FOOD_DISCIPLINE_LOW_SPACE_RATIO = 0.16

LOOKAHEAD_MAX_DEPTH = 3
LOOKAHEAD_TIME_BUDGET_MS = 80
LOOKAHEAD_OUR_BRANCH_LIMIT = 4
LOOKAHEAD_ENEMY_BRANCH_LIMIT = 4
LOOKAHEAD_ENEMY_JOINT_LIMIT = 64
LOOKAHEAD_SCORE_WEIGHT = 1.15
LOOKAHEAD_WORST_CASE_WEIGHT = 0.65
LOOKAHEAD_AVERAGE_CASE_WEIGHT = 0.35
LOOKAHEAD_DEATH_SCORE = -100000.0
LOOKAHEAD_FATAL_SCORE_CUTOFF = -50000.0
LOOKAHEAD_FATAL_MULTIPLIER = 0.01
LOOKAHEAD_SPACE_WEIGHT = 0.16
LOOKAHEAD_OPEN_NEIGHBOR_WEIGHT = 2.5
LOOKAHEAD_WALL_DISTANCE_WEIGHT = 0.35
LOOKAHEAD_HEALTH_WEIGHT = 0.08
LOOKAHEAD_FOOD_WEIGHT = 16.0
LOOKAHEAD_GROWTH_WEIGHT = 10.0
LOOKAHEAD_LOW_SPACE_PENALTY = 28.0
LOOKAHEAD_DEAD_END_PENALTY = 14.0
LOOKAHEAD_HEAD_RISK_PENALTY = 22.0
FORCED_COLLISION_ENEMY_MOVE_LIMIT = 2

OPPOSITE_DIRECTION = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def get_obstacle_map(game_state: GameState, blocked_tail_ids: set[str] | None = None):
    blocked_tail_ids = blocked_tail_ids or set()
    obstacle_map = np.zeros((game_state.board.height, game_state.board.width), dtype=bool)
    
    for snake in game_state.board.snakes:
        for index, body_part in enumerate(snake.body):
            if body_part is None:
                # we don't see this body section, could be many parts long
                continue
            if not in_bounds(body_part, game_state.board.width, game_state.board.height):
                continue
            if index == len(snake.body) - 1 and snake.id not in blocked_tail_ids:
                continue
            obstacle_map[body_part.y, body_part.x] = 1
    
    return obstacle_map


def get_vision_mask(width: int, height: int, center: Point | None, radius: int | None) -> np.ndarray:
    if center is None:
        return np.zeros((height, width), dtype=bool)
    if radius is None:
        return np.ones((height, width), dtype=bool)

    y, x = np.ogrid[:height, :width]
    distance = abs(x - center.x) + abs(y - center.y)
    return distance <= radius

def manhattan_distance(a: Point, b: Point) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def move_point(point: Point, direction: Direction) -> Point:
    return Point(x=point.x + direction.dx, y=point.y + direction.dy)


def point_key(point: Point) -> tuple[int, int]:
    return (point.x, point.y)


def in_bounds(point: Point, width: int, height: int) -> bool:
    return 0 <= point.x < width and 0 <= point.y < height


def wall_distance(point: Point, width: int, height: int) -> int:
    return min(point.x, point.y, width - 1 - point.x, height - 1 - point.y)


def enemy_head_options(game_state: GameState, min_length: int | None = None) -> set[tuple[int, int]]:
    width = game_state.board.width
    height = game_state.board.height
    options: set[tuple[int, int]] = set()

    for snake in game_state.board.snakes:
        if snake.id == game_state.you.id or snake.head is None:
            continue
        if min_length is not None and snake.length < min_length and snake_length_is_certain(snake, width, height):
            continue
        for direction in Direction:
            point = move_point(snake.head, direction)
            if in_bounds(point, width, height):
                options.add((point.x, point.y))

    return options


def shortest_food_distance(obstacle_map: np.ndarray, start: Point, food_list: list[Point]) -> int | None:
    best = UNREACHABLE_DISTANCE
    for food in food_list:
        if start.x == food.x and start.y == food.y:
            return 0
        _, distance = a_star_wrapper(obstacle_map, start, food)
        best = min(best, distance)
    return None if best >= UNREACHABLE_DISTANCE else best


def known_food_points(game_state: GameState, agent_state: "AgentState") -> list[Point]:
    food_list: list[Point] = list(game_state.board.food)
    for point in agent_state.remembered_food.values():
        if all(point.x != food.x or point.y != food.y for food in food_list):
            food_list.append(point)
    return food_list


def known_food_keys(game_state: GameState, agent_state: "AgentState") -> set[tuple[int, int]]:
    return {point_key(food) for food in known_food_points(game_state, agent_state)}


def snake_tail_point(snake) -> Point | None:
    for body_part in reversed(snake.body):
        if body_part is not None and body_part.x >= 0 and body_part.y >= 0:
            return body_part
    return None


def snake_length_is_certain(snake, width: int, height: int) -> bool:
    known_body = [
        body_part
        for body_part in snake.body
        if body_part is not None and in_bounds(body_part, width, height)
    ]
    return len(known_body) >= snake.length and all(body_part is not None for body_part in snake.body)


def enemy_food_urgency(health: int | None) -> float:
    health = health if health is not None else 100
    if health <= STARVE_URGENCY_HEALTH:
        return STARVE_URGENCY_MULTIPLIER
    if health <= STARVE_MEDIUM_HEALTH:
        return STARVE_MEDIUM_MULTIPLIER
    return 1.0


def flood_fill_space(obstacle_map: np.ndarray, start: Point) -> int:
    height, width = obstacle_map.shape
    if start is None:
        return 0

    if not (0 <= start.x < width and 0 <= start.y < height):
        return 0
    if obstacle_map[start.y, start.x]:
        return 0

    visited = {(start.x, start.y)}
    stack = [(start.x, start.y)]
    reachable = 0

    while stack:
        x, y = stack.pop()
        reachable += 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if obstacle_map[ny, nx]:
                continue
            if (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            stack.append((nx, ny))

    return reachable


def count_open_neighbors(obstacle_map: np.ndarray, point: Point) -> int:
    height, width = obstacle_map.shape
    count = 0
    for direction in Direction:
        nx = point.x + direction.dx
        ny = point.y + direction.dy
        if 0 <= nx < width and 0 <= ny < height and not obstacle_map[ny, nx]:
            count += 1
    return count

# ---------------------------------------------------------
# Battlesnake Agent Implementation
# ---------------------------------------------------------
@dataclass
class AgentState:
    past_turn: Direction
    last_seen_turn: Dict[tuple[int, int], int]
    remembered_food: Dict[tuple[int, int], Point]
    focused_strategy: str | None
    focused_strategy_turns: int


@dataclass
class MoveSafety:
    reachable_space: int
    open_neighbors: int
    wall_distance: int
    targets_food: bool
    head_risk: bool
    risky_tail: bool


@dataclass(frozen=True)
class SimSnake:
    id: str
    health: int
    length: int
    body: tuple[tuple[int, int], ...]
    length_uncertain: bool = False

    @property
    def head(self) -> tuple[int, int] | None:
        return self.body[0] if self.body else None


@dataclass(frozen=True)
class SimState:
    width: int
    height: int
    food: frozenset[tuple[int, int]]
    hazards: frozenset[tuple[int, int]]
    snakes: tuple[SimSnake, ...]
    you_id: str
    initial_you_length: int
    hazard_damage: int


class Strategy:
    def vote(self, game_state: GameState, possible_moves: List[Direction], agent_state: AgentState) -> Dict[Direction, float]:
        raise NotImplementedError()

    @staticmethod
    def _normalize(scores: Dict[Direction, float], possible_moves: List[Direction]) -> Dict[Direction, float]:
        if not possible_moves:
            return {}

        filtered = {move: max(0.0, scores.get(move, 0.0)) for move in possible_moves}
        total = sum(filtered.values())
        if total <= 0:
            equal_share = 1.0 / len(possible_moves)
            return {move: equal_share for move in possible_moves}
        return {move: value / total for move, value in filtered.items()}


class StrategyEat(Strategy):
    def vote(self, game_state: GameState, possible_moves: List[Direction], agent_state: AgentState) -> Dict[Direction, float]:
        if not possible_moves:
            return {}

        head = game_state.you.head
        if head is None:
            return self._normalize({}, possible_moves)

        food_list = known_food_points(game_state, agent_state)
        if not food_list:
            return self._normalize({}, possible_moves)

        obstacle_map = get_obstacle_map(game_state)
        health = game_state.you.health or 0
        scores: Dict[Direction, float] = {}

        for move in possible_moves:
            new_head = move_point(head, move)
            nearest_food_distance = shortest_food_distance(obstacle_map, new_head, food_list)
            reachable_space = flood_fill_space(obstacle_map, new_head)
            if nearest_food_distance is None:
                food_pressure = 0.0
            else:
                food_pressure = 1.0 / (1.0 + nearest_food_distance)
            health_bias = 1.0 + max(0, EAT_HEALTH_BIAS_BASE - health) / EAT_HEALTH_BIAS_SCALE
            scores[move] = (food_pressure * health_bias) + (reachable_space * EAT_REACHABLE_SPACE_WEIGHT)

        return self._normalize(scores, possible_moves)


class StrategyChill(Strategy):
    def vote(self, game_state: GameState, possible_moves: List[Direction], agent_state: AgentState) -> Dict[Direction, float]:
        if not possible_moves:
            return {}

        head = game_state.you.head
        if head is None:
            return self._normalize({}, possible_moves)

        obstacle_map = get_obstacle_map(game_state)
        scores: Dict[Direction, float] = {}

        for move in possible_moves:
            new_head = move_point(head, move)
            reachable_space = flood_fill_space(obstacle_map, new_head)
            open_neighbors = count_open_neighbors(obstacle_map, new_head)
            distance_to_wall = wall_distance(new_head, game_state.board.width, game_state.board.height)
            scores[move] = (
                (reachable_space * CHILL_REACHABLE_SPACE_WEIGHT)
                + (open_neighbors * CHILL_OPEN_EXITS_WEIGHT)
                + (distance_to_wall * CHILL_WALL_DISTANCE_WEIGHT)
            )

        return self._normalize(scores, possible_moves)


class StrategyAvoidGettingStuck(Strategy):
    def vote(self, game_state: GameState, possible_moves: List[Direction], agent_state: AgentState) -> Dict[Direction, float]:
        if not possible_moves:
            return {}

        head = game_state.you.head
        if head is None:
            return self._normalize({}, possible_moves)

        obstacle_map = get_obstacle_map(game_state)
        scores: Dict[Direction, float] = {}

        for move in possible_moves:
            new_head = move_point(head, move)
            reachable_space = flood_fill_space(obstacle_map, new_head)
            open_neighbors = count_open_neighbors(obstacle_map, new_head)
            distance_to_wall = wall_distance(new_head, game_state.board.width, game_state.board.height)

            corridor_multiplier = 1.0
            if open_neighbors <= AVOID_STUCK_DEAD_END_THRESHOLD:
                corridor_multiplier = AVOID_STUCK_DEAD_END_MULTIPLIER
            elif open_neighbors <= AVOID_STUCK_CORRIDOR_THRESHOLD:
                corridor_multiplier = AVOID_STUCK_CORRIDOR_MULTIPLIER

            scores[move] = (
                (reachable_space * AVOID_STUCK_REACHABLE_SPACE_WEIGHT)
                + (open_neighbors * AVOID_STUCK_OPEN_EXITS_WEIGHT)
                + (distance_to_wall * AVOID_STUCK_WALL_DISTANCE_WEIGHT)
            ) * corridor_multiplier

        return self._normalize(scores, possible_moves)
    
class StrategyFlee(Strategy):
    def vote(self, game_state: GameState, possible_moves: List[Direction], agent_state: AgentState) -> Dict[Direction, float]:
        if not possible_moves:
            return {}

        head = game_state.you.head
        if head is None:
            return self._normalize({}, possible_moves)

        enemy_heads = [snake.head for snake in game_state.board.snakes if snake.id != game_state.you.id and snake.head is not None]
        if not enemy_heads:
            return self._normalize({}, possible_moves)

        obstacle_map = get_obstacle_map(game_state)
        scores: Dict[Direction, float] = {}

        for move in possible_moves:
            new_head = move_point(head, move)
            distance_to_enemy = min(manhattan_distance(new_head, enemy_head) for enemy_head in enemy_heads)
            reachable_space = flood_fill_space(obstacle_map, new_head)
            distance_to_wall = wall_distance(new_head, game_state.board.width, game_state.board.height)
            scores[move] = (
                (distance_to_enemy * 1.5)
                + (reachable_space * FLEE_REACHABLE_SPACE_WEIGHT)
                + (distance_to_wall * FLEE_WALL_DISTANCE_WEIGHT)
            )

        return self._normalize(scores, possible_moves)


class StrategyHeadToHead(Strategy):
    def vote(self, game_state: GameState, possible_moves: List[Direction], agent_state: AgentState) -> Dict[Direction, float]:
        if not possible_moves:
            return {}

        head = game_state.you.head
        if head is None:
            return self._normalize({}, possible_moves)

        obstacle_map = get_obstacle_map(game_state)
        dangerous_targets = enemy_head_options(game_state, min_length=game_state.you.length)
        shorter_targets = enemy_head_options(game_state, min_length=None) - dangerous_targets
        scores: Dict[Direction, float] = {}

        for move in possible_moves:
            new_head = move_point(head, move)
            target = (new_head.x, new_head.y)
            reachable_space = flood_fill_space(obstacle_map, new_head)

            if target in dangerous_targets:
                scores[move] = HEAD_TO_HEAD_RISK_SCORE
            elif target in shorter_targets:
                scores[move] = HEAD_TO_HEAD_ATTACK_SCORE + reachable_space * HEAD_TO_HEAD_SPACE_WEIGHT
            else:
                scores[move] = HEAD_TO_HEAD_SAFE_SCORE + reachable_space * HEAD_TO_HEAD_SPACE_WEIGHT

        return self._normalize(scores, possible_moves)


class StrategyFollowTail(Strategy):
    def vote(self, game_state: GameState, possible_moves: List[Direction], agent_state: AgentState) -> Dict[Direction, float]:
        if not possible_moves:
            return {}

        head = game_state.you.head
        tail = snake_tail_point(game_state.you)
        if head is None or tail is None:
            return self._normalize({}, possible_moves)

        obstacle_map = get_obstacle_map(game_state)
        scores: Dict[Direction, float] = {}

        for move in possible_moves:
            new_head = move_point(head, move)
            _, tail_distance = a_star_wrapper(obstacle_map, new_head, tail)
            reachable_space = flood_fill_space(obstacle_map, new_head)
            if tail_distance >= UNREACHABLE_DISTANCE:
                tail_score = 0.0
            else:
                tail_score = TAIL_CHASE_DISTANCE_WEIGHT / (1.0 + tail_distance)
            scores[move] = tail_score + reachable_space * TAIL_CHASE_REACHABLE_SPACE_WEIGHT

        return self._normalize(scores, possible_moves)


class StrategyExplore(Strategy):
    def vote(self, game_state: GameState, possible_moves: List[Direction], agent_state: AgentState) -> Dict[Direction, float]:
        if not possible_moves:
            return {}

        head = game_state.you.head
        if head is None:
            return self._normalize({}, possible_moves)

        obstacle_map = get_obstacle_map(game_state)
        width = game_state.board.width
        height = game_state.board.height
        current_turn = game_state.turn
        scores: Dict[Direction, float] = {}

        for move in possible_moves:
            new_head = move_point(head, move)
            exploration_score = 0.0

            for y in range(height):
                for x in range(width):
                    if obstacle_map[y, x]:
                        continue
                    last_seen = agent_state.last_seen_turn.get((x, y))
                    age = current_turn + EXPLORE_OLD_CELL_TURN_THRESHOLD if last_seen is None else current_turn - last_seen
                    if age < EXPLORE_OLD_CELL_TURN_THRESHOLD:
                        continue

                    distance = abs(new_head.x - x) + abs(new_head.y - y)
                    exploration_score += (age * EXPLORE_STALE_CELL_WEIGHT) / (1.0 + distance * EXPLORE_DISTANCE_DECAY)

            reachable_space = flood_fill_space(obstacle_map, new_head)
            scores[move] = exploration_score + reachable_space * EXPLORE_REACHABLE_SPACE_WEIGHT

        return self._normalize(scores, possible_moves)


class StrategyStarve(Strategy):
    def vote(self, game_state: GameState, possible_moves: List[Direction], agent_state: AgentState) -> Dict[Direction, float]:
        if not possible_moves:
            return {}

        head = game_state.you.head
        if head is None:
            return self._normalize({}, possible_moves)

        food_list = known_food_points(game_state, agent_state)
        if not food_list:
            return self._normalize({}, possible_moves)

        obstacle_map = get_obstacle_map(game_state)
        dangerous_targets = enemy_head_options(game_state, min_length=game_state.you.length)
        scores: Dict[Direction, float] = {}
        enemy_snakes = [
            snake
            for snake in game_state.board.snakes
            if snake.id != game_state.you.id
            and snake.head is not None
            and (snake.health or 100) <= STARVE_MEDIUM_HEALTH
        ]

        if not enemy_snakes:
            return self._normalize({}, possible_moves)

        for move in possible_moves:
            new_head = move_point(head, move)
            if (new_head.x, new_head.y) in dangerous_targets:
                scores[move] = 0.0
                continue

            blocked_map = obstacle_map.copy()
            blocked_map[new_head.y, new_head.x] = True
            score = 0.0

            for snake in enemy_snakes:
                assert snake.head is not None
                urgency = enemy_food_urgency(snake.health)
                likely_food = sorted(food_list, key=lambda food: manhattan_distance(snake.head, food))[:3]

                for food in likely_food:
                    _, enemy_distance = a_star_wrapper(obstacle_map, snake.head, food)
                    if enemy_distance >= UNREACHABLE_DISTANCE:
                        continue

                    _, blocked_distance = a_star_wrapper(blocked_map, snake.head, food)
                    if blocked_distance >= UNREACHABLE_DISTANCE:
                        path_penalty = STARVE_CLOSE_FOOD_DISTANCE
                    else:
                        path_penalty = max(0, blocked_distance - enemy_distance)

                    our_distance = shortest_food_distance(obstacle_map, new_head, [food])
                    if our_distance is None:
                        our_distance = UNREACHABLE_DISTANCE

                    food_race_score = 0.0
                    if our_distance <= enemy_distance + 1:
                        food_race_score = max(0, STARVE_CLOSE_FOOD_DISTANCE + 2 - enemy_distance)
                    if manhattan_distance(new_head, food) <= 1 and enemy_distance <= STARVE_CLOSE_FOOD_DISTANCE + 1:
                        food_race_score += 2.0

                    score += urgency * (
                        path_penalty * STARVE_PATH_BLOCK_WEIGHT
                        + food_race_score * STARVE_FOOD_RACE_WEIGHT
                    )

            scores[move] = score

        return self._normalize(scores, possible_moves)


class BestAgent(BaseAgent):
    def __init__(self):
        self.agent_states: dict[str, AgentState] = {}
        self.strategies: List[Strategy] = [
            StrategyEat(),
            StrategyChill(),
            StrategyAvoidGettingStuck(),
            StrategyFlee(),
            StrategyHeadToHead(),
            StrategyFollowTail(),
            StrategyExplore(),
            StrategyStarve(),
        ]

    def _new_agent_state(self, game_state: GameState) -> AgentState:
        return AgentState(
            past_turn=self._infer_direction_from_body(game_state),
            last_seen_turn={},
            remembered_food={},
            focused_strategy=None,
            focused_strategy_turns=0,
        )

    def _update_memory(self, game_state: GameState, agent_state: AgentState) -> None:
        head = game_state.you.head
        width = game_state.board.width
        height = game_state.board.height
        view_radius = game_state.game.ruleset.settings.viewRadius
        visible_mask = get_vision_mask(width, height, head, view_radius)

        visible_food = {(food.x, food.y): Point(x=food.x, y=food.y) for food in game_state.board.food}

        for y in range(height):
            for x in range(width):
                if not visible_mask[y, x]:
                    continue
                agent_state.last_seen_turn[(x, y)] = game_state.turn
                if (x, y) in visible_food:
                    agent_state.remembered_food[(x, y)] = visible_food[(x, y)]
                else:
                    agent_state.remembered_food.pop((x, y), None)

    def _stale_area_ratio(self, game_state: GameState, agent_state: AgentState) -> float:
        width = game_state.board.width
        height = game_state.board.height
        board_area = max(1, width * height)
        stale_cells = 0

        for y in range(height):
            for x in range(width):
                last_seen = agent_state.last_seen_turn.get((x, y))
                if last_seen is None or game_state.turn - last_seen >= EXPLORE_OLD_CELL_TURN_THRESHOLD:
                    stale_cells += 1

        return stale_cells / board_area

    def _lowest_enemy_health(self, game_state: GameState) -> int | None:
        health_values = [
            snake.health
            for snake in game_state.board.snakes
            if snake.id != game_state.you.id and snake.health is not None and snake.head is not None
        ]
        return min(health_values) if health_values else None

    def _infer_direction_from_body(self, game_state: GameState, fallback: Direction = DEFAULT_START_DIRECTION) -> Direction:
        head = game_state.you.head
        body = game_state.you.body
        if head is None or len(body) < 2 or body[1] is None:
            return fallback

        inferred = Direction.from_board_delta((head.x - body[1].x, head.y - body[1].y))
        return inferred if inferred is not None else fallback

    def _fallback_direction(self, game_state: GameState, agent_state: AgentState) -> Direction:
        head = game_state.you.head
        if head is None:
            return agent_state.past_turn

        cur = agent_state.past_turn
        left_map = {
            Direction.UP: Direction.LEFT,
            Direction.RIGHT: Direction.UP,
            Direction.DOWN: Direction.RIGHT,
            Direction.LEFT: Direction.DOWN,
        }
        right_map = {k: v for v, k in left_map.items()}
        candidates = [left_map[cur], right_map[cur], cur, Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

        obstacle_map = get_obstacle_map(game_state)
        width = game_state.board.width
        height = game_state.board.height
        scores: Dict[Direction, float] = {}

        for direction in candidates:
            nx = head.x + direction.dx
            ny = head.y + direction.dy
            if 0 <= nx < width and 0 <= ny < height and not obstacle_map[ny, nx]:
                new_head = Point(x=nx, y=ny)
                scores[direction] = (
                    (flood_fill_space(obstacle_map, new_head) * FALLBACK_OPEN_SPACE_WEIGHT)
                    + (wall_distance(new_head, width, height) * FALLBACK_WALL_DISTANCE_WEIGHT)
                )

        if scores:
            return max(scores.items(), key=lambda item: item[1])[0]

        return cur

    def _closest_food_distance(self, game_state: GameState, agent_state: AgentState) -> int | None:
        head = game_state.you.head
        food_list = known_food_points(game_state, agent_state)
        if head is None or not food_list:
            return None
        return min(manhattan_distance(head, food) for food in food_list)

    def _closest_enemy_head_distance(self, game_state: GameState) -> int | None:
        head = game_state.you.head
        if head is None:
            return None

        enemy_heads = [snake.head for snake in game_state.board.snakes if snake.id != game_state.you.id and snake.head is not None]
        if not enemy_heads:
            return None
        return min(manhattan_distance(head, enemy_head) for enemy_head in enemy_heads)

    def _reachable_space(self, game_state: GameState, start: Point) -> int:
        return flood_fill_space(get_obstacle_map(game_state), start)

    def _move_targets_known_food(self, game_state: GameState, agent_state: AgentState, move: Direction) -> bool:
        head = game_state.you.head
        if head is None:
            return False
        return point_key(move_point(head, move)) in known_food_keys(game_state, agent_state)

    def _enemy_tail_may_stay(self, game_state: GameState, agent_state: AgentState, snake) -> bool:
        if snake.id == game_state.you.id or snake.head is None:
            return False

        for food in known_food_points(game_state, agent_state):
            if manhattan_distance(snake.head, food) == 1:
                return True
        return False

    def _blocked_tail_ids_for_move(
        self,
        game_state: GameState,
        agent_state: AgentState,
        move: Direction | None = None,
    ) -> set[str]:
        blocked_tail_ids: set[str] = set()

        if move is not None and self._move_targets_known_food(game_state, agent_state, move):
            blocked_tail_ids.add(game_state.you.id)

        for snake in game_state.board.snakes:
            if self._enemy_tail_may_stay(game_state, agent_state, snake):
                blocked_tail_ids.add(snake.id)

        return blocked_tail_ids

    def _move_obstacle_map(
        self,
        game_state: GameState,
        agent_state: AgentState,
        move: Direction | None = None,
    ) -> np.ndarray:
        blocked_tail_ids = self._blocked_tail_ids_for_move(game_state, agent_state, move)
        return get_obstacle_map(game_state, blocked_tail_ids=blocked_tail_ids)

    def _minimum_viable_space(self, game_state: GameState) -> int:
        board_area = max(1, game_state.board.width * game_state.board.height)
        board_cap = max(TACTICAL_MIN_SPACE_FLOOR, int(board_area * TACTICAL_BOARD_SPACE_CAP_RATIO))
        return min(max(TACTICAL_MIN_SPACE_FLOOR, game_state.you.length + 2), board_cap)

    def _move_safety(self, game_state: GameState, agent_state: AgentState, move: Direction) -> MoveSafety:
        head = game_state.you.head
        if head is None:
            return MoveSafety(0, 0, 0, False, False, False)

        new_head = move_point(head, move)
        obstacle_map = self._move_obstacle_map(game_state, agent_state, move)
        risky_tail = obstacle_map[new_head.y, new_head.x]
        reachable_space = flood_fill_space(obstacle_map, new_head)
        dangerous_targets = enemy_head_options(game_state, min_length=game_state.you.length)

        return MoveSafety(
            reachable_space=reachable_space,
            open_neighbors=count_open_neighbors(obstacle_map, new_head),
            wall_distance=wall_distance(new_head, game_state.board.width, game_state.board.height),
            targets_food=self._move_targets_known_food(game_state, agent_state, move),
            head_risk=point_key(new_head) in dangerous_targets,
            risky_tail=bool(risky_tail),
        )

    def _survival_multiplier(self, game_state: GameState, safety: MoveSafety) -> float:
        health = game_state.you.health or 0
        min_space = self._minimum_viable_space(game_state)
        board_area = max(1, game_state.board.width * game_state.board.height)

        multiplier = 1.0
        if safety.risky_tail:
            multiplier *= TACTICAL_RISKY_TAIL_MULTIPLIER
        if safety.head_risk:
            multiplier *= TACTICAL_HEAD_RISK_MULTIPLIER

        if safety.reachable_space <= max(2, game_state.you.length):
            multiplier *= TACTICAL_TIGHT_SPACE_MULTIPLIER
        elif safety.reachable_space < min_space:
            multiplier *= TACTICAL_LOW_SPACE_MULTIPLIER

        if safety.open_neighbors <= 1:
            if not (safety.targets_food and health <= TACTICAL_FOOD_ESCAPE_HEALTH):
                multiplier *= TACTICAL_DEAD_END_MULTIPLIER

        safe_space_ratio = safety.reachable_space / board_area
        multiplier *= 1.0 + min(TACTICAL_SAFE_SPACE_BONUS, safe_space_ratio * TACTICAL_SAFE_SPACE_BONUS)
        return multiplier

    def _apply_tactical_safety(
        self,
        game_state: GameState,
        agent_state: AgentState,
        scores: Dict[Direction, float],
    ) -> Dict[Direction, float]:
        return {
            move: score * self._survival_multiplier(game_state, self._move_safety(game_state, agent_state, move))
            for move, score in scores.items()
        }

    def _snake_direction(self, snake) -> Direction | None:
        if snake.head is None or len(snake.body) < 2 or snake.body[1] is None:
            return None
        if snake.body[1].x < 0 or snake.body[1].y < 0:
            return None

        delta = (snake.head.x - snake.body[1].x, snake.head.y - snake.body[1].y)
        return Direction.from_board_delta(delta)

    def _enemy_legal_head_targets(
        self,
        game_state: GameState,
        agent_state: AgentState,
        snake,
    ) -> list[tuple[Direction, tuple[int, int]]]:
        if snake.id == game_state.you.id or snake.head is None:
            return []

        obstacle_map = self._move_obstacle_map(game_state, agent_state)
        forbidden_direction = OPPOSITE_DIRECTION.get(self._snake_direction(snake))
        targets: list[tuple[Direction, tuple[int, int]]] = []

        for direction in Direction:
            if direction == forbidden_direction:
                continue

            target = move_point(snake.head, direction)
            if not in_bounds(target, game_state.board.width, game_state.board.height):
                continue
            if obstacle_map[target.y, target.x]:
                continue
            targets.append((direction, point_key(target)))

        return targets

    def _constrained_enemy_head_targets(
        self,
        game_state: GameState,
        agent_state: AgentState,
    ) -> set[tuple[int, int]]:
        targets: set[tuple[int, int]] = set()
        for snake in game_state.board.snakes:
            if snake.id == game_state.you.id or snake.head is None:
                continue

            enemy_moves = self._enemy_legal_head_targets(game_state, agent_state, snake)
            if 0 < len(enemy_moves) <= FORCED_COLLISION_ENEMY_MOVE_LIMIT:
                targets.update(target for _, target in enemy_moves)

        return targets

    def _avoid_constrained_enemy_collisions(
        self,
        game_state: GameState,
        agent_state: AgentState,
        possible_moves: List[Direction],
    ) -> List[Direction]:
        head = game_state.you.head
        if head is None or not possible_moves:
            return possible_moves

        constrained_targets = self._constrained_enemy_head_targets(game_state, agent_state)
        if not constrained_targets:
            return possible_moves

        evasive_moves = [
            move
            for move in possible_moves
            if point_key(move_point(head, move)) not in constrained_targets
        ]
        return evasive_moves or possible_moves

    def _food_move_is_safe(self, game_state: GameState, agent_state: AgentState, move: Direction) -> bool:
        safety = self._move_safety(game_state, agent_state, move)
        if safety.risky_tail or safety.head_risk:
            return False

        min_space = self._minimum_viable_space(game_state)
        board_area = max(1, game_state.board.width * game_state.board.height)
        if safety.reachable_space < max(3, game_state.you.length) and safety.open_neighbors <= 1:
            return False
        if safety.reachable_space < min_space and safety.reachable_space / board_area < FOOD_DISCIPLINE_LOW_SPACE_RATIO:
            return False
        return True

    def _best_food_path(
        self,
        game_state: GameState,
        agent_state: AgentState,
    ) -> tuple[Point | None, Direction | None, int | None]:
        head = game_state.you.head
        food_list = known_food_points(game_state, agent_state)
        if head is None or not food_list:
            return None, None, None

        obstacle_map = self._move_obstacle_map(game_state, agent_state)
        if in_bounds(head, game_state.board.width, game_state.board.height):
            obstacle_map[head.y, head.x] = False

        best_food: Point | None = None
        best_move: Direction | None = None
        best_distance = UNREACHABLE_DISTANCE
        for food in food_list:
            move, distance = a_star_wrapper(obstacle_map, head, food)
            if distance < best_distance:
                best_food = food
                best_move = move
                best_distance = distance

        if best_distance >= UNREACHABLE_DISTANCE:
            return None, None, None
        return best_food, best_move, best_distance

    def _food_target_is_contested(
        self,
        game_state: GameState,
        target_food: Point,
        our_distance: int,
    ) -> bool:
        if game_state.you.head is None:
            return False

        for snake in game_state.board.snakes:
            if snake.id == game_state.you.id or snake.head is None:
                continue

            obstacle_map = get_obstacle_map(game_state)
            obstacle_map[snake.head.y, snake.head.x] = False
            _, enemy_distance = a_star_wrapper(obstacle_map, snake.head, target_food)
            if enemy_distance >= UNREACHABLE_DISTANCE:
                continue
            if enemy_distance < our_distance:
                return True
            if enemy_distance == our_distance and snake.length >= game_state.you.length:
                return True

        return False

    def _apply_food_discipline(
        self,
        game_state: GameState,
        agent_state: AgentState,
        scores: Dict[Direction, float],
    ) -> Dict[Direction, float]:
        if not scores:
            return scores

        head = game_state.you.head
        if head is None:
            return scores

        food_list = known_food_points(game_state, agent_state)
        if not food_list:
            return scores

        target_food, path_move, current_distance = self._best_food_path(game_state, agent_state)
        if target_food is None or current_distance is None:
            return scores

        target_is_contested = self._food_target_is_contested(game_state, target_food, current_distance)
        safe_path_available = (
            path_move in scores
            and current_distance <= FOOD_DISCIPLINE_MAX_DISTANCE
            and self._food_move_is_safe(game_state, agent_state, path_move)
            and not target_is_contested
        )
        path_safety_score = (
            self._move_safety_score(game_state, agent_state, path_move)
            if safe_path_available and path_move is not None
            else 0.0
        )

        obstacle_map = self._move_obstacle_map(game_state, agent_state)
        adjusted = dict(scores)
        health = game_state.you.health or 0
        hunger_multiplier = 1.0 + max(0, EAT_HEALTH_MEDIUM_THRESHOLD - health) / 100.0

        for move in scores:
            safety = self._move_safety(game_state, agent_state, move)
            move_is_safe = self._food_move_is_safe(game_state, agent_state, move)
            new_head = move_point(head, move)
            targets_food = point_key(new_head) in known_food_keys(game_state, agent_state)
            next_distance = shortest_food_distance(obstacle_map, new_head, food_list)

            if targets_food and move_is_safe:
                direct_target = Point(x=new_head.x, y=new_head.y)
                if not self._food_target_is_contested(game_state, direct_target, 2):
                    adjusted[move] *= FOOD_DISCIPLINE_DIRECT_MULTIPLIER * hunger_multiplier
                continue

            if safe_path_available and move == path_move:
                adjusted[move] *= FOOD_DISCIPLINE_PATH_MULTIPLIER * hunger_multiplier
                continue

            if next_distance is not None and current_distance is not None and next_distance < current_distance:
                adjusted[move] *= FOOD_DISCIPLINE_PROGRESS_MULTIPLIER

            if not safe_path_available:
                continue

            move_safety_score = self._move_safety_score(game_state, agent_state, move)
            if move_safety_score > path_safety_score * 1.45 and not safety.head_risk:
                continue

            if next_distance is None or next_distance >= current_distance:
                penalty = (
                    FOOD_DISCIPLINE_ADJACENT_BYPASS_PENALTY
                    if current_distance <= 2
                    else FOOD_DISCIPLINE_BYPASS_PENALTY
                )
                adjusted[move] *= penalty

        return adjusted

    def _sim_state_from_game(self, game_state: GameState, agent_state: AgentState) -> SimState:
        snakes: list[SimSnake] = []
        for snake in game_state.board.snakes:
            if snake.head is None:
                continue
            body = tuple(
                (part.x, part.y)
                for part in snake.body
                if part is not None and in_bounds(part, game_state.board.width, game_state.board.height)
            )
            if not body:
                body = ((snake.head.x, snake.head.y),)
            elif body[0] != (snake.head.x, snake.head.y):
                body = ((snake.head.x, snake.head.y),) + body
            snakes.append(
                SimSnake(
                    id=snake.id,
                    health=snake.health if snake.health is not None else 100,
                    length=snake.length,
                    body=body,
                    length_uncertain=not snake_length_is_certain(snake, game_state.board.width, game_state.board.height),
                )
            )

        hazard_damage = game_state.game.ruleset.settings.hazardDamagePerTurn or 0
        return SimState(
            width=game_state.board.width,
            height=game_state.board.height,
            food=frozenset(known_food_keys(game_state, agent_state)),
            hazards=frozenset((hazard.x, hazard.y) for hazard in game_state.board.hazards),
            snakes=tuple(snakes),
            you_id=game_state.you.id,
            initial_you_length=game_state.you.length,
            hazard_damage=hazard_damage,
        )

    def _sim_snake_by_id(self, sim_state: SimState, snake_id: str) -> SimSnake | None:
        for snake in sim_state.snakes:
            if snake.id == snake_id:
                return snake
        return None

    def _sim_you(self, sim_state: SimState) -> SimSnake | None:
        return self._sim_snake_by_id(sim_state, sim_state.you_id)

    def _sim_obstacle_map(
        self,
        sim_state: SimState,
        free_cell: tuple[int, int] | None = None,
    ) -> np.ndarray:
        obstacle_map = np.zeros((sim_state.height, sim_state.width), dtype=bool)
        for snake in sim_state.snakes:
            for index, (x, y) in enumerate(snake.body):
                if index == len(snake.body) - 1:
                    continue
                if 0 <= x < sim_state.width and 0 <= y < sim_state.height:
                    obstacle_map[y, x] = True

        if free_cell is not None:
            x, y = free_cell
            if 0 <= x < sim_state.width and 0 <= y < sim_state.height:
                obstacle_map[y, x] = False
        return obstacle_map

    def _sim_direction(self, snake: SimSnake) -> Direction | None:
        if len(snake.body) < 2:
            return None
        head = snake.body[0]
        neck = snake.body[1]
        return Direction.from_board_delta((head[0] - neck[0], head[1] - neck[1]))

    def _sim_shortest_food_distance(
        self,
        sim_state: SimState,
        obstacle_map: np.ndarray,
        start: tuple[int, int],
    ) -> int | None:
        if not sim_state.food:
            return None
        start_point = Point(x=start[0], y=start[1])
        best = UNREACHABLE_DISTANCE
        for food in sim_state.food:
            if start == food:
                return 0
            _, distance = a_star_wrapper(obstacle_map, start_point, Point(x=food[0], y=food[1]))
            best = min(best, distance)
        return None if best >= UNREACHABLE_DISTANCE else best

    def _rank_sim_moves(
        self,
        sim_state: SimState,
        snake: SimSnake,
        moves: list[Direction],
        is_enemy: bool,
    ) -> list[Direction]:
        head = snake.head
        if head is None:
            return moves

        obstacle_map = self._sim_obstacle_map(sim_state, free_cell=head)
        you = self._sim_you(sim_state)
        scored_moves: list[tuple[float, Direction]] = []
        for move in moves:
            new_head = (head[0] + move.dx, head[1] + move.dy)
            if not (0 <= new_head[0] < sim_state.width and 0 <= new_head[1] < sim_state.height):
                scored_moves.append((-10000.0, move))
                continue

            point = Point(x=new_head[0], y=new_head[1])
            blocked = obstacle_map[new_head[1], new_head[0]]
            reachable_space = 0 if blocked else flood_fill_space(obstacle_map, point)
            open_neighbors = 0 if blocked else count_open_neighbors(obstacle_map, point)
            food_distance = None if blocked else self._sim_shortest_food_distance(sim_state, obstacle_map, new_head)

            score = reachable_space * 0.12 + open_neighbors * 2.0
            if blocked:
                score -= 120.0
            if new_head in sim_state.food:
                score += 42.0 + max(0, 55 - snake.health) * 0.4
            elif food_distance is not None:
                score += 12.0 / (1.0 + food_distance)

            if is_enemy and you is not None and you.head is not None:
                distance_to_us = abs(new_head[0] - you.head[0]) + abs(new_head[1] - you.head[1])
                if distance_to_us <= 1 and (snake.length >= you.length or snake.length_uncertain):
                    score += 24.0

            scored_moves.append((score, move))

        scored_moves.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored_moves]

    def _limited(self, values: list, limit: int | None) -> list:
        if limit is None or limit <= 0:
            return values
        return values[:limit]

    def _sim_candidate_moves(self, sim_state: SimState, snake: SimSnake, is_enemy: bool) -> list[Direction]:
        head = snake.head
        if head is None:
            return []

        current_direction = self._sim_direction(snake)
        forbidden_direction = OPPOSITE_DIRECTION.get(current_direction) if current_direction is not None else None
        moves = [
            direction
            for direction in Direction
            if direction != forbidden_direction
            and 0 <= head[0] + direction.dx < sim_state.width
            and 0 <= head[1] + direction.dy < sim_state.height
        ]
        if not moves:
            moves = [current_direction or Direction.UP]

        ranked = self._rank_sim_moves(sim_state, snake, moves, is_enemy=is_enemy)
        limit = LOOKAHEAD_ENEMY_BRANCH_LIMIT if is_enemy else LOOKAHEAD_OUR_BRANCH_LIMIT
        return self._limited(ranked, limit)

    def _sim_enemy_joint_moves(self, sim_state: SimState) -> list[dict[str, Direction]]:
        enemies = [snake for snake in sim_state.snakes if snake.id != sim_state.you_id]
        if not enemies:
            return [{}]

        move_lists = [
            (snake.id, self._sim_candidate_moves(sim_state, snake, is_enemy=True))
            for snake in enemies
        ]
        move_lists = [(snake_id, moves) for snake_id, moves in move_lists if moves]
        if not move_lists:
            return [{}]

        joint_moves: list[dict[str, Direction]] = []
        for combo in itertools.product(*(moves for _, moves in move_lists)):
            joint_moves.append(
                {
                    snake_id: move
                    for (snake_id, _), move in zip(move_lists, combo, strict=True)
                }
            )

        return self._limited(joint_moves, LOOKAHEAD_ENEMY_JOINT_LIMIT)

    def _simulate_joint_move(self, sim_state: SimState, moves: dict[str, Direction]) -> SimState:
        moved_snakes: list[SimSnake] = []
        dead_ids: set[str] = set()
        consumed_food: set[tuple[int, int]] = set()

        for snake in sim_state.snakes:
            head = snake.head
            move = moves.get(snake.id)
            if head is None or move is None:
                dead_ids.add(snake.id)
                continue

            new_head = (head[0] + move.dx, head[1] + move.dy)
            eating = new_head in sim_state.food
            if not (0 <= new_head[0] < sim_state.width and 0 <= new_head[1] < sim_state.height):
                dead_ids.add(snake.id)
                continue

            health = 100 if eating else snake.health - 1
            if new_head in sim_state.hazards:
                health -= sim_state.hazard_damage
            if health <= 0:
                dead_ids.add(snake.id)

            if eating:
                consumed_food.add(new_head)

            new_body = (new_head,) + snake.body
            new_length = snake.length + (1 if eating else 0)
            if not eating:
                new_body = new_body[:-1] if len(new_body) > 1 else new_body

            moved_snakes.append(
                SimSnake(
                    id=snake.id,
                    health=health,
                    length=new_length,
                    body=new_body,
                    length_uncertain=snake.length_uncertain,
                )
            )

        active_snakes = [snake for snake in moved_snakes if snake.id not in dead_ids]
        body_cells: set[tuple[int, int]] = set()
        for snake in active_snakes:
            for cell in snake.body[1:]:
                body_cells.add(cell)

        for snake in active_snakes:
            if snake.head in body_cells:
                dead_ids.add(snake.id)

        active_snakes = [snake for snake in active_snakes if snake.id not in dead_ids]
        heads: dict[tuple[int, int], list[SimSnake]] = {}
        for snake in active_snakes:
            if snake.head is not None:
                heads.setdefault(snake.head, []).append(snake)

        for snakes_at_head in heads.values():
            if len(snakes_at_head) <= 1:
                continue

            def collision_length(snake: SimSnake) -> int:
                if snake.id != sim_state.you_id and snake.length_uncertain:
                    return snake.length + sim_state.width * sim_state.height
                return snake.length

            max_length = max(collision_length(snake) for snake in snakes_at_head)
            longest = [snake for snake in snakes_at_head if collision_length(snake) == max_length]
            if len(longest) == 1:
                survivor_id = longest[0].id
                for snake in snakes_at_head:
                    if snake.id != survivor_id:
                        dead_ids.add(snake.id)
            else:
                for snake in snakes_at_head:
                    dead_ids.add(snake.id)

        survivors = tuple(snake for snake in moved_snakes if snake.id not in dead_ids)
        return SimState(
            width=sim_state.width,
            height=sim_state.height,
            food=frozenset(point for point in sim_state.food if point not in consumed_food),
            hazards=sim_state.hazards,
            snakes=survivors,
            you_id=sim_state.you_id,
            initial_you_length=sim_state.initial_you_length,
            hazard_damage=sim_state.hazard_damage,
        )

    def _sim_head_risk(self, sim_state: SimState, you: SimSnake) -> bool:
        if you.head is None:
            return True
        for snake in sim_state.snakes:
            if snake.id == you.id or snake.head is None:
                continue
            if snake.length < you.length and not snake.length_uncertain:
                continue
            for direction in Direction:
                if (snake.head[0] + direction.dx, snake.head[1] + direction.dy) == you.head:
                    return True
        return False

    def _evaluate_sim_state(self, sim_state: SimState) -> float:
        you = self._sim_you(sim_state)
        if you is None or you.head is None:
            return LOOKAHEAD_DEATH_SCORE

        head_point = Point(x=you.head[0], y=you.head[1])
        obstacle_map = self._sim_obstacle_map(sim_state, free_cell=you.head)
        reachable_space = flood_fill_space(obstacle_map, head_point)
        open_neighbors = count_open_neighbors(obstacle_map, head_point)
        distance_to_wall = wall_distance(head_point, sim_state.width, sim_state.height)
        food_distance = self._sim_shortest_food_distance(sim_state, obstacle_map, you.head)
        board_area = max(1, sim_state.width * sim_state.height)
        board_cap = max(TACTICAL_MIN_SPACE_FLOOR, int(board_area * TACTICAL_BOARD_SPACE_CAP_RATIO))
        min_space = min(max(TACTICAL_MIN_SPACE_FLOOR, you.length + 2), board_cap)

        score = (
            reachable_space * LOOKAHEAD_SPACE_WEIGHT
            + open_neighbors * LOOKAHEAD_OPEN_NEIGHBOR_WEIGHT
            + distance_to_wall * LOOKAHEAD_WALL_DISTANCE_WEIGHT
            + you.health * LOOKAHEAD_HEALTH_WEIGHT
            + max(0, you.length - sim_state.initial_you_length) * LOOKAHEAD_GROWTH_WEIGHT
        )

        if food_distance is not None:
            health_bias = 1.0 + max(0, EAT_HEALTH_BIAS_BASE - you.health) / EAT_HEALTH_BIAS_SCALE
            score += (LOOKAHEAD_FOOD_WEIGHT * health_bias) / (1.0 + food_distance)

        if reachable_space < min_space:
            missing_ratio = (min_space - reachable_space) / max(1, min_space)
            score -= LOOKAHEAD_LOW_SPACE_PENALTY * missing_ratio
        if open_neighbors <= 1:
            score -= LOOKAHEAD_DEAD_END_PENALTY
        if self._sim_head_risk(sim_state, you):
            score -= LOOKAHEAD_HEAD_RISK_PENALTY

        return score

    def _lookahead_value(
        self,
        sim_state: SimState,
        depth: int,
        deadline: float,
        memo: dict[tuple[int, SimState], float],
    ) -> float:
        if time.perf_counter() >= deadline or depth <= 0:
            return self._evaluate_sim_state(sim_state)

        you = self._sim_you(sim_state)
        if you is None:
            return LOOKAHEAD_DEATH_SCORE

        memo_key = (depth, sim_state)
        if memo_key in memo:
            return memo[memo_key]

        our_moves = self._sim_candidate_moves(sim_state, you, is_enemy=False)
        if not our_moves:
            return LOOKAHEAD_DEATH_SCORE

        best_value = LOOKAHEAD_DEATH_SCORE
        for move in our_moves:
            if time.perf_counter() >= deadline:
                break
            value = self._lookahead_value_for_our_move(sim_state, move, depth, deadline, memo)
            best_value = max(best_value, value)

        memo[memo_key] = best_value
        return best_value

    def _lookahead_value_for_our_move(
        self,
        sim_state: SimState,
        our_move: Direction,
        depth: int,
        deadline: float,
        memo: dict[tuple[int, SimState], float],
    ) -> float:
        values: list[float] = []
        for enemy_moves in self._sim_enemy_joint_moves(sim_state):
            if time.perf_counter() >= deadline:
                break
            joint_moves = dict(enemy_moves)
            joint_moves[sim_state.you_id] = our_move
            next_state = self._simulate_joint_move(sim_state, joint_moves)
            child_value = self._lookahead_value(next_state, depth - 1, deadline, memo)
            values.append(child_value)

        if not values:
            return self._evaluate_sim_state(sim_state)

        worst_value = min(values)
        average_value = sum(values) / len(values)
        return (
            worst_value * LOOKAHEAD_WORST_CASE_WEIGHT
            + average_value * LOOKAHEAD_AVERAGE_CASE_WEIGHT
        )

    def _apply_lookahead_scores(
        self,
        game_state: GameState,
        agent_state: AgentState,
        scores: Dict[Direction, float],
    ) -> Dict[Direction, float]:
        if not scores or LOOKAHEAD_MAX_DEPTH <= 0 or LOOKAHEAD_TIME_BUDGET_MS <= 0:
            return scores

        budget_ms = LOOKAHEAD_TIME_BUDGET_MS
        if game_state.game.timeout:
            budget_ms = min(budget_ms, max(5, game_state.game.timeout - 25))
        deadline = time.perf_counter() + budget_ms / 1000.0

        sim_state = self._sim_state_from_game(game_state, agent_state)
        if self._sim_you(sim_state) is None:
            return scores

        values: Dict[Direction, float] = {}
        depth_schedule = [LOOKAHEAD_MAX_DEPTH]
        if LOOKAHEAD_MAX_DEPTH > 2:
            depth_schedule = [2, LOOKAHEAD_MAX_DEPTH]

        for depth in depth_schedule:
            memo: dict[tuple[int, SimState], float] = {}
            pass_values: Dict[Direction, float] = {}
            for move in scores:
                if time.perf_counter() >= deadline:
                    break
                pass_values[move] = self._lookahead_value_for_our_move(
                    sim_state,
                    move,
                    depth,
                    deadline,
                    memo,
                )

            if pass_values and (len(pass_values) == len(scores) or not values):
                values.update(pass_values)
            if len(pass_values) < len(scores):
                break

        if not values:
            return scores

        min_value = min(values.values())
        max_value = max(values.values())
        spread = max_value - min_value
        adjusted = dict(scores)
        for move, value in values.items():
            if value <= LOOKAHEAD_FATAL_SCORE_CUTOFF:
                adjusted[move] *= LOOKAHEAD_FATAL_MULTIPLIER
                continue
            normalized = 0.5 if spread <= 0.000001 else (value - min_value) / spread
            adjusted[move] += normalized * LOOKAHEAD_SCORE_WEIGHT

        return adjusted

    def _best_safety_move(
        self,
        game_state: GameState,
        agent_state: AgentState,
        moves: list[Direction],
    ) -> Direction | None:
        if not moves:
            return None
        return max(moves, key=lambda move: self._move_safety_score(game_state, agent_state, move))

    def _final_safety_override(
        self,
        game_state: GameState,
        agent_state: AgentState,
        scores: Dict[Direction, float],
        chosen: Direction,
    ) -> Direction:
        if chosen not in scores:
            return chosen

        safety_by_move = {move: self._move_safety(game_state, agent_state, move) for move in scores}
        chosen_safety = safety_by_move[chosen]
        health = game_state.you.health or 0
        min_space = self._minimum_viable_space(game_state)
        length = game_state.you.length
        tight_space = max(min_space, int(length * FINAL_TIGHT_SPACE_LENGTH_MULTIPLIER))

        non_collision_moves = [
            move for move, safety in safety_by_move.items() if not safety.risky_tail
        ]
        head_safe_moves = [
            move
            for move, safety in safety_by_move.items()
            if not safety.risky_tail and not safety.head_risk
        ]
        roomy_escape_moves = [
            move
            for move, safety in safety_by_move.items()
            if (
                not safety.risky_tail
                and not safety.head_risk
                and safety.open_neighbors >= FINAL_ESCAPE_OPEN_NEIGHBORS
                and safety.reachable_space >= max(min_space, length + 1)
            )
        ]
        open_escape_moves = [
            move
            for move, safety in safety_by_move.items()
            if (
                not safety.risky_tail
                and not safety.head_risk
                and safety.open_neighbors >= FINAL_ESCAPE_OPEN_NEIGHBORS
            )
        ]

        if chosen_safety.risky_tail:
            return self._best_safety_move(game_state, agent_state, non_collision_moves) or chosen
        if chosen_safety.head_risk:
            return self._best_safety_move(game_state, agent_state, head_safe_moves) or chosen

        urgent_food_escape = (
            chosen_safety.targets_food
            and health <= TACTICAL_FOOD_ESCAPE_HEALTH
            and chosen_safety.reachable_space >= max(1, length)
        )
        if urgent_food_escape:
            return chosen

        escape = self._best_safety_move(game_state, agent_state, roomy_escape_moves)
        if escape is None:
            escape = self._best_safety_move(game_state, agent_state, open_escape_moves)

        if escape is not None and escape != chosen:
            escape_safety = safety_by_move[escape]
            if chosen_safety.open_neighbors <= 1:
                if (
                    escape_safety.reachable_space >= chosen_safety.reachable_space + FINAL_SPACE_ADVANTAGE_MIN
                    or escape_safety.reachable_space >= length * FINAL_DEAD_END_SPACE_MULTIPLIER
                ):
                    return escape

            if chosen_safety.reachable_space < tight_space:
                required_space = max(
                    chosen_safety.reachable_space + FINAL_SPACE_ADVANTAGE_MIN,
                    int(chosen_safety.reachable_space * FINAL_SPACE_ADVANTAGE_RATIO),
                )
                if escape_safety.reachable_space >= required_space:
                    return escape

        best_safe = self._best_safety_move(game_state, agent_state, head_safe_moves)
        if best_safe is not None and best_safe != chosen:
            chosen_score = self._move_safety_score(game_state, agent_state, chosen)
            best_score = self._move_safety_score(game_state, agent_state, best_safe)
            score_gap = max(FINAL_SPACE_ADVANTAGE_MIN, length * 0.5)
            strategy_score_close = scores[best_safe] >= scores[chosen] * 0.75
            if best_score >= chosen_score + score_gap and strategy_score_close:
                return best_safe

        return chosen

    def _strategy_name(self, strategy: Strategy) -> str:
        return type(strategy).__name__

    def _should_switch_focus(
        self,
        current: Strategy,
        candidate: Strategy,
        weights: Dict[Strategy, float],
        health: int,
        enemy_dist: int | None,
        space_ratio: float,
        focused_strategy_turns: int,
    ) -> bool:
        if current is candidate:
            return False

        candidate_is_emergency = (
            isinstance(candidate, StrategyAvoidGettingStuck)
            and space_ratio <= AVOID_STUCK_TIGHT_SPACE_RATIO
        ) or (
            isinstance(candidate, StrategyFlee)
            and enemy_dist is not None
            and enemy_dist <= FLEE_CLOSE_DISTANCE
        ) or (
            isinstance(candidate, StrategyEat)
            and health < EAT_CRITICAL_HEALTH_THRESHOLD
        )
        if candidate_is_emergency:
            return True

        if focused_strategy_turns < STRATEGY_COMMITMENT_MIN_TURNS:
            return False

        return weights[candidate] > weights[current] * STRATEGY_SWITCH_MARGIN

    def _apply_strategy_focus(
        self,
        weights: Dict[Strategy, float],
        agent_state: AgentState,
        health: int,
        enemy_dist: int | None,
        space_ratio: float,
    ) -> None:
        if not weights:
            return

        strategies_by_name = {self._strategy_name(strategy): strategy for strategy in weights}
        dominant = max(weights, key=weights.get)
        focused = strategies_by_name.get(agent_state.focused_strategy or "")

        if focused is None:
            focused = dominant
            agent_state.focused_strategy = self._strategy_name(focused)
            agent_state.focused_strategy_turns = 0
        elif (
            self._should_switch_focus(
                focused,
                dominant,
                weights,
                health,
                enemy_dist,
                space_ratio,
                agent_state.focused_strategy_turns,
            )
        ):
            focused = dominant
            agent_state.focused_strategy = self._strategy_name(focused)
            agent_state.focused_strategy_turns = 0

        weights[focused] *= STRATEGY_COMMITMENT_MULTIPLIER
        agent_state.focused_strategy_turns += 1

    def get_name(self):
        return "Der Snaketürke"

    def get_color(self):
        return "#FF3CEB"

    def get_author(self):
        return "Der Snaketürke"

    def start(self, game_state: GameState):
        """start is called when the battlesnake begins a game"""
        state = self._new_agent_state(game_state)
        self._update_memory(game_state, state)
        self.agent_states[game_state.game.id] = state

    def move(self, game_state: GameState) -> MoveAction:
        """move is called on every turn and returns your next move"""
        game_id = game_state.game.id
        state = self.agent_states.get(game_id)
        if state is None:
            # fallback, falls start() nicht korrekt aufgerufen wurde
            state = self._new_agent_state(game_state)
            self.agent_states[game_id] = state

        self._update_memory(game_state, state)

        # Richte den Status an der echten Körperausrichtung aus.
        state.past_turn = self._infer_direction_from_body(game_state, fallback=state.past_turn)

        possible = self.getPossibleMoves(game_state, state)

        if not possible:
            chosen = self._fallback_direction(game_state, state)
            state.past_turn = chosen
            return MoveAction(move=chosen)

        strat_weights = self.calculateStrategyWeights(game_state, state)

        scores = self.evaluateVotes(game_state, possible, strat_weights, state)
        scores = self._apply_tactical_safety(game_state, state, scores)
        scores = self._apply_food_discipline(game_state, state, scores)
        scores = self._apply_lookahead_scores(game_state, state, scores)
        best_score = max(scores.values())
        best_moves = [move for move, score in scores.items() if score == best_score]
        if len(best_moves) == 1:
            chosen = best_moves[0]
        else:
            chosen = max(best_moves, key=lambda move: self._move_safety_score(game_state, state, move))
        chosen = self._final_safety_override(game_state, state, scores, chosen)

        state.past_turn = chosen
        return MoveAction(move=chosen)

    def _move_safety_score(self, game_state: GameState, agent_state: AgentState, move: Direction) -> float:
        head = game_state.you.head
        if head is None:
            return 0.0

        safety = self._move_safety(game_state, agent_state, move)
        risk_penalty = 0.0
        if safety.head_risk:
            risk_penalty += game_state.board.width * game_state.board.height
        if safety.risky_tail:
            risk_penalty += game_state.you.length
        return (
            (safety.reachable_space * SAFE_MOVE_SPACE_WEIGHT)
            + (safety.wall_distance * SAFE_MOVE_WALL_DISTANCE_WEIGHT)
            + safety.open_neighbors
            - risk_penalty
        )

    def evaluateVotes(
        self,
        game_state: GameState,
        possible_moves: List[Direction],
        strategy_weights: Dict[Strategy, float],
        agent_state: AgentState,
    ) -> Dict[Direction, float]:
        scores: Dict[Direction, float] = {move: 0.0 for move in possible_moves}
        for strategy, weight in strategy_weights.items():
            if weight <= 0:
                continue
            votes = strategy.vote(game_state, possible_moves, agent_state)
            for move, value in votes.items():
                if move in scores:
                    scores[move] += value * weight
        return scores

    def getPossibleMoves(self, game_state: GameState, agent_state: AgentState) -> List[Direction]:
        head = game_state.you.head
        if head is None:
            return []

        cur = agent_state.past_turn
        left_map = {
            Direction.UP: Direction.LEFT,
            Direction.RIGHT: Direction.UP,
            Direction.DOWN: Direction.RIGHT,
            Direction.LEFT: Direction.DOWN,
        }
        right_map = {k: v for v, k in left_map.items()}  # inverse mapping
        straight = cur

        candidates = [left_map[cur], right_map[cur], straight]

        # Opposite directions (verhindert Rückwärtsbewegung)
        opposite_map = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        forbidden_direction = opposite_map[cur]

        obstacle_map = get_obstacle_map(game_state)
        width = game_state.board.width
        height = game_state.board.height

        legal: List[Direction] = []
        for d in candidates:
            if d == forbidden_direction:
                continue
            nx = head.x + d.dx
            ny = head.y + d.dy
            # check bounds
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if obstacle_map[ny, nx]:
                continue
            legal.append(d)

        safety_by_move = {move: self._move_safety(game_state, agent_state, move) for move in legal}
        min_space = self._minimum_viable_space(game_state)

        collision_evasive_legal = self._avoid_constrained_enemy_collisions(game_state, agent_state, legal)
        if len(collision_evasive_legal) < len(legal):
            safe_collision_evasive = [
                move
                for move in collision_evasive_legal
                if not safety_by_move[move].head_risk and not safety_by_move[move].risky_tail
            ]
            return safe_collision_evasive or collision_evasive_legal

        non_collision_legal = [
            move
            for move in legal
            if not safety_by_move[move].risky_tail
        ]
        legal = non_collision_legal or legal

        roomy_escape_legal = [
            move
            for move in legal
            if (
                not safety_by_move[move].head_risk
                and safety_by_move[move].open_neighbors >= FINAL_ESCAPE_OPEN_NEIGHBORS
                and safety_by_move[move].reachable_space >= max(min_space, game_state.you.length + 1)
            )
        ]
        if roomy_escape_legal:
            legal = roomy_escape_legal

        viable_legal = [
            move
            for move in legal
            if (
                safety_by_move[move].reachable_space >= min_space
                or safety_by_move[move].open_neighbors >= FINAL_ESCAPE_OPEN_NEIGHBORS
            )
        ]
        legal = viable_legal or legal

        safer_legal = [move for move in legal if not safety_by_move[move].head_risk]
        legal = safer_legal or legal
        return self._avoid_constrained_enemy_collisions(game_state, agent_state, legal)

    def calculateStrategyWeights(self, game_state: GameState, agent_state: AgentState) -> Dict[Strategy, float]:
        you = game_state.you
        head = you.head

        weights = {strategy: BASE_STRATEGY_WEIGHT for strategy in self.strategies}
        if head is None:
            total = sum(weights.values())
            return {strategy: weight / total for strategy, weight in weights.items()}

        health = you.health or 0
        food_dist = self._closest_food_distance(game_state, agent_state)
        enemy_dist = self._closest_enemy_head_distance(game_state)
        space = self._reachable_space(game_state, head)
        board_area = max(1, game_state.board.width * game_state.board.height)
        space_ratio = space / board_area
        longer_snakes = sum(1 for snake in game_state.board.snakes if snake.id != you.id and snake.length >= you.length)

        eat = next(strategy for strategy in self.strategies if isinstance(strategy, StrategyEat))
        chill = next(strategy for strategy in self.strategies if isinstance(strategy, StrategyChill))
        avoid_stuck = next(strategy for strategy in self.strategies if isinstance(strategy, StrategyAvoidGettingStuck))
        flee = next(strategy for strategy in self.strategies if isinstance(strategy, StrategyFlee))
        head_to_head = next(strategy for strategy in self.strategies if isinstance(strategy, StrategyHeadToHead))
        tail_chase = next(strategy for strategy in self.strategies if isinstance(strategy, StrategyFollowTail))
        explore = next(strategy for strategy in self.strategies if isinstance(strategy, StrategyExplore))
        starve = next(strategy for strategy in self.strategies if isinstance(strategy, StrategyStarve))

        if food_dist is None:
            weights[eat] *= FOOD_MISSING_WEIGHT_MULTIPLIER
        else:
            if health < EAT_HEALTH_LOW_THRESHOLD:
                weights[eat] *= EAT_HEALTH_LOW_MULTIPLIER
            elif health < EAT_HEALTH_MEDIUM_THRESHOLD:
                weights[eat] *= EAT_HEALTH_MEDIUM_MULTIPLIER
            if food_dist <= EAT_CLOSE_FOOD_DISTANCE:
                weights[eat] *= EAT_CLOSE_FOOD_MULTIPLIER
            elif food_dist <= EAT_NEAR_FOOD_DISTANCE:
                weights[eat] *= EAT_NEAR_FOOD_MULTIPLIER

        if enemy_dist is not None:
            if enemy_dist <= FLEE_CLOSE_DISTANCE:
                weights[flee] *= FLEE_CLOSE_DISTANCE_MULTIPLIER
                weights[head_to_head] *= 2.5
            elif enemy_dist <= FLEE_NEAR_DISTANCE:
                weights[flee] *= FLEE_NEAR_DISTANCE_MULTIPLIER
                weights[head_to_head] *= 1.7
            elif enemy_dist <= FLEE_MEDIUM_DISTANCE:
                weights[flee] *= FLEE_MEDIUM_DISTANCE_MULTIPLIER

        if longer_snakes > 0:
            weights[flee] *= BASE_STRATEGY_WEIGHT + FLEE_LONGER_SNAKE_MULTIPLIER_STEP * longer_snakes
            weights[head_to_head] *= 1.0 + FLEE_LONGER_SNAKE_MULTIPLIER_STEP * longer_snakes

        if space_ratio <= min(AVOID_STUCK_TIGHT_SPACE_RATIO, CHILL_DENSE_SPACE_RATIO_THRESHOLD):
            weights[avoid_stuck] *= AVOID_STUCK_TIGHT_SPACE_MULTIPLIER
            weights[chill] *= CHILL_DENSE_SPACE_MULTIPLIER
        elif space_ratio <= max(AVOID_STUCK_MEDIUM_SPACE_RATIO, CHILL_DENSE_SPACE_RATIO_THRESHOLD):
            weights[avoid_stuck] *= AVOID_STUCK_MEDIUM_SPACE_MULTIPLIER
            weights[chill] *= CHILL_DENSE_SPACE_MULTIPLIER
        elif space_ratio >= CHILL_OPEN_SPACE_RATIO_THRESHOLD and health > EAT_HEALTH_MEDIUM_THRESHOLD:
            weights[chill] *= CHILL_OPEN_SPACE_MULTIPLIER
            weights[avoid_stuck] *= 0.8

        if space <= len(you.body) * LOW_SPACE_BODY_FACTOR:
            weights[avoid_stuck] *= LOW_SPACE_SELF_BODY_MULTIPLIER
            weights[chill] *= LOW_SPACE_SELF_BODY_MULTIPLIER
            weights[tail_chase] *= TAIL_CHASE_LOW_SPACE_MULTIPLIER
            weights[flee] *= 1.3
        elif space <= len(you.body) * (LOW_SPACE_BODY_FACTOR + 1):
            weights[avoid_stuck] *= MID_SPACE_SELF_BODY_MULTIPLIER
            weights[tail_chase] *= 1.25

        if space_ratio >= HIGH_SPACE_RATIO and health > EAT_HEALTH_MEDIUM_THRESHOLD:
            weights[chill] *= HIGH_SPACE_CHILL_MULTIPLIER

        if food_dist is not None and space_ratio >= GROWTH_SAFE_SPACE_RATIO:
            weights[eat] *= GROWTH_FOOD_MULTIPLIER
            if health > EAT_HEALTH_MEDIUM_THRESHOLD:
                weights[chill] *= GROWTH_OPEN_SPACE_CHILL_MULTIPLIER

        if health < min(LOW_HEALTH_AGGRESSION_THRESHOLD, EAT_CRITICAL_HEALTH_THRESHOLD):
            weights[eat] *= EAT_CRITICAL_HEALTH_MULTIPLIER
            weights[chill] *= LOW_HEALTH_CHILL_MULTIPLIER
            weights[avoid_stuck] *= 1.2
            weights[tail_chase] *= TAIL_CHASE_LOW_HEALTH_MULTIPLIER

        if food_dist is None:
            weights[tail_chase] *= TAIL_CHASE_NO_FOOD_MULTIPLIER
        elif space_ratio <= AVOID_STUCK_MEDIUM_SPACE_RATIO:
            weights[tail_chase] *= TAIL_CHASE_LOW_SPACE_MULTIPLIER
        else:
            weights[tail_chase] *= 0.55

        stale_area_ratio = self._stale_area_ratio(game_state, agent_state)
        if health < EXPLORE_HEALTH_THRESHOLD:
            weights[explore] *= EXPLORE_LOW_HEALTH_MULTIPLIER
        elif stale_area_ratio >= EXPLORE_UNKNOWN_RATIO_THRESHOLD:
            weights[explore] *= EXPLORE_WEIGHT_MULTIPLIER
        else:
            weights[explore] *= 0.4

        lowest_enemy_health = self._lowest_enemy_health(game_state)
        if not game_state.board.food and not agent_state.remembered_food:
            weights[starve] *= 0.2
        elif health < LOW_HEALTH_AGGRESSION_THRESHOLD:
            weights[starve] *= STARVE_LOW_HEALTH_MULTIPLIER
        elif lowest_enemy_health is not None and lowest_enemy_health <= STARVE_URGENCY_HEALTH:
            weights[starve] *= STARVE_WEIGHT_MULTIPLIER * STARVE_URGENCY_MULTIPLIER
        elif lowest_enemy_health is not None and lowest_enemy_health <= STARVE_MEDIUM_HEALTH:
            weights[starve] *= STARVE_WEIGHT_MULTIPLIER * STARVE_MEDIUM_MULTIPLIER
        else:
            weights[starve] *= 0.05

        self._apply_strategy_focus(weights, agent_state, health, enemy_dist, space_ratio)

        total = sum(weights.values())
        return {strategy: weight / total for strategy, weight in weights.items()}


    def end(self, game_state: GameState):
        if game_state.game.id in self.agent_states:
            del self.agent_states[game_state.game.id]

# ---------------------------------------------------------
# A* Algorithm
# ---------------------------------------------------------
def a_star_wrapper(grid: np.ndarray, start: Point, goal: Point) -> tuple[Direction | None, int]:
    """Converts from battlesnake x-y coords to i-j index-tuples used by a_star()."""
    if start.x == goal.x and start.y == goal.y:
        return None, 0

    path = a_star(grid, (start.y, start.x), (goal.y, goal.x))
    if path is None:
        return None, UNREACHABLE_DISTANCE

    next_pos = path[1]
    result_direction = Direction.from_board_delta((next_pos[1] - start.x, next_pos[0] - start.y))
    return result_direction, len(path) 

def a_star(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]] | None:
    h, w = grid.shape
    open_set, g_score, came_from = [(0, start)], {start: 0}, {}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not grid[nr, nc]:
                neighbor, new_g = (nr, nc), g_score[current] + 1

                if new_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = new_g
                    f_score = new_g + abs(nr - goal[0]) + abs(nc - goal[1])
                    heapq.heappush(open_set, (f_score, neighbor))

    return None



if __name__ == "__main__":
    import sys
    from battlesnake_server import start_server

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <port>")
        sys.exit(1)

    agent = BestAgent()
    port = int(sys.argv[1])

    start_server(agent=agent, port=port)
