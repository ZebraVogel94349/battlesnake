"""Fast immutable Battlesnake/Blackout simulation helpers for tree search.

The simulator deliberately has no dependency on the native Hisss state.  A
``WorldState`` is a small, frozen (and therefore safely shareable/hashable)
snapshot.  Full-information states follow Hisss' standard rules, including its
one-turn delayed physical growth.  Blackout placeholders are represented as
alive snakes with an unknown head; they are never silently treated as dead.

Action indices use the PPO model order throughout this module::

    0 = up, 1 = down, 2 = left, 3 = right

``WorldState.from_game_state`` accepts both the Pydantic objects used by the
starter and equivalent dictionaries.  Passing an RNG asks it to make a cheap
determinization for headless enemies.  Without an RNG, their known visible
cells remain hard blockers while the snake remains an abstract living player.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
import random
from typing import Any, Final, TypeAlias

import numpy as np


Coord: TypeAlias = tuple[int, int]
RngLike: TypeAlias = random.Random | np.random.Generator | Any

UP: Final[int] = 0
DOWN: Final[int] = 1
LEFT: Final[int] = 2
RIGHT: Final[int] = 3
ACTIONS: Final[tuple[int, int, int, int]] = (UP, DOWN, LEFT, RIGHT)
ACTION_DELTAS: Final[tuple[Coord, Coord, Coord, Coord]] = (
    (0, 1),
    (0, -1),
    (-1, 0),
    (1, 0),
)
ACTION_NAMES: Final[tuple[str, str, str, str]] = (
    "up",
    "down",
    "left",
    "right",
)

_OBS_CENTER: Final[int] = 14
_OBS_SIZE: Final[int] = 29
_DEFAULT_MAX_HEALTH: Final[int] = 100
_PLACEMENT_POINTS: Final[tuple[float, float, float, float]] = (2.0, 1.0, 0.0, 0.0)


def _read(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _coord(value: Any, width: int, height: int) -> Coord | None:
    x = _read(value, "x")
    y = _read(value, "y")
    if isinstance(x, int) and isinstance(y, int) and 0 <= x < width and 0 <= y < height:
        return x, y
    return None


def _as_rng(rng: RngLike | int | None) -> RngLike | None:
    if isinstance(rng, (int, np.integer)):
        return random.Random(int(rng))
    return rng


def _random_index(rng: RngLike | None, size: int, fallback: int = 0) -> int:
    if size <= 0:
        raise ValueError("cannot choose from an empty sequence")
    if rng is None:
        return fallback % size
    # random.Random and numpy's Generator both expose random().  Using that
    # common denominator also keeps custom seeded RNGs easy to inject in tests.
    value = float(rng.random())
    return min(size - 1, max(0, int(value * size)))


@dataclass(frozen=True, slots=True)
class SnakeState:
    """One snake in a simulation snapshot.

    ``body`` contains physical, currently known cells.  ``length`` is the
    logical Battlesnake length and can be larger than ``len(body)``; that gap
    is the pending-growth counter used by Hisss.  For a Blackout snake whose
    head is hidden, ``head`` is ``None`` and ``body`` contains only visible
    blocker cells.  ``body_complete=False`` marks any fog-truncated body.
    """

    id: str
    body: tuple[Coord, ...]
    head: Coord | None
    health: int
    length: int
    max_health: int = _DEFAULT_MAX_HEALTH
    alive: bool = True
    body_complete: bool = True
    health_known: bool = True
    length_known: bool = True
    elimination: str | None = None
    death_turn: int | None = None
    killed_by: str | None = None

    @property
    def unknown(self) -> bool:
        return self.head is None or not self.body_complete

    @property
    def tail(self) -> Coord | None:
        if not self.body or not self.body_complete:
            return None
        return self.body[-1]

    @property
    def pending_growth(self) -> int:
        return max(0, self.length - len(self.body))


@dataclass(frozen=True, slots=True)
class WorldState:
    """Immutable simulator state; own snake is always ``snakes[0]``."""

    width: int
    height: int
    turn: int
    snakes: tuple[SnakeState, ...]
    food: frozenset[Coord] = frozenset()
    hazards: frozenset[Coord] = frozenset()
    # Pairs are used instead of a dict so the whole state remains hashable.
    food_spawn_turns: tuple[tuple[Coord, int], ...] = ()
    minimum_food: int = 0
    food_spawn_chance: int = 0
    hazard_damage: int = 0
    wrapped: bool = False
    royale: bool = False
    shrink_every_n_turns: int = 0
    view_radius: int | None = None
    root_alive_count: int = 0

    @classmethod
    def from_game_state(
        cls,
        game_state: Any,
        rng: RngLike | int | None = None,
    ) -> "WorldState":
        """Reconstruct a conservative root state from a Blackout request.

        The private ``you`` object is placed at index zero.  Every opponent
        listed by the protocol stays alive, including ``[-1, -1]``-only
        opponents.  With ``rng=None`` those snakes remain abstract.  Supplying
        a seeded RNG samples a plausible out-of-view head so rollouts can move
        them; the sample is intentionally marked incomplete because Blackout
        does not reveal its true body, health, or length.
        """

        board = _read(game_state, "board")
        game = _read(game_state, "game")
        ruleset = _read(game, "ruleset")
        settings = _read(ruleset, "settings")
        width = int(_read(board, "width", 15))
        height = int(_read(board, "height", 15))
        turn = int(_read(game_state, "turn", 0))
        view_radius_raw = _read(settings, "viewRadius")
        view_radius = None if view_radius_raw is None else int(view_radius_raw)

        you = _read(game_state, "you")
        if you is None:
            raise ValueError("game_state.you is required")
        you_id = str(_read(you, "id", "you"))

        raw_snakes = list(_read(board, "snakes", ()) or ())
        opponents = [s for s in raw_snakes if str(_read(s, "id", "")) != you_id]
        opponents.sort(key=lambda s: str(_read(s, "id", "")))

        def make_snake(raw: Any, *, own: bool) -> SnakeState:
            raw_body = list(_read(raw, "body", ()) or ())
            parsed = [_coord(point, width, height) for point in raw_body]
            contains_hidden = any(point is None for point in parsed)

            # Hisss stores one physical cell for initially stacked snakes.
            # Stable de-duplication recreates that representation while keeping
            # the logical API length for delayed growth.
            body_list: list[Coord] = []
            seen: set[Coord] = set()
            for point in parsed:
                if point is not None and point not in seen:
                    seen.add(point)
                    body_list.append(point)

            raw_head = _coord(_read(raw, "head"), width, height)
            head = raw_head
            if head is None and parsed and parsed[0] is not None:
                head = parsed[0]
            if own and head is None and body_list:
                head = body_list[0]
            if head is not None and (not body_list or body_list[0] != head):
                body_list.insert(0, head)

            reported_length = _read(raw, "length", len(raw_body) or len(body_list) or 1)
            try:
                length = int(reported_length)
            except (TypeError, ValueError):
                length = len(raw_body) or len(body_list) or 1
            length = max(1, length, len(body_list))

            health_raw = _read(raw, "health")
            health_known = own or view_radius is None
            if health_known and isinstance(health_raw, int) and health_raw > 0:
                health = int(health_raw)
            elif own and isinstance(health_raw, int):
                health = max(0, int(health_raw))
            else:
                health = _DEFAULT_MAX_HEALTH
                health_known = False

            elimination_obj = _read(raw, "elimination_event")
            if elimination_obj is None:
                elimination_obj = _read(raw, "elimination")
            elimination = _read(elimination_obj, "cause")
            death_turn = _read(elimination_obj, "turn")
            killed_by = _read(elimination_obj, "by")
            alive = elimination is None and (not own or health > 0)

            body_complete = own or not contains_hidden
            # A missing/invalid first protocol segment means the head itself is
            # unknown even if a later body segment is visible.
            if not own and (not parsed or parsed[0] is None):
                head = None
            return SnakeState(
                id=str(_read(raw, "id", "you" if own else "unknown")),
                body=tuple(body_list),
                head=head,
                health=health,
                length=length,
                alive=alive,
                body_complete=body_complete,
                health_known=health_known,
                # Blackout deliberately rewrites an opponent's protocol
                # ``length`` to the restricted body-list length.  Even an
                # all-visible list can therefore hide logical pending growth
                # (most notably the initial one-cell body with length three).
                # Treat opponent length as exact only outside fog rulesets.
                length_known=own or view_radius is None,
                elimination=None if elimination is None else str(elimination),
                death_turn=None if death_turn is None else int(death_turn),
                killed_by=None if killed_by is None else str(killed_by),
            )

        own_snake = make_snake(you, own=True)
        snakes = (own_snake, *(make_snake(raw, own=False) for raw in opponents))

        food: set[Coord] = set()
        food_turns: list[tuple[Coord, int]] = []
        for raw_food in list(_read(board, "food", ()) or ()):
            point = _coord(raw_food, width, height)
            if point is None:
                continue
            food.add(point)
            spawn_turn = _read(raw_food, "spawn_turn", turn)
            try:
                spawn_turn = int(spawn_turn)
            except (TypeError, ValueError):
                spawn_turn = turn
            food_turns.append((point, spawn_turn))

        hazards = frozenset(
            point
            for raw_hazard in list(_read(board, "hazards", ()) or ())
            if (point := _coord(raw_hazard, width, height)) is not None
        )
        ruleset_name = str(_read(ruleset, "name", "")).lower()
        map_name = str(_read(game, "map", "")).lower()
        royale_settings = _read(settings, "royale")

        world = cls(
            width=width,
            height=height,
            turn=turn,
            snakes=snakes,
            food=frozenset(food),
            hazards=hazards,
            food_spawn_turns=tuple(sorted(set(food_turns))),
            minimum_food=max(0, int(_read(settings, "minimumFood", 0) or 0)),
            food_spawn_chance=max(
                0, min(100, int(_read(settings, "foodSpawnChance", 0) or 0))
            ),
            hazard_damage=max(0, int(_read(settings, "hazardDamagePerTurn", 0) or 0)),
            wrapped=ruleset_name == "wrapped" or map_name == "wrapped",
            royale=ruleset_name == "royale" or map_name == "royale",
            shrink_every_n_turns=max(
                0, int(_read(royale_settings, "shrinkEveryNTurns", 0) or 0)
            ),
            view_radius=view_radius,
            root_alive_count=sum(snake.alive for snake in snakes),
        )
        return determinize_world(world, rng) if rng is not None else world

    @property
    def alive_count(self) -> int:
        return sum(snake.alive for snake in self.snakes)


def _player_index(state: WorldState, player: int | str) -> int:
    if isinstance(player, str):
        for index, snake in enumerate(state.snakes):
            if snake.id == player:
                return index
        raise KeyError(f"unknown snake id: {player}")
    index = int(player)
    if not 0 <= index < len(state.snakes):
        raise IndexError(f"snake index out of range: {index}")
    return index


def _in_bounds(state: WorldState, point: Coord) -> bool:
    return 0 <= point[0] < state.width and 0 <= point[1] < state.height


def _advance(state: WorldState, point: Coord, action: int) -> Coord:
    if action not in ACTIONS:
        raise ValueError(f"invalid model action {action!r}")
    dx, dy = ACTION_DELTAS[action]
    x, y = point[0] + dx, point[1] + dy
    if state.wrapped:
        x %= state.width
        y %= state.height
    return x, y


def determinize_world(
    state: WorldState,
    rng: RngLike | int,
    *,
    minimum_hidden_length: int = 3,
) -> WorldState:
    """Sample heads for headless Blackout opponents.

    This is a deliberately cheap information-set sample, not a claim that the
    hidden body is known.  It chooses an unoccupied cell outside player zero's
    view (preferring a cell adjacent to the first visible enemy segment) and
    retains all visible cells.  Callers can re-determinize at the root of each
    simulation by invoking ``from_game_state`` with their per-simulation RNG.
    """

    resolved_rng = _as_rng(rng)
    if resolved_rng is None:
        raise ValueError("determinize_world requires an RNG or integer seed")
    if not state.snakes:
        return state
    observer_head = state.snakes[0].head
    radius = state.view_radius
    occupied = {point for snake in state.snakes for point in snake.body}
    # Food and a living snake can never share a physical cell.  In Blackout,
    # newly spawned food is announced globally even when it lies outside our
    # view, so it must also constrain a sampled hidden head.
    occupied.update(state.food)
    sampled: list[SnakeState] = []

    for index, snake in enumerate(state.snakes):
        if index == 0 or not snake.alive:
            sampled.append(snake)
            continue

        # The Blackout protocol's restricted ``length`` is only a lower bound.
        # Give every determinization a physically plausible logical length,
        # including the important case where the head (and a one-cell initial
        # body) happens to be fully visible.
        if not snake.length_known:
            snake = replace(
                snake,
                length=max(snake.length, len(snake.body), minimum_hidden_length),
            )
        if snake.head is not None:
            sampled.append(snake)
            continue

        candidates: list[Coord] = []
        if snake.body:
            anchor = snake.body[0]
            for action in ACTIONS:
                point = _advance(state, anchor, action)
                if not _in_bounds(state, point) or point in occupied:
                    continue
                if (
                    observer_head is not None
                    and radius is not None
                    and abs(point[0] - observer_head[0]) + abs(point[1] - observer_head[1])
                    <= radius
                ):
                    continue
                candidates.append(point)

        if not candidates:
            for x in range(state.width):
                for y in range(state.height):
                    point = (x, y)
                    if point in occupied:
                        continue
                    if (
                        observer_head is not None
                        and radius is not None
                        and abs(x - observer_head[0]) + abs(y - observer_head[1]) <= radius
                    ):
                        continue
                    candidates.append(point)
        if not candidates:
            sampled.append(snake)
            continue

        candidates.sort()
        head = candidates[_random_index(resolved_rng, len(candidates))]
        occupied.add(head)
        body = (head, *(point for point in snake.body if point != head))
        sampled.append(
            replace(
                snake,
                head=head,
                body=body,
                length=max(snake.length, len(body), minimum_hidden_length),
                body_complete=False,
                health_known=False,
                length_known=False,
            )
        )
    sampled_state = replace(state, snakes=tuple(sampled))

    # The Blackout request omits old food outside our view, while the engine
    # still maintains ``minimumFood`` on the full board.  Complete each belief
    # particle with hidden old food instead of incorrectly spawning a newly
    # announced replacement on the next simulated turn.
    food = set(sampled_state.food)
    spawn_turns = dict(sampled_state.food_spawn_turns)
    missing_food = max(0, sampled_state.minimum_food - len(food))
    if missing_food and observer_head is not None and radius is not None:
        occupied = {
            point
            for sampled_snake in sampled_state.snakes
            if sampled_snake.alive
            for point in sampled_snake.body
        }
        candidates = [
            (x, y)
            for y in range(sampled_state.height)
            for x in range(sampled_state.width)
            if (x, y) not in occupied
            and (x, y) not in food
            and abs(x - observer_head[0]) + abs(y - observer_head[1]) > radius
        ]
        while candidates and missing_food:
            point = candidates.pop(_random_index(resolved_rng, len(candidates)))
            food.add(point)
            # It is deliberately old: a current-turn spawn would have been
            # globally announced and therefore present in the request.
            spawn_turns[point] = sampled_state.turn - 1
            missing_food -= 1
        sampled_state = replace(
            sampled_state,
            food=frozenset(food),
            food_spawn_turns=tuple(sorted(spawn_turns.items())),
        )
    return sampled_state


def _known_food(state: WorldState, player: int) -> frozenset[Coord]:
    """Food observable by ``player`` without leaking a particle's hidden food."""

    if state.view_radius is None:
        return state.food
    snake = state.snakes[player]
    if snake.head is None:
        return frozenset()
    radius = state.view_radius
    spawn_turns = dict(state.food_spawn_turns)
    return frozenset(
        point
        for point in state.food
        if spawn_turns.get(point) == state.turn
        or abs(point[0] - snake.head[0]) + abs(point[1] - snake.head[1]) <= radius
    )


def legal_actions(state: WorldState, player: int | str = 0) -> tuple[int, ...]:
    """Return non-certain-death actions in PPO model order.

    Like Hisss, this filters walls, starvation/hazard deaths, and occupied body
    cells.  It does not filter possible future head-to-head contests.  A known
    complete tail is enterable exactly when it will vacate this turn.  Unknown
    visible segments are conservatively non-vacating blockers.
    """

    index = _player_index(state, player)
    snake = state.snakes[index]
    if not snake.alive:
        return ()
    if snake.head is None:
        # The action cannot be geometrically checked until determinization, but
        # keeping all four choices preserves the snake as an active opponent.
        return ACTIONS

    result: list[int] = []
    for action in ACTIONS:
        target = _advance(state, snake.head, action)
        if not _in_bounds(state, target):
            continue

        collision = False
        for other in state.snakes:
            if not other.alive or target not in other.body:
                continue
            tail_vacates = (
                other.body_complete
                and other.length_known
                and bool(other.body)
                and target == other.body[-1]
                and other.length <= len(other.body)
            )
            if not tail_vacates:
                collision = True
                break
        if collision:
            continue

        if target not in state.food:
            damage = 1 + (state.hazard_damage if target in state.hazards else 0)
            if snake.health <= damage:
                continue
        result.append(action)
    return tuple(result)


def _joint_action_map(
    state: WorldState,
    joint_actions: Sequence[int] | Mapping[int | str, int],
) -> dict[int, int]:
    if isinstance(joint_actions, Mapping):
        result: dict[int, int] = {}
        for key, action in joint_actions.items():
            result[_player_index(state, key)] = int(action)
        return result

    values = tuple(int(action) for action in joint_actions)
    if len(values) == len(state.snakes):
        return dict(enumerate(values))
    alive_indices = [i for i, snake in enumerate(state.snakes) if snake.alive]
    if len(values) == len(alive_indices):
        return dict(zip(alive_indices, values, strict=True))
    raise ValueError(
        "joint_actions must contain one action per snake (or per living snake)"
    )


def _spawn_food(
    state: WorldState,
    moved_snakes: Sequence[SnakeState],
    remaining_food: set[Coord],
    rng: RngLike | None,
) -> tuple[set[Coord], list[Coord]]:
    n_to_place = max(0, state.minimum_food - len(remaining_food))
    chance_roll = _random_index(
        rng,
        100,
        fallback=(state.turn * 37 + len(remaining_food) * 17 + 11),
    )
    if chance_roll < state.food_spawn_chance:
        n_to_place += 1
    if n_to_place <= 0:
        return remaining_food, []

    forbidden = set(remaining_food)
    for snake in moved_snakes:
        if not snake.alive:
            continue
        forbidden.update(point for point in snake.body if _in_bounds(state, point))
        if snake.head is not None:
            for action in ACTIONS:
                point = _advance(state, snake.head, action)
                if _in_bounds(state, point):
                    forbidden.add(point)

    available = [
        (x, y)
        for y in range(state.height)
        for x in range(state.width)
        if (x, y) not in forbidden
    ]
    spawned: list[Coord] = []
    while available and len(spawned) < n_to_place:
        pick = _random_index(
            rng,
            len(available),
            fallback=state.turn + len(spawned) * 13,
        )
        point = available.pop(pick)
        remaining_food.add(point)
        spawned.append(point)
    return remaining_food, spawned


def _update_royale_hazards(
    state: WorldState,
    hazards: set[Coord],
    new_turn: int,
    rng: RngLike | None,
) -> set[Coord]:
    every = state.shrink_every_n_turns
    if not state.royale or every <= 0 or new_turn < every or new_turn % every:
        return hazards
    safe = [
        (x, y)
        for x in range(state.width)
        for y in range(state.height)
        if (x, y) not in hazards
    ]
    if not safe:
        return hazards
    min_x = min(point[0] for point in safe)
    max_x = max(point[0] for point in safe)
    min_y = min(point[1] for point in safe)
    max_y = max(point[1] for point in safe)
    direction = _random_index(rng, 4, fallback=new_turn // every)
    if direction == 0:
        hazards.update((min_x, y) for y in range(state.height))
    elif direction == 1:
        hazards.update((max_x, y) for y in range(state.height))
    elif direction == 2:
        hazards.update((x, min_y) for x in range(state.width))
    else:
        hazards.update((x, max_y) for x in range(state.width))
    return hazards


def step_world(
    state: WorldState,
    joint_actions: Sequence[int] | Mapping[int | str, int],
    rng: RngLike | int | None = None,
    spawn_food: bool = False,
) -> WorldState:
    """Apply one simultaneous joint move and return a new state.

    The input is never mutated.  Actions may be a full sequence aligned with
    ``state.snakes`` or a mapping keyed by index/id.  ``spawn_food=False`` is a
    useful deterministic MCTS default; when enabled, minimum food and spawn
    chance follow Hisss and all random choices are driven by the supplied RNG.
    """

    action_map = _joint_action_map(state, joint_actions)
    resolved_rng = _as_rng(rng)
    moved: list[SnakeState] = []
    eaten: set[Coord] = set()

    # Movement/tail removal uses the old logical length.  Eating increments
    # length only afterwards, which is Hisss' delayed-growth rule.
    for index, snake in enumerate(state.snakes):
        if not snake.alive or snake.head is None:
            moved.append(snake)
            continue
        action = action_map.get(index, UP)
        new_head = _advance(state, snake.head, action)
        new_body = (new_head, *snake.body)
        if snake.length < len(new_body):
            new_body = new_body[:-1]

        health = max(0, snake.health - 1)
        if new_head in state.hazards:
            health = max(0, health - state.hazard_damage)
        length = snake.length
        if new_head in state.food:
            eaten.add(new_head)
            health = snake.max_health
            length += 1
        moved.append(
            replace(
                snake,
                body=new_body,
                head=new_head,
                health=health,
                length=length,
            )
        )

    remaining_food = set(state.food) - eaten
    spawned: list[Coord] = []
    if spawn_food:
        remaining_food, spawned = _spawn_food(
            state, moved, remaining_food, resolved_rng
        )

    # New heads are intentionally excluded from body occupancy; head-to-head
    # resolution below handles them.  Headless abstract snakes have no special
    # head cell, so every visible segment remains a conservative blocker.
    body_occupancy: list[set[Coord]] = []
    for snake in moved:
        if snake.head is None:
            body_occupancy.append(set(snake.body))
        else:
            body_occupancy.append(set(snake.body[1:]))

    known_lengths = [snake.length for snake in moved if snake.alive and snake.length_known]
    conservative_length = max(known_lengths, default=1)

    def collision_length(snake: SnakeState) -> int:
        if snake.length_known:
            return snake.length
        return max(snake.length, conservative_length)

    deaths: dict[int, tuple[str, str | None]] = {}
    for index, snake in enumerate(moved):
        if not snake.alive or snake.head is None:
            continue
        head = snake.head
        if not _in_bounds(state, head):
            deaths[index] = ("wall-collision", None)
            continue
        if snake.health <= 0:
            deaths[index] = ("out-of-health", None)
            continue
        if head in body_occupancy[index]:
            deaths[index] = ("snake-self-collision", None)
            continue

        for other_index, other in enumerate(moved):
            if other_index == index or not other.alive:
                continue
            if head in body_occupancy[other_index]:
                deaths[index] = ("snake-collision", other.id)
                break
            if (
                other.head is not None
                and head == other.head
                and collision_length(snake) <= collision_length(other)
            ):
                deaths[index] = ("head-collision", other.id)
                break

    final_snakes: list[SnakeState] = []
    for index, snake in enumerate(moved):
        death = deaths.get(index)
        if death is None:
            final_snakes.append(snake)
        else:
            final_snakes.append(
                replace(
                    snake,
                    alive=False,
                    elimination=death[0],
                    death_turn=state.turn,
                    killed_by=death[1],
                )
            )

    new_turn = state.turn + 1
    hazards = _update_royale_hazards(
        state, set(state.hazards), new_turn, resolved_rng
    )
    old_spawn_turns = {
        point: spawn_turn
        for point, spawn_turn in state.food_spawn_turns
        if point in remaining_food
    }
    for point in spawned:
        old_spawn_turns[point] = new_turn

    return replace(
        state,
        turn=new_turn,
        snakes=tuple(final_snakes),
        food=frozenset(remaining_food),
        hazards=frozenset(hazards),
        food_spawn_turns=tuple(sorted(old_spawn_turns.items())),
    )


def _blocked_for_space(state: WorldState) -> set[Coord]:
    blocked: set[Coord] = set()
    for snake in state.snakes:
        if not snake.alive:
            continue
        blocked.update(snake.body)
        if (
            snake.body_complete
            and snake.length_known
            and snake.body
            and snake.length <= len(snake.body)
        ):
            blocked.discard(snake.body[-1])
    return blocked


def _flood_area(state: WorldState, start: Coord, blocked: set[Coord]) -> int:
    if not _in_bounds(state, start):
        return 0
    todo: deque[Coord] = deque((start,))
    seen = {start}
    while todo:
        current = todo.popleft()
        for action in ACTIONS:
            nxt = _advance(state, current, action)
            if not _in_bounds(state, nxt) or nxt in blocked or nxt in seen:
                continue
            seen.add(nxt)
            todo.append(nxt)
    return len(seen)


def heuristic_action_priors(
    state: WorldState,
    player: int | str = 0,
) -> np.ndarray:
    """Return inexpensive geometry/food priors as a length-four distribution."""

    index = _player_index(state, player)
    snake = state.snakes[index]
    priors = np.zeros(4, dtype=np.float32)
    legal = legal_actions(state, index)
    if not legal:
        priors.fill(0.25)
        return priors
    if snake.head is None:
        priors[list(legal)] = 1.0 / len(legal)
        return priors

    blocked = _blocked_for_space(state)
    blocked.discard(snake.head)
    known_food = _known_food(state, index)
    center = ((state.width - 1) / 2.0, (state.height - 1) / 2.0)
    scores = np.full(4, -np.inf, dtype=np.float64)
    hungry = max(0.0, min(1.0, (55.0 - snake.health) / 45.0))

    for action in legal:
        target = _advance(state, snake.head, action)
        exits = 0
        for next_action in ACTIONS:
            nxt = _advance(state, target, next_action)
            if _in_bounds(state, nxt) and nxt not in blocked:
                exits += 1

        # With the old head removed from ``blocked``, all legal neighbouring
        # targets belong to the same connected component.  A flood-fill area
        # term was therefore constant across actions and cancelled in softmax,
        # while costing roughly a millisecond per call.
        score = 0.28 * exits
        score -= 0.018 * (abs(target[0] - center[0]) + abs(target[1] - center[1]))
        if target in state.hazards:
            score -= 0.35 + 2.0 * max(
                0.0,
                (state.hazard_damage + 8 - snake.health)
                / max(1.0, state.hazard_damage + 8),
            )
        if target in known_food:
            score += 0.5 + 2.2 * hungry
        elif known_food and hungry > 0:
            old_distance = min(
                abs(snake.head[0] - food[0]) + abs(snake.head[1] - food[1])
                for food in known_food
            )
            new_distance = min(
                abs(target[0] - food[0]) + abs(target[1] - food[1])
                for food in known_food
            )
            score += 0.32 * hungry * (old_distance - new_distance)

        for other in state.snakes:
            if not other.alive or other.head is None or other.id == snake.id:
                continue
            if abs(other.head[0] - target[0]) + abs(other.head[1] - target[1]) != 1:
                continue
            other_length = other.length
            if not other.length_known:
                other_length = max(other_length, snake.length)
            score += 0.35 if snake.length > other_length else -1.7
        scores[action] = score

    finite = scores[np.isfinite(scores)]
    shifted = np.exp(scores[list(legal)] - float(np.max(finite)))
    shifted_sum = float(np.sum(shifted))
    if shifted_sum <= 0 or not math.isfinite(shifted_sum):
        priors[list(legal)] = 1.0 / len(legal)
    else:
        priors[list(legal)] = (shifted / shifted_sum).astype(np.float32)
    return priors


def is_terminal(state: WorldState, player: int | str = 0) -> bool:
    """Return game terminal status without losing hidden opponents.

    Official Battlesnake (and the evaluation harness with all actions legal)
    ends on elimination, not merely because a living snake has no safe move.
    Such a snake will be eliminated by the following joint step.
    """

    _player_index(state, player)  # validate even though terminal is global
    if len(state.snakes) <= 1:
        return state.alive_count == 0
    return state.alive_count <= 1


def evaluate_world(
    state: WorldState,
    player: int | str = 0,
    root_alive_count: int | None = None,
) -> float:
    """Evaluate placement/survival/territory in ``[-1, 1]`` for MCTS leaves.

    Exact outcomes use the evaluator's ``(2, 1, 0, 0)`` placement awards,
    shifted by one onto the value scale.  Snakes eliminated on the same turn
    split the occupied placement slots.  Non-terminal heuristic values are
    capped below the exact win bound.  Abstract hidden opponents count as
    active and therefore can neither create a false terminal win nor a free
    elimination bonus.
    """

    index = _player_index(state, player)
    snake = state.snakes[index]
    if not snake.alive:
        # Rank every retained contestant by survival first and elimination
        # turn second, exactly like ``evaluate_inference._placement_points``.
        # Pre-root eliminations omitted by the Battlesnake protocol are below
        # every snake present at the root and therefore cannot change this
        # player's occupied slot.  ``step_world`` retains all root contestants,
        # so simultaneous and earlier/later rollout deaths remain observable.
        if snake.death_turn is None:
            return -1.0
        ranks = tuple(
            math.inf
            if other.alive
            else (
                float(other.death_turn)
                if other.death_turn is not None
                else -math.inf
            )
            for other in state.snakes
        )
        death_rank = float(snake.death_turn)
        higher = sum(rank > death_rank for rank in ranks)
        tied = sum(rank == death_rank for rank in ranks)
        if tied <= 0:  # Defensive guard; the selected snake is normally one.
            return -1.0
        occupied_awards = tuple(
            _PLACEMENT_POINTS[slot] if slot < len(_PLACEMENT_POINTS) else 0.0
            for slot in range(higher, higher + tied)
        )
        return float(sum(occupied_awards) / tied - 1.0)

    if state.alive_count <= 1:
        return 1.0
    if snake.head is None:
        return 0.0
    # The API-faithful engine has not eliminated an enclosed/starving snake
    # until the next simultaneous step, but every submitted action is already
    # a certain death.  Backing up a soft geometry score here lets MCTS prefer a
    # forced loss over a genuinely surviving line.
    if not legal_actions(state, index):
        # With only one opponent left, elimination still guarantees second
        # place (one point, hence value zero); it is not equivalent to dying
        # third or fourth.  In larger fields a forced next-turn death remains
        # an exact worst-placement value.
        return 0.0 if state.alive_count <= 2 else -1.0

    blocked = _blocked_for_space(state)
    blocked.discard(snake.head)
    own_area = _flood_area(state, snake.head, blocked)
    opponent_areas: list[int] = []
    opponents = [other for i, other in enumerate(state.snakes) if i != index and other.alive]
    for other in opponents:
        if other.head is None:
            # No information means no territory advantage, rather than the
            # optimistic zero-area assumption that caused false MCTS wins.
            opponent_areas.append(own_area)
            continue
        other_blocked = set(blocked)
        other_blocked.discard(other.head)
        opponent_areas.append(_flood_area(state, other.head, other_blocked))
    opponent_area = sum(opponent_areas) / len(opponent_areas) if opponent_areas else 0.0
    board_area = max(1, state.width * state.height)
    area_advantage = math.tanh((own_area - opponent_area) * 4.0 / board_area)

    opponent_lengths = [
        max(other.length, snake.length) if not other.length_known else other.length
        for other in opponents
    ]
    average_opponent_length = (
        sum(opponent_lengths) / len(opponent_lengths)
        if opponent_lengths
        else snake.length
    )
    length_advantage = math.tanh((snake.length - average_opponent_length) / 4.0)
    health_score = 2.0 * min(1.0, snake.health / max(1, snake.max_health)) - 1.0
    mobility_score = (len(legal_actions(state, index)) - 2.0) / 2.0

    food_score = 0.0
    known_food = _known_food(state, index)
    if known_food and snake.health < 55:
        distance = min(
            abs(snake.head[0] - food[0]) + abs(snake.head[1] - food[1])
            for food in known_food
        )
        food_score = 1.0 - min(1.0, distance / max(1.0, snake.health - 1.0))
    if snake.head in state.hazards:
        health_score -= min(1.0, state.hazard_damage / max(1.0, snake.health))

    initial_alive = state.root_alive_count if root_alive_count is None else root_alive_count
    initial_alive = max(1, int(initial_alive or state.alive_count))
    elimination_score = 0.0
    if initial_alive > 1:
        elimination_score = max(0.0, (initial_alive - state.alive_count) / (initial_alive - 1))

    value = (
        0.42 * area_advantage
        + 0.20 * length_advantage
        + 0.12 * health_score
        + 0.12 * mobility_score
        + 0.06 * food_score
        + 0.08 * elimination_score
    )
    value = max(-0.95, min(0.95, value))
    # In a two-snake non-terminal state, second place is already secured.
    # Preserve that lower bound even when the geometry heuristic is negative.
    if state.alive_count <= 2:
        value = max(0.0, value)
    return float(value)


def encode_world_observation(
    state: WorldState,
    player: int | str = 0,
    view_radius: int | None = 5,
) -> np.ndarray:
    """Project a world into PPO4's exact nine-channel ``(9, 29, 29)`` layout.

    Fog uses Manhattan distance around the selected snake.  Old food is only
    shown in view, while food whose recorded spawn turn equals ``state.turn``
    is globally announced, matching Hisss Blackout.  Passing ``view_radius=None``
    uses the radius stored in the state (and full visibility when that too is
    ``None``).
    """

    index = _player_index(state, player)
    snake = state.snakes[index]
    obs = np.zeros((9, _OBS_SIZE, _OBS_SIZE), dtype=np.float32)
    if not snake.alive or snake.head is None:
        return obs
    if state.width > 15 or state.height > 15:
        raise ValueError("PPO4 observation supports boards up to 15x15")

    radius = state.view_radius if view_radius is None else view_radius
    head_x, head_y = snake.head
    offset_x = _OBS_CENTER - head_x
    offset_y = _OBS_CENTER - head_y

    def visible(point: Coord) -> bool:
        return radius is None or abs(point[0] - head_x) + abs(point[1] - head_y) <= radius

    def tensor_point(point: Coord) -> tuple[int, int] | None:
        tx, ty = point[0] + offset_x, point[1] + offset_y
        if 0 <= tx < _OBS_SIZE and 0 <= ty < _OBS_SIZE:
            return tx, ty
        return None

    spawn_turns = dict(state.food_spawn_turns)
    for food in state.food:
        if not visible(food) and spawn_turns.get(food) != state.turn:
            continue
        if (point := tensor_point(food)) is not None:
            obs[0, point[0], point[1]] = 1.0

    obs[1].fill(-1.0)
    obs[
        1,
        offset_x : offset_x + state.width,
        offset_y : offset_y + state.height,
    ] = 1.0

    for body_index, body_point in enumerate(snake.body):
        if (point := tensor_point(body_point)) is not None:
            obs[2, point[0], point[1]] = (snake.length - body_index) / 10.0
    obs[3, _OBS_CENTER, _OBS_CENTER] = 1.0
    obs[4].fill(snake.health / float(_DEFAULT_MAX_HEALTH))
    if snake.body_complete and snake.body:
        if (point := tensor_point(snake.body[-1])) is not None:
            obs[5, point[0], point[1]] = 1.0

    for other_index, other in enumerate(state.snakes):
        if other_index == index or not other.alive:
            continue
        for body_point in other.body:
            if visible(body_point) and (point := tensor_point(body_point)) is not None:
                obs[6, point[0], point[1]] = 1.0
        if other.head is not None and visible(other.head):
            if (point := tensor_point(other.head)) is not None:
                obs[7, point[0], point[1]] = 1.0

    if radius is None:
        obs[8].fill(1.0)
    else:
        for dx in range(-_OBS_CENTER, _OBS_CENTER + 1):
            remaining = min(_OBS_CENTER, radius - abs(dx))
            if remaining < 0:
                continue
            obs[
                8,
                _OBS_CENTER + dx,
                _OBS_CENTER - remaining : _OBS_CENTER + remaining + 1,
            ] = 1.0
    return obs


def state_key(state: WorldState) -> WorldState:
    """Return a hashable transposition key (the immutable state itself)."""

    return state


__all__ = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "ACTIONS",
    "ACTION_DELTAS",
    "ACTION_NAMES",
    "SnakeState",
    "WorldState",
    "determinize_world",
    "legal_actions",
    "step_world",
    "heuristic_action_priors",
    "is_terminal",
    "evaluate_world",
    "encode_world_observation",
    "state_key",
]
