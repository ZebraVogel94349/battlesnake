"""History-conditioned Blackout belief particles for :mod:`ppo_mcts_v2`.

This module is intentionally separate from ``mcts_simulator.py`` so the
deployed tournament agent can keep importing its byte-identical simulator.
The tracker adds three pieces that the original root determinizer lacks:

* remembered food and last-seen opponent information across real turns;
* full, connected hypotheses for fogged opponent bodies;
* a small particle filter which advances last turn's root particles through
  our committed action and rejects worlds inconsistent with the new request.

Every ``WorldState`` returned here is immutable.  Search may therefore reuse a
reservoir entry directly without risking cross-simulation mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Iterable, Sequence

import numpy as np

from battlesnake_types import GameState, Point, Snake
from mcts_simulator import (
    ACTIONS,
    ACTION_DELTAS,
    SnakeState,
    WorldState,
    heuristic_action_priors,
    legal_actions,
    state_key,
    step_world,
)


Coord = tuple[int, int]


def _random(rng: np.random.Generator) -> float:
    return float(rng.random())


def _choice(
    rng: np.random.Generator,
    values: Sequence[int] | Sequence[Coord],
    probabilities: Sequence[float] | np.ndarray | None = None,
):
    if not values:
        raise ValueError("cannot sample an empty sequence")
    if len(values) == 1:
        return values[0]
    index = int(rng.choice(len(values), p=probabilities))
    return values[index]


def _valid(point: Point | None, width: int, height: int) -> Coord | None:
    if point is None or not (0 <= point.x < width and 0 <= point.y < height):
        return None
    return point.x, point.y


def _stable_body(snake: Snake, width: int, height: int) -> tuple[Coord, ...]:
    result: list[Coord] = []
    seen: set[Coord] = set()
    for raw in snake.body:
        point = _valid(raw, width, height)
        if point is not None and point not in seen:
            seen.add(point)
            result.append(point)
    return tuple(result)


def _visible(point: Coord, head: Coord | None, radius: int | None) -> bool:
    if head is None:
        return False
    return radius is None or abs(point[0] - head[0]) + abs(point[1] - head[1]) <= radius


def _advance(state: WorldState, point: Coord, action: int) -> Coord:
    dx, dy = ACTION_DELTAS[action]
    x, y = point[0] + dx, point[1] + dy
    if state.wrapped:
        x %= state.width
        y %= state.height
    return x, y


def _neighbours(
    point: Coord,
    width: int,
    height: int,
    wrapped: bool,
) -> tuple[Coord, ...]:
    result: list[Coord] = []
    for dx, dy in ACTION_DELTAS:
        x, y = point[0] + dx, point[1] + dy
        if wrapped:
            x %= width
            y %= height
        if 0 <= x < width and 0 <= y < height and (x, y) not in result:
            result.append((x, y))
    return tuple(result)


def _distance(
    first: Coord,
    second: Coord,
    width: int,
    height: int,
    wrapped: bool,
) -> int:
    dx = abs(first[0] - second[0])
    dy = abs(first[1] - second[1])
    if wrapped:
        dx = min(dx, width - dx)
        dy = min(dy, height - dy)
    return dx + dy


@dataclass(slots=True)
class OpponentMemory:
    last_head: Coord | None = None
    last_head_turn: int = -1
    max_length_lower_bound: int = 3
    last_indexed_body: tuple[Coord | None, ...] = ()


@dataclass(slots=True)
class BeliefDiagnostics:
    propagated: int = 0
    retained: int = 0
    fresh_samples: int = 0
    reused_samples: int = 0
    body_fallbacks: int = 0
    known_food: int = 0
    reservoir: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "propagated": self.propagated,
            "retained": self.retained,
            "fresh_samples": self.fresh_samples,
            "reused_samples": self.reused_samples,
            "body_fallbacks": self.body_fallbacks,
            "known_food": self.known_food,
            "reservoir": self.reservoir,
        }


@dataclass(slots=True)
class BeliefTrackerV2:
    """Per-game particle filter and compact Blackout memory."""

    max_reservoir: int = 96
    reuse_fraction: float = 0.55
    minimum_hidden_length: int = 3
    opponent_length_slack: int = 4
    body_attempts: int = 48
    opponents: dict[str, OpponentMemory] = field(default_factory=dict)
    known_food: dict[Coord, int] = field(default_factory=dict)
    diagnostics: BeliefDiagnostics = field(default_factory=BeliefDiagnostics)
    _reservoir: list[WorldState] = field(default_factory=list)
    _prepared_particles: list[WorldState] = field(default_factory=list)
    _last_turn: int = -1
    _last_action: int | None = None
    _current_turn: int = -1

    def reset(self) -> None:
        self.opponents.clear()
        self.known_food.clear()
        self.diagnostics = BeliefDiagnostics()
        self._reservoir.clear()
        self._prepared_particles.clear()
        self._last_turn = -1
        self._last_action = None
        self._current_turn = -1

    def prepare(self, game_state: GameState, rng: np.random.Generator) -> None:
        """Advance the previous reservoir and ingest the current observation."""

        self.diagnostics = BeliefDiagnostics()
        retained: list[WorldState] = []
        if (
            self._last_action is not None
            and self._last_turn + 1 == game_state.turn
            and self._reservoir
        ):
            for particle in self._reservoir:
                if not particle.snakes or not particle.snakes[0].alive:
                    continue
                joint = self._sample_joint_actions(particle, self._last_action, rng)
                advanced = step_world(particle, joint, rng=rng, spawn_food=True)
                self.diagnostics.propagated += 1
                if self._matches_observation(advanced, game_state):
                    retained.append(advanced)
                    if len(retained) >= self.max_reservoir:
                        break
        self._prepared_particles = retained
        self.diagnostics.retained = len(retained)
        self._observe_history(game_state)
        self._current_turn = game_state.turn
        self.diagnostics.known_food = len(self.known_food)

    def sample(
        self,
        game_state: GameState,
        rng: np.random.Generator,
    ) -> WorldState:
        if self._current_turn != game_state.turn:
            raise RuntimeError("prepare() must be called before sampling a turn")
        if self._prepared_particles and _random(rng) < self.reuse_fraction:
            self.diagnostics.reused_samples += 1
            return _choice(rng, self._prepared_particles)
        # Greedy body construction can occasionally pick a placeholder cell
        # that would actually be visible.  Reject such a hypothesis against the
        # complete current API projection instead of leaking it into search.
        last: WorldState | None = None
        for _ in range(8):
            self.diagnostics.fresh_samples += 1
            last = self._fresh_particle(game_state, rng)
            if self._matches_observation(last, game_state):
                return last
        # The base simulator's conservative determinization preserves every
        # visible cell and is a safe final fallback when connected completion is
        # over-constrained by sparse protocol data.
        fallback = WorldState.from_game_state(game_state, rng=rng)
        return fallback if self._matches_observation(fallback, game_state) else last

    def commit(
        self,
        sampled_particles: Iterable[WorldState],
        action: int,
        turn: int,
    ) -> None:
        """Keep a bounded, de-duplicated root reservoir for the next real turn."""

        unique: dict[WorldState, WorldState] = {}
        for particle in sampled_particles:
            unique.setdefault(state_key(particle), particle)
            if len(unique) >= self.max_reservoir:
                break
        if len(unique) < self.max_reservoir:
            for particle in self._prepared_particles:
                unique.setdefault(state_key(particle), particle)
                if len(unique) >= self.max_reservoir:
                    break
        self._reservoir = list(unique.values())
        self._last_action = int(action)
        self._last_turn = int(turn)
        self.diagnostics.reservoir = len(self._reservoir)

    def _observe_history(self, game_state: GameState) -> None:
        width, height = game_state.board.width, game_state.board.height
        own_head = _valid(game_state.you.head, width, height)
        radius = game_state.game.ruleset.settings.viewRadius
        observed_food = {
            (food.x, food.y): int(food.spawn_turn)
            for food in game_state.board.food
            if 0 <= food.x < width and 0 <= food.y < height
        }
        for point in tuple(self.known_food):
            if _visible(point, own_head, radius) and point not in observed_food:
                self.known_food.pop(point, None)
        if own_head is not None:
            self.known_food.pop(own_head, None)
        self.known_food.update(observed_food)

        current_ids = {snake.id for snake in game_state.board.snakes}
        for snake_id in tuple(self.opponents):
            if snake_id not in current_ids:
                self.opponents.pop(snake_id, None)
        for snake in game_state.board.snakes:
            if snake.id == game_state.you.id:
                continue
            memory = self.opponents.setdefault(snake.id, OpponentMemory())
            indexed = tuple(_valid(point, width, height) for point in snake.body)
            head = _valid(snake.head, width, height)
            if head is not None:
                memory.last_head = head
                memory.last_head_turn = game_state.turn
            memory.max_length_lower_bound = max(
                memory.max_length_lower_bound,
                int(snake.length),
                len(indexed),
                sum(point is not None for point in indexed),
            )
            memory.last_indexed_body = indexed

    def _sample_joint_actions(
        self,
        particle: WorldState,
        our_action: int,
        rng: np.random.Generator,
    ) -> list[int]:
        joint = [0] * len(particle.snakes)
        joint[0] = int(our_action)
        for index in range(1, len(particle.snakes)):
            actions = legal_actions(particle, index)
            if not actions:
                continue
            priors = heuristic_action_priors(particle, index).astype(np.float64)
            selected = priors[list(actions)]
            total = float(selected.sum())
            probabilities = (
                None if not math.isfinite(total) or total <= 0 else selected / total
            )
            joint[index] = int(_choice(rng, actions, probabilities))
        return joint

    def _fresh_particle(
        self,
        game_state: GameState,
        rng: np.random.Generator,
    ) -> WorldState:
        base = WorldState.from_game_state(game_state, rng=None)
        raw_by_id = {snake.id: snake for snake in game_state.board.snakes}
        occupied: set[Coord] = set(base.snakes[0].body)
        sampled_snakes: list[SnakeState] = [base.snakes[0]]

        for original in base.snakes[1:]:
            raw = raw_by_id.get(original.id)
            if raw is None or not original.alive:
                sampled_snakes.append(original)
                continue
            forbidden = occupied | set(base.food) | set(self.known_food)
            sampled = self._sample_opponent(
                base,
                original,
                raw,
                forbidden,
                rng,
                game_state.turn,
            )
            sampled_snakes.append(sampled)
            occupied.update(sampled.body)

        food = {
            point
            for point in set(base.food) | set(self.known_food)
            if point not in occupied
        }
        spawn_turns = dict(base.food_spawn_turns)
        for point, spawn_turn in self.known_food.items():
            if point in food:
                spawn_turns[point] = spawn_turn

        own_head = sampled_snakes[0].head
        hidden_cells = [
            (x, y)
            for y in range(base.height)
            for x in range(base.width)
            if (x, y) not in occupied
            and (x, y) not in food
            and not _visible((x, y), own_head, base.view_radius)
        ]
        while len(food) < base.minimum_food and hidden_cells:
            point = _choice(rng, hidden_cells)
            hidden_cells.remove(point)
            food.add(point)
            spawn_turns[point] = base.turn - 1

        return replace(
            base,
            snakes=tuple(sampled_snakes),
            food=frozenset(food),
            food_spawn_turns=tuple(
                sorted((point, turn) for point, turn in spawn_turns.items() if point in food)
            ),
        )

    def _sample_opponent(
        self,
        state: WorldState,
        original: SnakeState,
        raw: Snake,
        occupied: set[Coord],
        rng: np.random.Generator,
        turn: int,
    ) -> SnakeState:
        memory = self.opponents.setdefault(original.id, OpponentMemory())
        indexed = tuple(_valid(point, state.width, state.height) for point in raw.body)
        fixed = {index: point for index, point in enumerate(indexed) if point is not None}
        raw_head = _valid(raw.head, state.width, state.height)
        if raw_head is not None:
            fixed[0] = raw_head

        head = raw_head
        if head is None:
            candidates = self._head_candidates(
                state,
                memory,
                fixed,
                occupied,
                turn,
            )
            if candidates:
                head = _choice(rng, candidates)
            elif original.head is not None:
                head = original.head
            else:
                # A fully occupied board is already pathological.  Keeping the
                # abstract snake is safer than inventing an in-view head.
                return original

        slots = max(1, len(indexed))
        if indexed and all(point == indexed[0] for point in indexed) and indexed[0] is not None:
            slots = 1
            fixed = {0: indexed[0]}
        body = self._sample_body_path(
            head,
            slots,
            fixed,
            occupied,
            state,
            rng,
        )
        if body is None:
            self.diagnostics.body_fallbacks += 1
            visible = [point for point in indexed if point is not None and point != head]
            body = tuple(dict.fromkeys((head, *visible)))

        lower_length = max(
            self.minimum_hidden_length,
            int(raw.length),
            memory.max_length_lower_bound,
            len(body),
        )
        slack = min(
            self.opponent_length_slack,
            max(0, state.width * state.height - lower_length),
        )
        # A short geometric tail avoids systematically assuming that every
        # lower-bound protocol length is exact.
        extra = 0
        while extra < slack and _random(rng) < 0.28:
            extra += 1
        length = lower_length + extra
        health = int(rng.integers(35, original.max_health + 1))
        return replace(
            original,
            head=head,
            body=body,
            health=health,
            length=length,
            body_complete=True,
            health_known=True,
            length_known=True,
        )

    def _head_candidates(
        self,
        state: WorldState,
        memory: OpponentMemory,
        fixed: dict[int, Coord],
        occupied: set[Coord],
        turn: int,
    ) -> list[Coord]:
        observer = state.snakes[0].head
        candidates: list[Coord] = []
        elapsed = turn - memory.last_head_turn
        for y in range(state.height):
            for x in range(state.width):
                point = (x, y)
                if point in occupied or point in state.food:
                    continue
                if _visible(point, observer, state.view_radius):
                    continue
                if memory.last_head is not None and elapsed >= 0:
                    distance = _distance(
                        memory.last_head,
                        point,
                        state.width,
                        state.height,
                        state.wrapped,
                    )
                    if distance > elapsed or (distance - elapsed) % 2:
                        continue
                possible = True
                for index, body_point in fixed.items():
                    distance = _distance(
                        point,
                        body_point,
                        state.width,
                        state.height,
                        state.wrapped,
                    )
                    if distance > index or (distance - index) % 2:
                        possible = False
                        break
                if possible:
                    candidates.append(point)
        return candidates

    def _sample_body_path(
        self,
        head: Coord,
        slots: int,
        fixed: dict[int, Coord],
        occupied: set[Coord],
        state: WorldState,
        rng: np.random.Generator,
    ) -> tuple[Coord, ...] | None:
        fixed = dict(fixed)
        fixed[0] = head
        for _ in range(self.body_attempts):
            path = [head]
            used = {head}
            valid = True
            for index in range(1, slots):
                required = fixed.get(index)
                if required is not None:
                    choices = [required]
                else:
                    choices = list(
                        _neighbours(
                            path[-1], state.width, state.height, state.wrapped
                        )
                    )
                    rng.shuffle(choices)
                accepted: Coord | None = None
                for candidate in choices:
                    if candidate != path[-1] and candidate not in _neighbours(
                        path[-1], state.width, state.height, state.wrapped
                    ):
                        continue
                    if candidate in occupied or (candidate in used and candidate != path[-1]):
                        continue
                    future_ok = True
                    for future_index, future_point in fixed.items():
                        if future_index <= index:
                            continue
                        distance = _distance(
                            candidate,
                            future_point,
                            state.width,
                            state.height,
                            state.wrapped,
                        )
                        remaining = future_index - index
                        if distance > remaining or (distance - remaining) % 2:
                            future_ok = False
                            break
                    if future_ok:
                        accepted = candidate
                        break
                if accepted is None:
                    valid = False
                    break
                path.append(accepted)
                used.add(accepted)
            if valid and all(
                index < len(path) and path[index] == point
                for index, point in fixed.items()
                if index < slots
            ):
                return tuple(dict.fromkeys(path))
        return None

    @staticmethod
    def _matches_observation(world: WorldState, game_state: GameState) -> bool:
        if world.turn != game_state.turn or not world.snakes:
            return False
        width, height = game_state.board.width, game_state.board.height
        expected_own = _stable_body(game_state.you, width, height)
        own = world.snakes[0]
        observed_head = _valid(game_state.you.head, width, height)
        if (
            not own.alive
            or own.head != observed_head
            or own.body != expected_own
            or own.length != game_state.you.length
            or own.health != (game_state.you.health or 0)
        ):
            return False

        radius = game_state.game.ruleset.settings.viewRadius
        world_by_id = {snake.id: snake for snake in world.snakes}
        observed_opponents = {
            snake.id
            for snake in game_state.board.snakes
            if snake.id != game_state.you.id and snake.elimination_event is None
        }
        particle_opponents = {
            snake.id
            for snake in world.snakes[1:]
            if snake.alive
        }
        if particle_opponents != observed_opponents:
            return False
        for observed in game_state.board.snakes:
            if observed.id == game_state.you.id:
                continue
            particle = world_by_id.get(observed.id)
            if particle is None or not particle.alive:
                return False
            observed_enemy_head = _valid(observed.head, width, height)
            if observed_enemy_head is not None and particle.head != observed_enemy_head:
                return False
            if observed_enemy_head is None and particle.head is not None and _visible(
                particle.head, observed_head, radius
            ):
                return False
            observed_cells = set(_stable_body(observed, width, height))
            particle_cells = {
                point for point in particle.body if _visible(point, observed_head, radius)
            }
            if particle_cells != observed_cells:
                return False

        observed_food = {
            (food.x, food.y)
            for food in game_state.board.food
            if 0 <= food.x < width and 0 <= food.y < height
        }
        spawn_turns = dict(world.food_spawn_turns)
        projected_food = {
            point
            for point in world.food
            if _visible(point, observed_head, radius)
            or spawn_turns.get(point) == world.turn
        }
        observed_hazards = {
            (point.x, point.y)
            for point in game_state.board.hazards
            if 0 <= point.x < width and 0 <= point.y < height
        }
        return projected_food == observed_food and world.hazards == observed_hazards


__all__ = [
    "BeliefDiagnostics",
    "BeliefTrackerV2",
    "OpponentMemory",
]
