from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from battlesnake_types import GameState
from mcts_simulator import (
    ACTIONS,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    SnakeState,
    WorldState,
    determinize_world,
    encode_world_observation,
    evaluate_world,
    heuristic_action_priors,
    is_terminal,
    legal_actions,
    step_world,
)


def snake(
    snake_id: str,
    body: tuple[tuple[int, int], ...],
    *,
    health: int = 100,
    length: int | None = None,
) -> SnakeState:
    return SnakeState(
        id=snake_id,
        body=body,
        head=body[0] if body else None,
        health=health,
        length=len(body) if length is None else length,
    )


def world(*snakes: SnakeState, **kwargs) -> WorldState:
    return WorldState(
        width=kwargs.pop("width", 7),
        height=kwargs.pop("height", 7),
        turn=kwargs.pop("turn", 0),
        snakes=tuple(snakes),
        root_alive_count=sum(item.alive for item in snakes),
        **kwargs,
    )


def test_action_order_is_the_ppo_order() -> None:
    assert ACTIONS == (UP, DOWN, LEFT, RIGHT) == (0, 1, 2, 3)


def test_delayed_growth_matches_hisss() -> None:
    state = world(
        snake("you", ((2, 2), (2, 1), (1, 1)), length=3),
        snake("enemy", ((6, 6),), length=3),
        food=frozenset({(2, 3)}),
    )

    just_ate = step_world(state, (UP, DOWN))
    assert just_ate.snakes[0].length == 4
    assert len(just_ate.snakes[0].body) == 3
    assert just_ate.snakes[0].health == 100

    physically_grown = step_world(just_ate, (RIGHT, DOWN))
    assert physically_grown.snakes[0].length == 4
    assert len(physically_grown.snakes[0].body) == 4


def test_food_rescues_snake_from_hazard_damage() -> None:
    state = world(
        snake("you", ((1, 1),), health=1, length=3),
        snake("enemy", ((6, 6),), length=3),
        food=frozenset({(1, 2)}),
        hazards=frozenset({(1, 2)}),
        hazard_damage=14,
    )
    assert legal_actions(state, 0) == (UP,)
    result = step_world(state, (UP, DOWN))
    assert result.snakes[0].alive
    assert result.snakes[0].health == 100


def test_living_snake_with_no_nonfatal_action_is_forced_lower_placement() -> None:
    state = world(
        snake("you", ((0, 0),), health=1, length=3),
        snake("enemy", ((6, 6),), length=3),
        snake("enemy-2", ((5, 5),), length=3),
    )

    assert legal_actions(state, 0) == ()
    assert not is_terminal(state, 0)
    assert evaluate_world(state, 0) == -1.0


def test_two_player_leaf_cannot_be_worse_than_second_place() -> None:
    trapped = world(
        snake("you", ((0, 0),), health=1, length=3),
        snake("enemy", ((6, 6),), length=3),
    )
    assert legal_actions(trapped, 0) == ()
    assert evaluate_world(trapped, 0) == 0.0

    # The same placement floor also applies to an ordinary non-terminal leaf
    # whose heuristic terms happen to be strongly negative.
    disadvantaged = world(
        snake("you", ((0, 0),), health=2, length=1),
        snake(
            "enemy",
            ((6, 6), (6, 5), (6, 4), (5, 4), (4, 4), (3, 4)),
            length=6,
        ),
    )
    assert disadvantaged.alive_count == 2
    assert legal_actions(disadvantaged, 0)
    assert evaluate_world(disadvantaged, 0) >= 0.0


def test_sole_survivor_has_exact_first_place_value() -> None:
    state = world(snake("you", ((3, 3),), length=3))
    assert state.alive_count == 1
    assert evaluate_world(state, 0) == 1.0


def test_root_alive_count_does_not_shift_remaining_players_out_of_top_slots() -> None:
    # A mid-game Battlesnake request omits the two snakes eliminated before
    # this search root.  The two remaining contestants still compete for the
    # evaluator's first and second slots, rather than third and fourth.
    initial = world(
        snake("you", ((1, 1),), length=3),
        snake("enemy", ((5, 5),), length=3),
    )
    eliminated_you = replace(
        initial.snakes[0],
        alive=False,
        elimination="test",
        death_turn=12,
    )
    state = replace(initial, turn=13, snakes=(eliminated_you, initial.snakes[1]))

    assert state.root_alive_count == 2
    assert evaluate_world(state, 0) == 0.0


@pytest.mark.parametrize(
    ("death_turns", "expected"),
    [
        # One survivor ranks above us and two earlier deaths rank below us:
        # second place earns one point, hence value zero.
        ((10, None, 9, 8), 0.0),
        # One survivor plus a simultaneous two-way tie occupies second/third.
        ((10, None, 10, 8), -0.5),
        # One survivor plus a simultaneous three-way tie occupies slots 2--4.
        ((10, None, 10, 10), -2.0 / 3.0),
        # A four-way simultaneous elimination averages every placement award.
        ((10, 10, 10, 10), -0.25),
        # Two earlier deaths followed by a simultaneous tie for first.
        ((10, 10, 8, 7), 0.5),
    ],
)
def test_terminal_value_matches_tied_placement_awards(
    death_turns: tuple[int | None, ...], expected: float
) -> None:
    initial = world(
        snake("you", ((0, 0),), length=3),
        snake("enemy-1", ((2, 2),), length=3),
        snake("enemy-2", ((4, 4),), length=3),
        snake("enemy-3", ((6, 6),), length=3),
    )
    ranked = tuple(
        current
        if death_turn is None
        else replace(
            current,
            alive=False,
            elimination="test",
            death_turn=death_turn,
        )
        for current, death_turn in zip(initial.snakes, death_turns, strict=True)
    )
    state = replace(initial, turn=11, snakes=ranked)

    assert state.root_alive_count == 4
    assert evaluate_world(state, 0) == pytest.approx(expected)


def test_complete_vacating_tail_is_legal_but_pending_tail_is_not() -> None:
    body = ((1, 1), (1, 2), (2, 2), (2, 1))
    state = world(
        snake("you", body, length=4),
        snake("enemy", ((6, 6),), length=3),
    )
    assert RIGHT in legal_actions(state, 0)
    assert step_world(state, (RIGHT, DOWN)).snakes[0].alive

    pending = world(
        snake("you", body, length=5),
        snake("enemy", ((6, 6),), length=3),
    )
    assert RIGHT not in legal_actions(pending, 0)


def test_head_to_head_uses_post_food_logical_length() -> None:
    state = world(
        snake("you", ((1, 2), (1, 1), (0, 1)), length=3),
        snake("enemy", ((3, 2), (3, 1), (4, 1)), length=3),
        food=frozenset({(2, 2)}),
    )
    result = step_world(state, (RIGHT, LEFT))
    # Both eat the same food before the same-length head collision is resolved.
    assert not result.snakes[0].alive
    assert not result.snakes[1].alive
    assert result.snakes[0].elimination == "head-collision"
    assert result.snakes[1].elimination == "head-collision"


def test_head_swap_is_body_collision_not_head_to_head() -> None:
    state = world(
        snake("you", ((1, 1), (1, 0), (0, 0)), length=3),
        snake("enemy", ((2, 1), (2, 0), (3, 0)), length=3),
    )
    result = step_world(state, (RIGHT, LEFT))
    assert [item.elimination for item in result.snakes] == [
        "snake-collision",
        "snake-collision",
    ]


def _blackout_request() -> GameState:
    return GameState.model_validate(
        {
            "turn": 17,
            "game": {
                "id": "game",
                "source": "test",
                "timeout": 500,
                "ruleset": {
                    "name": "blackout",
                    "version": "v1",
                    "settings": {
                        "foodSpawnChance": 15,
                        "hazardDamagePerTurn": 14,
                        "minimumFood": 1,
                        "viewRadius": 5,
                        "royale": {"shrinkEveryNTurns": 25},
                        "squad": {
                            "allowBodyCollisions": False,
                            "sharedElimination": False,
                            "sharedHealth": False,
                            "sharedLength": False,
                        },
                    },
                },
            },
            "board": {
                "height": 15,
                "width": 15,
                "food": [{"x": 8, "y": 7, "spawn_turn": 10}],
                "hazards": [],
                "snakes": [
                    {
                        "id": "you",
                        "name": "you",
                        "length": 3,
                        "latency": "0",
                        "squad": None,
                        "health": 80,
                        "head": {"x": 7, "y": 7},
                        "body": [
                            {"x": 7, "y": 7},
                            {"x": 7, "y": 6},
                            {"x": 7, "y": 5},
                        ],
                        "customizations": {
                            "color": [1, 2, 3],
                            "head": "default",
                            "tail": "default",
                        },
                    },
                    {
                        "id": "hidden",
                        "name": "hidden",
                        "length": 1,
                        "latency": "0",
                        "squad": None,
                        "health": 0,
                        "head": {"x": -1, "y": -1},
                        "body": [{"x": -1, "y": -1}],
                        "customizations": {
                            "color": [4, 5, 6],
                            "head": "default",
                            "tail": "default",
                        },
                    },
                ],
            },
            "you": {
                "id": "you",
                "name": "you",
                "length": 3,
                "latency": "0",
                "squad": None,
                "health": 80,
                "head": {"x": 7, "y": 7},
                "body": [
                    {"x": 7, "y": 7},
                    {"x": 7, "y": 6},
                    {"x": 7, "y": 5},
                ],
                "customizations": {
                    "color": [1, 2, 3],
                    "head": "default",
                    "tail": "default",
                },
            },
        }
    )


def test_hidden_enemy_stays_alive_and_does_not_create_terminal_win() -> None:
    state = WorldState.from_game_state(_blackout_request())
    assert state.snakes[0].id == "you"
    assert state.snakes[1].alive
    assert state.snakes[1].head is None
    assert not is_terminal(state)
    assert evaluate_world(state) < 1.0

    sampled = determinize_world(state, 1234)
    assert sampled.snakes[1].head is not None
    assert sampled.snakes[1].length >= 3
    assert (
        abs(sampled.snakes[1].head[0] - state.snakes[0].head[0])
        + abs(sampled.snakes[1].head[1] - state.snakes[0].head[1])
        > 5
    )


def test_hidden_enemy_head_never_overlaps_globally_announced_food() -> None:
    hidden = SnakeState(
        id="enemy",
        body=(),
        head=None,
        health=100,
        length=3,
        alive=True,
        body_complete=False,
        health_known=False,
        length_known=False,
    )
    state = world(
        snake("you", ((0, 0),), length=3),
        hidden,
        width=3,
        height=3,
        turn=9,
        view_radius=0,
        food=frozenset({(1, 0)}),
        food_spawn_turns=(((1, 0), 9),),
        minimum_food=1,
    )

    for seed in range(64):
        sampled = determinize_world(state, seed)
        assert sampled.snakes[1].head is not None
        assert sampled.snakes[1].head not in sampled.food


def test_hidden_enemy_stays_abstract_when_only_hidden_cell_is_food() -> None:
    hidden = SnakeState(
        id="enemy",
        body=(),
        head=None,
        health=100,
        length=3,
        alive=True,
        body_complete=False,
        health_known=False,
        length_known=False,
    )
    state = world(
        snake("you", ((0, 0),), length=3),
        hidden,
        width=2,
        height=1,
        turn=9,
        view_radius=0,
        food=frozenset({(1, 0)}),
        food_spawn_turns=(((1, 0), 9),),
        minimum_food=1,
    )

    sampled = determinize_world(state, 7)

    assert sampled.snakes[1].alive
    assert sampled.snakes[1].head is None
    assert sampled.snakes[1].body == ()
    assert legal_actions(sampled, 1) == ACTIONS
    assert not is_terminal(sampled, 0)
    assert sampled.food == state.food


def test_blackout_opponent_length_is_unknown_even_when_body_is_all_visible() -> None:
    request = _blackout_request().model_copy(deep=True)
    opponent = request.board.snakes[1]
    opponent.head.x, opponent.head.y = 7, 8
    opponent.body = [opponent.head]
    opponent.length = 1  # Hisss' restricted-body protocol value, not logical length.

    state = WorldState.from_game_state(request)

    assert state.snakes[1].body_complete
    assert not state.snakes[1].length_known
    # Unknown logical growth means the visible enemy cell cannot be assumed to
    # vacate merely because the restricted body list has length one.
    assert UP not in legal_actions(state, 0)

    sampled = determinize_world(state, 5)
    assert sampled.snakes[1].length >= 3
    assert not sampled.snakes[1].length_known


def test_search_food_spawn_restores_minimum_after_last_food_is_eaten() -> None:
    state = world(
        snake("you", ((1, 1),), length=3),
        snake("enemy", ((5, 5),), length=3),
        food=frozenset({(1, 2)}),
        minimum_food=1,
        food_spawn_chance=0,
    )

    result = step_world(state, (UP, DOWN), rng=np.random.default_rng(3), spawn_food=True)

    assert (1, 2) not in result.food
    assert len(result.food) == 1
    assert dict(result.food_spawn_turns)[next(iter(result.food))] == result.turn


def test_hidden_old_food_is_sampled_without_false_global_announcement() -> None:
    request = _blackout_request().model_copy(deep=True)
    request.board.food = []
    request.game.ruleset.settings.foodSpawnChance = 0

    root = WorldState.from_game_state(request)
    particle = determinize_world(root, 91)

    assert len(root.food) == 0
    assert len(particle.food) == 1
    hidden = next(iter(particle.food))
    assert abs(hidden[0] - particle.snakes[0].head[0]) + abs(
        hidden[1] - particle.snakes[0].head[1]
    ) > 5
    assert dict(particle.food_spawn_turns)[hidden] < particle.turn
    assert encode_world_observation(particle)[0].sum() == 0

    advanced = step_world(
        particle,
        (UP, DOWN),
        rng=np.random.default_rng(4),
        spawn_food=True,
    )
    assert len(advanced.food) == 1


def test_observation_layout_and_priors() -> None:
    state = WorldState.from_game_state(_blackout_request())
    obs = encode_world_observation(state)
    assert obs.shape == (9, 29, 29)
    assert obs.dtype == np.float32
    assert obs[0, 15, 14] == 1.0  # food at (8, 7), relative to head (7, 7)
    assert obs[3, 14, 14] == 1.0
    assert obs[4, 0, 0] == pytest.approx(0.8)
    assert obs[8].sum() == 61  # Manhattan ball with radius five

    priors = heuristic_action_priors(state)
    assert priors.shape == (4,)
    assert priors.dtype == np.float32
    assert priors.sum() == pytest.approx(1.0)
    assert np.all(priors >= 0)


def test_new_food_is_globally_visible_for_exactly_its_spawn_turn() -> None:
    state = world(
        snake("you", ((1, 1),), length=3),
        snake("enemy", ((13, 13),), length=3),
        width=15,
        height=15,
        turn=8,
        food=frozenset({(14, 14)}),
        food_spawn_turns=(((14, 14), 8),),
        view_radius=5,
    )
    assert encode_world_observation(state)[0].sum() == 1
    older = replace(state, turn=9)
    assert encode_world_observation(older)[0].sum() == 0


def test_full_information_steps_match_hisss() -> None:
    hisss = pytest.importorskip("hisss")
    from hisss.game.battlesnake import BattleSnakeGame
    from hisss.game.config import BattleSnakeConfig

    model_to_hisss = (hisss.UP, hisss.DOWN, hisss.LEFT, hisss.RIGHT)
    for seed in range(32):
        rng = random.Random(seed)
        bodies = {
            0: [[2, 2], [2, 1], [1, 1]],
            1: [[6, 6], [6, 7], [7, 7]],
        }
        lengths = [3 + rng.randrange(2), 3 + rng.randrange(2)]
        healths = [rng.randrange(1, 101), rng.randrange(1, 101)]
        actions = (rng.randrange(4), rng.randrange(4))
        targets = []
        for body, action in zip(bodies.values(), actions, strict=True):
            dx, dy = ((0, 1), (0, -1), (-1, 0), (1, 0))[action]
            targets.append((body[0][0] + dx, body[0][1] + dy))
        food = [list(targets[rng.randrange(2)])] if seed % 3 == 0 else []
        hazards = [list(targets[(seed + 1) % 2])] if seed % 4 == 0 else []

        cfg = BattleSnakeConfig(
            w=9,
            h=9,
            num_players=2,
            min_food=0,
            food_spawn_chance=0,
            init_snake_pos=bodies,
            init_food_pos=food,
            init_snake_len=lengths,
            init_snake_health=healths,
            init_hazards=hazards,
            royale=True,
            shrink_n_turns=100,
            hazard_damage=14,
            all_actions_legal=True,
        )
        env = BattleSnakeGame(cfg)
        try:
            simulated = world(
                snake("snake-0", tuple(map(tuple, bodies[0])), health=healths[0], length=lengths[0]),
                snake("snake-1", tuple(map(tuple, bodies[1])), health=healths[1], length=lengths[1]),
                width=9,
                height=9,
                food=frozenset(map(tuple, food)),
                hazards=frozenset(map(tuple, hazards)),
                hazard_damage=14,
                royale=True,
                shrink_every_n_turns=100,
            )
            simulated = step_world(simulated, actions)
            env.step(tuple(model_to_hisss[action] for action in actions))
            native = env.get_state()

            assert [item.alive for item in simulated.snakes] == native.snakes_alive
            assert [item.health for item in simulated.snakes] == native.snake_health
            assert [item.length for item in simulated.snakes] == native.snake_len
            assert [list(item.body) for item in simulated.snakes] == [
                native.snake_pos[0],
                native.snake_pos[1],
            ]
            assert simulated.food == frozenset(map(tuple, native.food_pos))
            for player, event in (native.elimination_events or {}).items():
                assert simulated.snakes[player].elimination == event.cause
        finally:
            env.close()
