from __future__ import annotations

import random

import hisss
import numpy as np

from battlesnake_types import GameState
from obs_config import make_game_config, to_model_obs
from ppo4 import (
    D4_ACTION_PERMUTATIONS,
    PPOAgent4,
    encode_observation,
    transform_observations,
)


def _game_state(
    *,
    head=(1, 1),
    body=((1, 1), (1, 0)),
    length=2,
    health=100,
    food=(),
    opponents=(),
    turn=10,
):
    def point(cell):
        if cell is None:
            return None
        return {"x": cell[0], "y": cell[1]}

    you = {
        "id": "you",
        "name": "you",
        "length": length,
        "latency": "0",
        "squad": None,
        "health": health,
        "head": point(head),
        "body": [point(cell) for cell in body],
        "customizations": {"color": [0, 0, 0], "head": None, "tail": None},
    }
    snakes = [you]
    for index, opponent in enumerate(opponents):
        opp_body = opponent.get("body", ())
        opp_head = opponent.get("head")
        snakes.append(
            {
                "id": f"opp-{index}",
                "name": f"opp-{index}",
                "length": opponent.get("length", len(opp_body)),
                "latency": "0",
                "squad": None,
                "health": None,
                "head": point(opp_head),
                "body": [point(cell) for cell in opp_body],
                "customizations": {
                    "color": [0, 0, 0],
                    "head": None,
                    "tail": None,
                },
            }
        )
    return GameState.model_validate(
        {
            "turn": turn,
            "game": {
                "id": "test-game",
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
                "food": [
                    {"x": cell[0], "y": cell[1], "spawn_turn": turn}
                    for cell in food
                ],
                "hazards": [],
                "snakes": snakes,
            },
            "you": you,
        }
    )


def test_direct_encoder_matches_native_blackout_observation():
    cfg = make_game_config()
    cfg.all_actions_legal = True
    rng = random.Random(731)
    checks = 0
    for _ in range(12):
        env = hisss.BattleSnakeGame(cfg)
        try:
            for _ in range(40):
                players = env.players_at_turn()
                native = env.get_obs()[0]
                for observation_index, player in enumerate(players):
                    state = GameState.model_validate_json(
                        hisss.to_battlesnake_json(env, player)
                    )
                    np.testing.assert_array_equal(
                        encode_observation(state),
                        to_model_obs(native[observation_index]),
                    )
                    checks += 1
                env.step(tuple(rng.randrange(4) for _ in players))
                if env.is_terminal():
                    break
        finally:
            env.close()
    assert checks >= 40


def test_encoder_keeps_globally_announced_food():
    state = _game_state(head=(1, 1), food=((14, 14),))
    obs = encode_observation(state)
    assert obs[0, 27, 27] == 1.0
    assert obs[8, 27, 27] == 0.0


def test_d4_transforms_and_action_permutations_agree():
    obs = np.zeros((9, 29, 29), dtype=np.float32)
    # An UP marker relative to the centered head.
    obs[0, 14, 15] = 1.0
    transformed = transform_observations(obs, 8)
    for symmetry in range(8):
        transformed_action = D4_ACTION_PERMUTATIONS[symmetry, 0]
        expected_delta = ((0, 1), (0, -1), (-1, 0), (1, 0))[
            transformed_action
        ]
        assert transformed[
            symmetry,
            0,
            14 + expected_delta[0],
            14 + expected_delta[1],
        ] == 1.0


def test_hard_mask_rejects_every_known_immediate_death():
    state = _game_state(
        head=(1, 1),
        body=((1, 1), (1, 0)),
        health=1,
        food=((2, 1),),
        opponents=({"head": None, "body": ((1, 2), None), "length": 2},),
    )
    agent = PPOAgent4.__new__(PPOAgent4)
    hard, soft = agent._action_tiers(state)
    # UP hits a visible headless enemy segment; DOWN hits our growing tail;
    # LEFT starves; only RIGHT reaches food.
    assert hard == [3]
    assert soft == [3]


def test_hidden_enemy_length_makes_head_to_head_conservative():
    state = _game_state(
        head=(7, 7),
        body=((7, 7), (7, 6), (7, 5), (7, 4)),
        length=4,
        opponents=(
            {
                "head": (9, 7),
                "body": ((9, 7), (10, 7), None),
                "length": 3,
            },
        ),
    )
    agent = PPOAgent4.__new__(PPOAgent4)
    hard, soft = agent._action_tiers(state)
    assert 3 in hard
    assert 3 not in soft


def test_visible_one_cell_blackout_enemy_length_is_still_uncertain():
    state = _game_state(
        head=(7, 7),
        body=((7, 7), (7, 6), (7, 5)),
        length=3,
        opponents=(
            {
                "head": (7, 9),
                "body": ((7, 9),),
                # Restricted Hisss protocol value while logical length is 3.
                "length": 1,
            },
        ),
    )
    agent = PPOAgent4.__new__(PPOAgent4)

    hard, soft = agent._action_tiers(state)

    assert 0 in hard
    assert 0 not in soft


def test_tail_is_free_only_after_pending_growth_finishes():
    body = ((1, 1), (2, 1), (2, 0), (1, 0))
    agent = PPOAgent4.__new__(PPOAgent4)

    normal = _game_state(head=(1, 1), body=body, length=4)
    growing = _game_state(head=(1, 1), body=body, length=5)
    normal_hard, _ = agent._action_tiers(normal)
    growing_hard, _ = agent._action_tiers(growing)

    assert 1 in normal_hard  # DOWN enters the tail that moves this turn.
    assert 1 not in growing_hard  # Pending growth keeps that tail occupied.


def test_search_detects_short_forced_self_trap():
    # The head at (1,1) can move UP into a one-cell cul-de-sac enclosed by its
    # own body, while LEFT remains open.
    state = _game_state(
        head=(1, 1),
        body=((1, 1), (0, 2), (1, 3), (2, 2), (2, 1), (2, 0), (1, 0)),
        length=7,
    )
    agent = PPOAgent4.__new__(PPOAgent4)
    agent.search_horizon = 6
    agent.search_node_budget = 2_000
    up_depth, up_complete = agent._survival_depth(state, 0)
    left_depth, left_complete = agent._survival_depth(state, 2)
    assert up_complete and left_complete
    assert up_depth < left_depth


def test_proven_soft_trap_falls_back_to_uncertain_hard_escape():
    # Regression from ...51594027 turn 420: LEFT is uncontested but a forced
    # self-trap; RIGHT is an uncertain H2H and the only long-lived route.
    state = _game_state(
        head=(5, 13),
        body=(
            (5, 13),
            (5, 14),
            (4, 14),
            (3, 14),
            (2, 14),
            (1, 14),
            (0, 14),
            (0, 13),
            (0, 12),
            (0, 11),
            (0, 10),
            (1, 10),
            (1, 11),
            (2, 11),
            (2, 12),
            (3, 12),
            (4, 12),
            (4, 11),
            (5, 11),
            (5, 12),
            (6, 12),
        ),
        length=21,
        opponents=(
            {
                "head": (7, 13),
                "body": ((7, 13), (7, 12), (7, 11), None),
                "length": 4,
            },
        ),
    )
    agent = PPOAgent4.__new__(PPOAgent4)
    agent.safety_search = True
    agent.search_horizon = 10
    agent.search_node_budget = 6_000

    hard, soft = agent._action_tiers(state)
    candidates, _ = agent._safe_candidates(state, hard, soft, policy_action=2)

    assert hard == [2, 3]
    assert soft == [2]
    assert candidates == [3]
