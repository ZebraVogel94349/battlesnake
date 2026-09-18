import copy
import unittest

import numpy as np

from snake25_bc import (
    Snake25ProxyEncoder,
    alive_snake_count,
    inferred_model_action,
    validation_game,
)


def snake(identifier, name, body, *, eliminated=False):
    return {
        "id": identifier,
        "name": name,
        "length": len(body),
        "health": 90 if name == "24" else None,
        "head": body[0] if body and body[0]["x"] >= 0 else None,
        "body": body,
        "elimination_event": ({"cause": "wall-collision"} if eliminated else None),
    }


def state_with_snake25():
    ours = snake(
        "ours",
        "24",
        [{"x": 7, "y": 7}, {"x": 7, "y": 6}, {"x": 7, "y": 5}],
    )
    opponent = snake(
        "real-25",
        "25",
        [{"x": 9, "y": 7}, {"x": 10, "y": 7}, {"x": 11, "y": 7}],
    )
    hidden = snake("hidden", "41", [{"x": -1, "y": -1}])
    eliminated = snake(
        "eliminated", "43", [{"x": -1, "y": -1}], eliminated=True
    )
    return {
        "turn": 20,
        "game": {
            "ruleset": {"settings": {"viewRadius": 5}},
        },
        "board": {
            "width": 15,
            "height": 15,
            "food": [{"x": 8, "y": 8, "spawn_turn": 1}],
            "snakes": [ours, opponent, hidden, eliminated],
        },
        "you": ours,
    }


class Snake25BehavioralCloningTests(unittest.TestCase):
    def test_infers_model_action_from_visible_head_delta(self):
        current = state_with_snake25()
        following = copy.deepcopy(current)
        snake25 = next(
            snake for snake in following["board"]["snakes"] if snake["name"] == "25"
        )
        snake25["head"] = {"x": 9, "y": 8}
        snake25["body"] = [
            {"x": 9, "y": 8},
            {"x": 9, "y": 7},
            {"x": 10, "y": 7},
        ]
        self.assertEqual(0, inferred_model_action(current, following))
        self.assertEqual(3, alive_snake_count(current))

    def test_proxy_is_snake25_centered_and_keeps_unseen_cells_fogged(self):
        with Snake25ProxyEncoder() as encoder:
            result = encoder.encode(state_with_snake25())
        self.assertIsNotNone(result)
        obs = result.observation
        self.assertEqual((9, 29, 29), obs.shape)
        self.assertTrue(np.isfinite(obs).all())
        self.assertEqual(1.0, obs[3, 14, 14])
        # World (14, 7) is five cells from Snake 25 but seven from the
        # recorder.  It must remain fogged in the proxy observation.
        self.assertEqual(0.0, obs[8, 19, 14])
        self.assertGreater(result.coverage, 0.0)
        self.assertLess(result.coverage, 1.0)

    def test_validation_split_is_stable_and_has_both_sides(self):
        decisions = [validation_game(f"game-{index}", 0.2) for index in range(100)]
        self.assertEqual(decisions, [
            validation_game(f"game-{index}", 0.2) for index in range(100)
        ])
        self.assertTrue(any(decisions))
        self.assertTrue(any(not decision for decision in decisions))


if __name__ == "__main__":
    unittest.main()
