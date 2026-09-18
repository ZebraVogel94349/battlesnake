import json
import unittest

import hisss
from hisss.game.battlesnake import BattleSnakeGame, UP
from hisss.game.config import BattleSnakeConfig
from hisss.game.encoding import BestRestrictedEncodingConfig
from hisss.game.export import to_battlesnake_json


class TestBlackoutConfig(unittest.TestCase):
    def test_blackout_config_matches_rules(self):
        cfg = hisss.blackout_config()
        self.assertEqual(4, cfg.num_players)
        self.assertEqual(15, cfg.w)
        self.assertEqual(15, cfg.h)
        self.assertEqual(5, cfg.view_radius)
        self.assertTrue(cfg.ec.include_view_mask)
        self.assertTrue(cfg.ec.include_distance_map)

    def test_blackout_duel_config_matches_rules(self):
        cfg = hisss.blackout_duel_config()
        self.assertEqual(2, cfg.num_players)
        self.assertEqual(15, cfg.w)
        self.assertEqual(15, cfg.h)
        self.assertEqual(5, cfg.view_radius)


class TestBlackoutExport(unittest.TestCase):
    def _make_game(self):
        cfg = BattleSnakeConfig(
            w=11,
            h=11,
            num_players=2,
            min_food=0,
            food_spawn_chance=0,
            init_snake_pos={
                0: [[5, 5], [5, 4], [5, 3]],
                1: [[9, 9], [9, 8], [9, 7]],
            },
            init_food_pos=[[0, 0]],
            init_snake_len=[3, 3],
            all_actions_legal=True,
            ec=BestRestrictedEncodingConfig(),
            view_radius=3,
        )
        return BattleSnakeGame(cfg)

    def test_blackout_ruleset_export(self):
        game = self._make_game()
        data = json.loads(to_battlesnake_json(game, 0))
        self.assertEqual("blackout", data["game"]["ruleset"]["name"])
        self.assertEqual("blackout", data["game"]["map"])
        self.assertEqual(3, data["game"]["ruleset"]["settings"]["viewRadius"])
        game.close()

    def test_spawn_turn_food_visible_then_hidden(self):
        game = self._make_game()
        data = json.loads(to_battlesnake_json(game, 0))
        self.assertIn({"x": 0, "y": 0, "spawn_turn": 0}, data["board"]["food"])

        game.step((UP, UP))
        data = json.loads(to_battlesnake_json(game, 0))
        self.assertEqual([], data["board"]["food"])
        game.close()


if __name__ == "__main__":
    unittest.main()
