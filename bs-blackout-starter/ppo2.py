import sys
import numpy as np
import torch
import hisss
from hisss.game.state import BattleSnakeState

from battlesnake_types import BaseAgent, GameState, MoveAction, Direction

from obs_config import OBS_SHAPE, to_model_obs, make_game_config, build_model, recurrent_logits


# this is the eventual model that will load the weights from the zip and take part in the challenge

class PPOAgent2(BaseAgent):
    def __init__(self, model_path="ppo_bs_lstm_cuda_780m_v4.zip"):
        # RecurrentPPO skeleton from the shared architecture; weights via
        # set_parameters (never .load — unpickling the full zip segfaults).
        # model_path=None keeps random init (only useful for obs/plumbing tests).
        self.model = build_model(device="cpu")
        if model_path is not None:
            self.model.set_parameters(model_path, device="cpu")
        self.model.policy.set_training_mode(False)
        self.env = None          # lazy creation
        self.obs_shape = OBS_SHAPE
        self.action_map = [hisss.UP, hisss.DOWN, hisss.LEFT, hisss.RIGHT]
        self.direction_map = {
            hisss.UP: Direction.UP,
            hisss.DOWN: Direction.DOWN,
            hisss.LEFT: Direction.LEFT,
            hisss.RIGHT: Direction.RIGHT,
        }
        self._reset_memory()

    def _reset_memory(self):
        """Forget the previous game: the LSTM state must never leak across games."""
        self._lstm_states = None
        self._episode_start = True

    def get_name(self):
        return "Der Snaketürke"

    def get_color(self):
        return "#94b1ff"

    def get_author(self):
        return "Julius"

    def start(self, game_state: GameState):
        """start is called when the battlesnake begins a game"""
        self.env = hisss.BattleSnakeGame(make_game_config())
        self._reset_memory()

    def _ensure_env(self, width: int, height: int):
        if width != 15 or height != 15:
            raise ValueError(f"Board missmatch {width}x{height}.")

        if self.env is None:
            cfg = make_game_config()
            cfg.w = 15
            cfg.h = 15
            self.env = hisss.BattleSnakeGame(cfg)
            self.env.reset()

    def extract_state(self, game_state: GameState) -> np.ndarray:
        # 1. Enforce board dimensions
        self._ensure_env(game_state.board.width, game_state.board.height)

        # 2. Identify agent's snake
        you_id = game_state.you.id
        all_snakes = list(game_state.board.snakes)
        agent_snake = None
        opponent_snakes = []
        for s in all_snakes:
            if s.id == you_id:
                agent_snake = s
            else:
                opponent_snakes.append(s)

        if agent_snake is None:
            raise ValueError("Agent snake not found in game_state")

        opponent_snakes.sort(key=lambda s: s.id)
        snake_list = [agent_snake] + opponent_snakes[:3]

        # Placeholder cell for fog-hidden/dead enemy slots: the board corner
        # farthest from our head is always outside the view radius (>= 14 > 5
        # Manhattan on 15x15), so the view mask hides it. Marking such slots
        # alive=False instead can make the reconstructed env terminal (only 1
        # snake "alive") and extract_state would return a BLANK obs — which is
        # exactly what happened whenever all three enemies were fogged.
        you_head = agent_snake.head
        phantom = (0 if (you_head.x if you_head else 7) >= 8 else 14,
                   0 if (you_head.y if you_head else 7) >= 8 else 14)

        snakes_alive = []
        snake_pos = {}
        snake_health = []
        snake_len = []

        for i in range(4):
            s = snake_list[i] if i < len(snake_list) else None
            # Keep hidden segments ((-1,-1) placeholders) as phantom cells
            # instead of dropping them: the encoder's body-decay value is
            # 0.1*(length - index), so segment INDICES must survive. Dropping
            # a fogged head would promote the first visible segment to a false
            # head and shift every decay value by one.
            body_positions = []
            any_visible = False
            for p in ([] if s is None else s.body):
                if p is not None and 0 <= p.x < 15 and 0 <= p.y < 15:
                    body_positions.append((p.x, p.y))
                    any_visible = True
                else:
                    body_positions.append(phantom)

            if any_visible:
                # Snake is (at least partly) visible and alive — the protocol
                # removes eliminated snakes from board.snakes, so anything
                # listed with visible cells is alive. NOTE: enemy health is
                # fogged to 0 for ALL enemies (private info); never feed 0
                # into set_state (the health channel is enemy-dropped anyway,
                # any positive works).
                snake_pos[i] = body_positions
                snakes_alive.append(True)
                snake_health.append(s.health if s.health else 100)
                snake_len.append(s.length if s.length is not None else len(body_positions))
            else:
                # Fog-hidden, eliminated, or missing slot: harmless phantom
                # outside our view (see above). The enemy channels are
                # compressed, so an "alive" count leak is impossible.
                snake_pos[i] = [phantom]
                snakes_alive.append(True)
                snake_health.append(100)
                snake_len.append(1)

        food_pos = [(f.x, f.y) for f in game_state.board.food if f is not None]

        hisss_state = BattleSnakeState(
            turn=game_state.turn,
            snakes_alive=snakes_alive,
            snake_pos=snake_pos,
            food_pos=food_pos,
            snake_health=snake_health,
            snake_len=snake_len,
        )

        self.env.set_state(hisss_state)

        if self.env.is_terminal():
            return np.zeros(self.obs_shape, dtype=np.float32)

        # Our snake is always placed at index 0. If it is alive but has no legal
        # move (fully trapped, every direction is certain death) hisss drops it
        # from players_at_turn. It is doomed this turn — return a blank obs and let
        # move() emit any action, rather than crashing on .index(0).
        if 0 not in self.env.players_at_turn():
            return np.zeros(self.obs_shape, dtype=np.float32)

        batch_obs = self.env.get_obs()[0]
        agent_idx = self.env.players_at_turn().index(0)

        return to_model_obs(batch_obs[agent_idx])

    def _masked_action(self, obs: np.ndarray, legal_model_indices: list[int]) -> int:
        """Pick the highest-probability action among the legal (non-suicidal) ones.

        This is inference-time action masking: we take the policy's preference
        ordering and choose its best *legal* action. Falls back to the unmasked
        argmax when no legal action is known (agent already doomed).

        The forward pass also advances the LSTM hidden state (one call per
        turn). Overriding the sampled/argmax action with the mask does NOT
        corrupt the state — the LSTM consumes only observations, not actions."""
        logits_t, self._lstm_states = recurrent_logits(
            self.model.policy, obs, self._lstm_states, self._episode_start)
        self._episode_start = False
        logits = logits_t.cpu().numpy()
        if not legal_model_indices:
            return int(np.argmax(logits))
        masked = np.full(logits.shape, -np.inf, dtype=np.float32)
        for m in legal_model_indices:
            masked[m] = logits[m]
        return int(np.argmax(masked))

    def _compute_legal(self, game_state: GameState) -> tuple[list[int], list[int]]:
        """Return (hard_legal, soft_legal) model action indices, from raw geometry.

        hard_legal: not a certain-death move — excludes walls, own body, and any
            *visible* enemy body cell. All known exactly, even under fog. No hisss
            env reconstruction (that hallucinated obstacles and caused wall deaths).
        soft_legal: hard_legal minus cells we could LOSE a head-to-head on — i.e.
            cells adjacent to a *visible* enemy head whose length >= ours. We still
            keep head-to-heads we'd win (we're longer), preserving the body-check
            strategy that crushes the short random agents.

        Phase A = walls + own body; Phase B = enemy bodies + losing head-to-heads.
        """
        head = game_state.you.head
        w, h = game_state.board.width, game_state.board.height
        if head is None:
            allm = list(range(len(self.action_map)))
            return allm, allm  # shouldn't happen for our live snake

        our_len = game_state.you.length if game_state.you.length is not None else len(game_state.you.body)

        def in_bounds(x, y):
            return 0 <= x < w and 0 <= y < h

        # Certain-death cells: own body + visible enemy bodies (all but each tail).
        blocked: set[tuple[int, int]] = {
            (p.x, p.y) for p in game_state.you.body[:-1]
            if p is not None and in_bounds(p.x, p.y)
        }
        # Cells a stronger/equal visible enemy head could swing into (lose head-to-head).
        contested: set[tuple[int, int]] = set()
        for s in game_state.board.snakes:
            if s.id == game_state.you.id:
                continue
            for p in s.body[:-1]:
                if p is not None and in_bounds(p.x, p.y):
                    blocked.add((p.x, p.y))
            if s.head is not None:
                s_len = s.length if s.length is not None else len(s.body)
                # Under fog an enemy body is truncated at the view boundary, so
                # the reported length is a LOWER bound. If the visible tail sits
                # at/behind the boundary the body may continue out of sight ->
                # true length unknown, assume we would lose the head-to-head.
                # (Short fully-visible snakes are unaffected, so the winning
                # head-to-heads vs randoms are kept.)
                vr = game_state.game.ruleset.settings.viewRadius
                tail = s.body[-1] if s.body else None
                length_unknown = tail is None or (
                    vr is not None
                    and abs(tail.x - head.x) + abs(tail.y - head.y) >= vr
                )
                if length_unknown or s_len >= our_len:
                    for d in self.direction_map.values():
                        cx, cy = s.head.x + d.dx, s.head.y + d.dy
                        if in_bounds(cx, cy):
                            contested.add((cx, cy))

        hard_legal, soft_legal = [], []
        for m, hisss_action in enumerate(self.action_map):
            d = self.direction_map[hisss_action]
            tx, ty = head.x + d.dx, head.y + d.dy
            if not in_bounds(tx, ty) or (tx, ty) in blocked:
                continue
            hard_legal.append(m)
            if (tx, ty) not in contested:
                soft_legal.append(m)
        return hard_legal, soft_legal

    def move(self, game_state: GameState) -> MoveAction:
        obs = self.extract_state(game_state)

        # Prefer moves that avoid losing head-to-heads (soft); if none exist, fall
        # back to merely non-suicidal moves (hard) rather than stranding the agent.
        hard_legal, soft_legal = self._compute_legal(game_state)
        legal_model = soft_legal if soft_legal else hard_legal

        action_int = self._masked_action(obs, legal_model)
        hisss_action = self.action_map[action_int]
        return MoveAction(move=self.direction_map[hisss_action])

    def end(self, game_state: GameState):
        """end is called when the battlesnake finishes a game"""
        self._reset_memory()

if __name__ == "__main__":
    from battlesnake_server import start_server

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <port>")
        sys.exit(1)

    agent = PPOAgent()
    port = int(sys.argv[1])

    start_server(agent=agent, port=port)
