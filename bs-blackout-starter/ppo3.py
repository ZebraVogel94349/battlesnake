from __future__ import annotations

import hashlib
import os
import sys
from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import hisss
from hisss.game.state import BattleSnakeState

from battlesnake_types import BaseAgent, GameState, MoveAction, Direction

from obs_config import (
    OBS_SHAPE,
    build_model,
    make_game_config,
    recurrent_logits,
    to_model_obs,
)


DEFAULT_MODEL_PATH = Path(__file__).with_name(
    "ppo_bs_lstm_cuda_v25_targeted_finish_champion.zip"
)


def _file_fingerprint(path: Path | None) -> str:
    """Return a complete model/source SHA, or an honest unavailable marker."""

    if path is None or not path.is_file():
        return "unavailable"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(paths: Sequence[Path]) -> str:
    """Fingerprint all source components that determine PPO3 inference."""

    digest = hashlib.sha256()
    resolved = sorted(
        (item.resolve() for item in paths),
        key=lambda item: item.name,
    )
    for path in resolved:
        component = _file_fingerprint(path)
        if component == "unavailable":
            return "unavailable"
        label = path.name.encode("utf-8", errors="surrogatepass")
        digest.update(len(label).to_bytes(4, byteorder="big", signed=False))
        digest.update(label)
        digest.update(bytes.fromhex(component))
    return digest.hexdigest()


class PPOAgent3(BaseAgent):
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        space_mask=None,
    ):
        # RecurrentPPO skeleton from the shared architecture; weights via
        # set_parameters (never .load — unpickling the full zip segfaults).
        # model_path=None keeps random init (only useful for obs/plumbing tests).
        self.device = "cpu"
        self.model_path = None if model_path is None else Path(model_path).resolve()
        self.model = build_model(device=self.device)
        if self.model_path is not None:
            self.model.set_parameters(str(self.model_path), device=self.device)
        self.model.policy.set_training_mode(False)
        # Inference-time flood-fill "space" mask (Workstream A): prefer moves whose
        # reachable pocket is at least our own length, on top of the hard/soft
        # geometry masks. Endgame self-trapping is the dominant failure mode; this
        # steers the argmax away from committing into a dead pocket a few turns
        # early. space_mask=None -> read env var PPO_SPACE_MASK (default on) so
        # eval wrappers can A/B it without new subclasses.
        if space_mask is None:
            space_mask = os.environ.get("PPO_SPACE_MASK", "1") not in ("0", "", "false", "False")
        self.space_mask = bool(space_mask)
        self.env = None          # lazy creation
        # Deployment ZIPs contain the nine-channel actor only. Privileged critic
        # channels are training-only and are intentionally not exported.
        self.obs_shape = OBS_SHAPE
        self.action_map = [hisss.UP, hisss.DOWN, hisss.LEFT, hisss.RIGHT]
        self.direction_map = {
            hisss.UP: Direction.UP,
            hisss.DOWN: Direction.DOWN,
            hisss.LEFT: Direction.LEFT,
            hisss.RIGHT: Direction.RIGHT,
        }
        source_dir = Path(__file__).resolve().parent
        self._model_fingerprint = _file_fingerprint(self.model_path)
        self._code_fingerprint = _source_fingerprint(
            (source_dir / "ppo3.py", source_dir / "obs_config.py")
        )
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

    def get_diagnostics(self) -> dict[str, object]:
        """Return the complete effective identity used by schema-v2 reports."""

        return {
            "agent": type(self).__name__,
            "model_sha256": self._model_fingerprint,
            "code_sha256": self._code_fingerprint,
            "space_mask": self.space_mask,
            "device": self.device,
            "observation_shape": list(self.obs_shape),
        }

    def start(self, game_state: GameState):
        """start is called when the battlesnake begins a game"""
        if self.env is not None:
            self.env.close()
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

    def _occupied_cells(self, game_state: GameState) -> set[tuple[int, int]]:
        """In-bounds body cells that block movement: own body + visible enemy
        bodies, each excluding the tail (it vacates next turn). Fog-hidden cells
        are simply absent -> treated as free everywhere (own body is never fogged,
        so this only ever under-counts unseen enemy segments, which is the
        deliberate optimistic choice — hallucinated walls caused wall deaths)."""
        w, h = game_state.board.width, game_state.board.height

        def in_bounds(x, y):
            return 0 <= x < w and 0 <= y < h

        occ: set[tuple[int, int]] = set()
        for s in game_state.board.snakes:
            body = list(s.body)
            for p in body[:-1]:
                if p is not None and in_bounds(p.x, p.y):
                    occ.add((p.x, p.y))
        return occ

    def _flood_free(self, start, occupied, w, h, cap) -> int:
        """Count free cells reachable from `start` (4-connected), stopping once
        `cap` is reached (we only care whether the pocket is >= our length)."""
        if not (0 <= start[0] < w and 0 <= start[1] < h) or start in occupied:
            return 0
        seen = {start}
        queue = deque([start])
        while queue:
            if len(seen) >= cap:
                break
            x, y = queue.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (0 <= nx < w and 0 <= ny < h
                        and (nx, ny) not in occupied and (nx, ny) not in seen):
                    seen.add((nx, ny))
                    queue.append((nx, ny))
        return len(seen)

    def _spacious_moves(self, game_state: GameState, moves: list[int]) -> list[int]:
        """Subset of `moves` whose target cell opens a pocket >= our own length
        (flood-fill on the 15x15 board; enemy heads may extend, so this is a
        heuristic, but it reliably rejects the small dead pockets that kill us)."""
        head = game_state.you.head
        if head is None or not moves:
            return list(moves)
        w, h = game_state.board.width, game_state.board.height
        our_len = game_state.you.length if game_state.you.length is not None else len(game_state.you.body)
        occupied = self._occupied_cells(game_state)
        out = []
        for m in moves:
            d = self.direction_map[self.action_map[m]]
            target = (head.x + d.dx, head.y + d.dy)
            if self._flood_free(target, occupied, w, h, our_len) >= our_len:
                out.append(m)
        return out

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
        blocked = self._occupied_cells(game_state)
        # Cells a stronger/equal visible enemy head could swing into (lose head-to-head).
        contested: set[tuple[int, int]] = set()
        for s in game_state.board.snakes:
            if s.id == game_state.you.id:
                continue
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

    def _select_legal(self, game_state: GameState,
                      hard_legal: list[int], soft_legal: list[int]) -> list[int]:
        """Choose the action tier the policy is masked to. Prefer moves that avoid
        losing head-to-heads (soft); fall back to merely non-suicidal (hard) rather
        than stranding the agent. With the space mask on, prefer moves that keep a
        pocket >= our length within the current tier — order (soft∩spacious) →
        (hard∩spacious) → soft → hard (an empty spacious set falls through)."""
        legal_model = soft_legal if soft_legal else hard_legal
        if self.space_mask:
            for tier in (soft_legal, hard_legal):
                spacious = self._spacious_moves(game_state, tier)
                if spacious:
                    return spacious
        return legal_model

    def move(self, game_state: GameState) -> MoveAction:
        obs = self.extract_state(game_state)
        hard_legal, soft_legal = self._compute_legal(game_state)
        legal_model = self._select_legal(game_state, hard_legal, soft_legal)
        action_int = self._masked_action(obs, legal_model)
        hisss_action = self.action_map[action_int]
        return MoveAction(move=self.direction_map[hisss_action])

    def end(self, game_state: GameState):
        """end is called when the battlesnake finishes a game"""
        self._reset_memory()
        if self.env is not None:
            self.env.close()
            self.env = None

if __name__ == "__main__":
    from battlesnake_server import start_server

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <port>")
        sys.exit(1)

    agent = PPOAgent3()
    port = int(sys.argv[1])

    start_server(agent=agent, port=port)
