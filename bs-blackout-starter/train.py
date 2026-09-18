import gymnasium as gym
import numpy as np

# simulator
import hisss

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# scaling
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import torch as th

import glob
import math
import os
import random
import uuid
from battlesnake_types import GameState, Direction

# NOTE: sb3-contrib has no MaskableRecurrentPPO, so the LSTM run drops
# training-time action masking. Illegal moves are handled by the penalty in
# step(); the inference-time mask in ppo.py is unaffected.
from obs_config import OBS_SHAPE, to_model_obs, make_game_config, build_model

# the enemies
from random_agent import RandomAgent
from hungry_agent import HungryAgent
from selfplay import FrozenOpponent, sample_pool_path, ensure_pool_seeded, POOL_DIR

# Phase-B opponent mix (self-play pool has snapshots). AT MOST ONE Hungry per
# game: two deterministic Hungrys converge on the same apple and head-crash
# each other early — short episodes, cheap placement wins, weak signal.
P_HUNGRY_SLOT = 0.45   # slot 0: Hungry anchor, else pool sample
P_RANDOM_SLOT = 0.15   # slot 1: Random anchor, else pool sample

# From scratch, self-play cannot start immediately (a random-init pool gives no
# learning signal). The snapshot callback stays closed until the eval suite
# measures this win rate vs 3x Hungry; until then the pool is empty and every
# env plays the phase-A heuristic mix.
SELFPLAY_GATE_WIN_RATE = 0.55

# Penalty for picking an illegal action (a legal one is substituted). Under
# MaskablePPO this was a dead safety net at -0.05; with RecurrentPPO it is the
# only training-time signal against illegal moves, hence stronger.
# Note by Julius: This can result in the agent commiting sucide when thats cheaper than risking illegal moves 
ILLEGAL_MOVE_PENALTY = -0.1


DIRECTION_TO_HISSS = {
    Direction.UP: hisss.UP,
    Direction.DOWN: hisss.DOWN,
    Direction.LEFT: hisss.LEFT,
    Direction.RIGHT: hisss.RIGHT
}


class BattlesnakeHisssEnv(gym.Env):
    """
    A Gymnasium wrapper for the Hisss Battlesnake simulator.
    Fog of war activated. Opponents: heuristics first, frozen self-play
    checkpoints once the pool has snapshots (see _new_episode_opponents).
    """
    def __init__(self):
        super().__init__()

        # 32 workers must not fight over torch threads; the frozen-opponent
        # forwards are tiny, single-threaded is fastest in aggregate.
        th.set_num_threads(1)

        self.game_cfg = make_game_config()
        self.env = hisss.BattleSnakeGame(self.game_cfg)

        # HungryAgent keeps per-game food memory keyed by game.id, so every
        # state we hand an opponent must carry the same id for the whole episode.
        self._game_id = f"train-{uuid.uuid4()}"

        self.opps: list = []

        self.is_closed = False

        # 4 possible directions in Battlesnake
        self.action_space = gym.spaces.Discrete(4)
        self._actions = [hisss.UP, hisss.DOWN, hisss.LEFT, hisss.RIGHT]

        # Trimmed channel set in CHW format (obs_config.to_model_obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=OBS_SHAPE,  # (9, 29, 29) CHW
            dtype=np.float32
        )
        self.env.reset()

    def _new_episode_opponents(self) -> None:
        """Sample a fresh opponent lineup for this episode.

        Phase A (pool empty, pre-gate): 1x Hungry + 2x Random, shuffled.
        Phase B (pool has snapshots): [Hungry|pool, Random|pool, pool],
        shuffled — never more than one Hungry per game (see P_HUNGRY_SLOT).
        Frozen opponents act on raw hisss obs and need no start()/game.id
        plumbing; their LSTM state resets with the fresh instance.
        """
        def pool_or_random():
            path = sample_pool_path()
            return FrozenOpponent(path) if path else RandomAgent()

        if sample_pool_path() is None:
            lineup = [HungryAgent(), RandomAgent(), RandomAgent()]
        else:
            lineup = [
                HungryAgent() if random.random() < P_HUNGRY_SLOT else pool_or_random(),
                RandomAgent() if random.random() < P_RANDOM_SLOT else pool_or_random(),
                pool_or_random(),
            ]
        random.shuffle(lineup)
        self.opps = lineup

        for idx, opp in enumerate(self.opps, start=1):
            if isinstance(opp, FrozenOpponent):
                continue
            try:
                st = GameState.model_validate_json(
                    hisss.to_battlesnake_json(self.env, idx, include_eliminated=False))
                st.game.id = self._game_id
                opp.start(st)
            except Exception as e:
                print(f"Warning: opponent {idx} start() failed: {e!r}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        self._new_episode_opponents()

        batch_obs = self.env.get_obs()[0]
        agent_idx = self.env.players_at_turn().index(0)
        obs = to_model_obs(batch_obs[agent_idx])

        return obs, {}

    def step(self, action):
            # If game already terminal, return zeros and done=True immediately
        if self.env.is_terminal():
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {}

        if not self.env.is_player_at_turn(0):
            terminal_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return terminal_obs, -1.0, True, False, {}

        # Agent action validation
        agent_valid = self.env.available_actions(0)
        agent_action = self._actions[action]
        assert self.action_space.contains(action)

        penalty = 0.0
        if agent_action not in agent_valid:
            # If invalid (e.g. wall / own body), pick a random valid one and penalize
            agent_action = random.choice(agent_valid)
            penalty = ILLEGAL_MOVE_PENALTY

        # Collect the opponents' moves
        moves: list[int] = []
        moves.append(agent_action)
        frozen_obs = None   # batch obs, fetched once per step iff a frozen opponent moves
        players = None
        for idx, cur_agent in enumerate(self.opps, start=1):
            if not self.env.is_player_at_turn(idx):
                continue
            legal = self.env.available_actions(idx)
            if isinstance(cur_agent, FrozenOpponent):
                # Frozen self-play opponent: acts on its raw (fogged) hisss obs,
                # exactly what it saw during its own training — no JSON round-trip.
                try:
                    if frozen_obs is None:
                        frozen_obs = self.env.get_obs()[0]
                        players = self.env.players_at_turn()
                    obs_o = to_model_obs(frozen_obs[players.index(idx)])
                    mask = np.array([a in legal for a in self._actions], dtype=bool)
                    if not mask.any():
                        mask[:] = True
                    move = self._actions[cur_agent.act(obs_o, mask)]
                except Exception as e:
                    print(f"Warning: frozen opponent {idx} failed: {e!r}")
                    move = random.choice(legal) if legal else hisss.UP
            else:
                # Scripted opponent (Hungry/Random): export env state to json and call agent
                cur_str = hisss.to_battlesnake_json(self.env, idx, include_eliminated=False)
                cur_state = GameState.model_validate_json(cur_str)
                cur_state.game.id = self._game_id
                try:
                    move = DIRECTION_TO_HISSS[cur_agent.move(cur_state).move]
                except Exception as e:
                    # Runtime error in the opponent: keep the episode going with a legal move.
                    print(f"Warning: opponent {idx} move() failed: {e!r}")
                    move = random.choice(legal) if legal else hisss.UP
            if move not in legal:
                move = random.choice(legal) if legal else hisss.UP

            moves.append(move)

        rewards, done, _ = self.env.step(actions=tuple(moves))
        reward = float(rewards[0]) + penalty

        if not done and not self.env.is_terminal() and self.env.is_player_at_turn(0):
            reward += 0.003  # survival reward per timestep. 0.005 was too dominant, 0.001 made the agent too passive; 0.003 is the middle ground.

        # After the step, check if game is over or agent is dead
        if done or self.env.is_terminal() or not self.env.is_player_at_turn(0):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, reward, True, False, {}

        # Otherwise, extract fresh observation for agent 0
        try:
            batch_obs = self.env.get_obs()[0]
            agent_idx = self.env.players_at_turn().index(0)
            obs = to_model_obs(batch_obs[agent_idx])
        except Exception as e:
            print(f"Warning: hisss internal error during get_obs: {e}. Terminating episode.")
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -1.0, True, False, {}

        # Return standard Gymnasium format: obs, reward, terminated, truncated, info
        assert self.observation_space.contains(obs), f"Expected shape {self.observation_space.shape}, got {obs.shape}"
        return obs, reward, done, False, {}


class SelfPlaySnapshotCallback(BaseCallback):
    """Freeze the current model into the self-play pool — but only once the
    gate is open (set by MaskedEvalCallback when win_rate_hungry reaches
    SELFPLAY_GATE_WIN_RATE; a from-scratch random policy would poison the
    pool). The first snapshot lands right after the gate opens, then every
    snapshot_freq steps. Written atomically (tmp + os.replace) so an env worker
    can never sample a half-written zip. Oldest snapshots beyond max_snapshots
    are pruned."""
    def __init__(self, snapshot_freq: int = 5_000_000, pool_dir: str = POOL_DIR,
                 max_snapshots: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.snapshot_freq = snapshot_freq
        self.pool_dir = pool_dir
        self.max_snapshots = max_snapshots
        self._next = 0
        # Resuming a run whose pool already has snapshots: gate stays open.
        self.gate_open = bool(glob.glob(os.path.join(pool_dir, "snap_*.zip")))

    def _on_step(self) -> bool:
        if not self.gate_open:
            return True
        if self.num_timesteps >= self._next:
            self._next = self.num_timesteps + self.snapshot_freq
            os.makedirs(self.pool_dir, exist_ok=True)
            tmp = os.path.join(self.pool_dir, "_snap_tmp.zip")
            dst = os.path.join(self.pool_dir, f"snap_{self.num_timesteps}.zip")
            self.model.save(tmp)
            os.replace(tmp, dst)
            snaps = sorted(glob.glob(os.path.join(self.pool_dir, "snap_*.zip")),
                           key=os.path.getmtime)
            while len(snaps) > self.max_snapshots:
                os.remove(snaps.pop(0))
            if self.verbose:
                print(f"[selfplay] pool snapshot saved: {dst}")
        return True


class MaskedEvalCallback(BaseCallback):
    """Every `eval_freq` total steps, play masked eval games (the real PPOAgent
    inference path — hard + soft head-to-head mask, LSTM state carried across
    turns) against each opponent suite and log win-rate / draw-rate / avg-turns
    / death-cause fractions to TensorBoard (suffixed per suite, e.g.
    eval/win_rate_hungry).

    This is the metric that actually matters (ep_rew_mean is a poor proxy). It
    also opens the self-play gate on the snapshot callback once win_rate_hungry
    reaches gate_win_rate. Runs in the main process, so training pauses briefly
    during each eval. Wrapped in try/except so an eval bug can never kill
    training.
    """
    def __init__(self, eval_freq: int = 2_500_000,
                 tmp_path: str = "./_eval_tmp.zip",
                 snapshot_cb: SelfPlaySnapshotCallback | None = None,
                 gate_win_rate: float = SELFPLAY_GATE_WIN_RATE,
                 suite_games: tuple[int, int, int] = (100, 50, 32),
                 verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.tmp_path = tmp_path
        self.snapshot_cb = snapshot_cb
        self.gate_win_rate = gate_win_rate
        self.suite_games = suite_games  # (hungry, random, self_snap0) game counts
        self._next_eval = eval_freq
        self._evals_done = 0

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_freq
            try:
                self._run_eval()
            except Exception as e:
                print(f"[eval] skipped due to error: {e!r}")
        return True

    def _run_eval(self) -> None:
        from ppo import PPOAgent

        self.model.save(self.tmp_path)
        ppo = PPOAgent(model_path=self.tmp_path)

        # (label, opponent factory, games, every_n_evals): hungry is the target
        # metric (3x Hungry kept for comparability with all previous runs —
        # mutual Hungry head-kills inflate it, read accordingly), random the
        # regression watch. Once the pool has snapshots, self_snap0 plays the
        # EARLIEST snapshot: win rate > 0.25 means better than where self-play
        # started (PPO opponents are expensive, hence fewer games, half cadence).
        suites = [
            ("hungry", HungryAgent, self.suite_games[0], 1),
            ("random", RandomAgent, self.suite_games[1], 1),
        ]
        snaps = sorted(glob.glob(os.path.join(POOL_DIR, "snap_*.zip")),
                       key=os.path.getmtime)
        if snaps:
            snap0 = snaps[0]
            suites.append(("self_snap0",
                           lambda: PPOAgent(model_path=snap0),
                           self.suite_games[2], 2))

        for label, factory, n_games, every_n in suites:
            if self._evals_done % every_n == 0:
                win_rate = self._run_suite(ppo, label, factory, n_games)
                if (label == "hungry" and self.snapshot_cb is not None
                        and not self.snapshot_cb.gate_open
                        and win_rate >= self.gate_win_rate):
                    print(f"[selfplay] gate OPEN at {self.num_timesteps} steps: "
                          f"win_rate_hungry={win_rate:.2f} >= {self.gate_win_rate}")
                    self.snapshot_cb.gate_open = True
        self._evals_done += 1

        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)

    def _run_suite(self, ppo, label: str, factory, n_games: int) -> float:
        from collections import Counter
        import hisss as _hisss
        from compare_agents import build_game_config

        d2h = DIRECTION_TO_HISSS
        wins = draws = 0
        turns: list[int] = []
        deaths: Counter = Counter()

        # Opponents are reused across the suite's games (each game gets a fresh
        # game.id and a start() call, so per-game state — Hungry's food memory,
        # PPOAgent's LSTM state — resets per game); constructing PPOAgent
        # opponents per game would rebuild the net 3x/game.
        opponents = [factory() for _ in range(3)]
        for gi in range(n_games):
            order = [ppo] + opponents
            shift = gi % 4                       # rotate seats to remove positional bias
            order = order[shift:] + order[:shift]
            ppo_seat = order.index(ppo)
            env = _hisss.BattleSnakeGame(build_game_config(4, 15, 15))
            gid = f"eval-{label}-{gi}"
            for idx, ag in enumerate(order):
                st = GameState.model_validate_json(_hisss.to_battlesnake_json(env, idx))
                st.game.id = gid
                try:
                    ag.start(st)
                except Exception:
                    pass
            tc = 0
            while not env.is_terminal() and tc < 1000:
                moves = []
                for idx in env.players_at_turn():
                    # No include_eliminated: the live protocol removes dead
                    # snakes, and ppo.py's reconstruction cannot tell a corpse
                    # from an alive enemy (health is fogged to 0 for all).
                    st = GameState.model_validate_json(
                        _hisss.to_battlesnake_json(env, idx))
                    st.game.id = gid
                    try:
                        mv = d2h[order[idx].move(st).move]
                    except Exception:
                        mv = _hisss.UP
                    legal = env.available_actions(idx)
                    mv = mv if mv in legal else (random.choice(legal) if legal else _hisss.UP)
                    moves.append(mv)
                env.step(actions=tuple(moves))
                tc += 1
            turns.append(tc)
            state = env.get_state()
            alive = [i for i, al in enumerate(state.snakes_alive) if al]
            if alive == [ppo_seat]:
                wins += 1
            elif ppo_seat in alive:
                draws += 1
            if state.elimination_events and ppo_seat in state.elimination_events:
                deaths[str(state.elimination_events[ppo_seat].cause)] += 1

        n = float(n_games)
        self.logger.record(f"eval/win_rate_{label}", wins / n)
        self.logger.record(f"eval/draw_rate_{label}", draws / n)
        self.logger.record(f"eval/avg_turns_{label}", sum(turns) / len(turns))
        for cause, c in deaths.items():
            self.logger.record(f"eval/death_{label}_{cause}", c / n)
        if self.verbose:
            print(f"[eval @ {self.num_timesteps} steps] vs 3x{label}: win_rate={wins/n:.2f} "
                  f"draws={draws} avg_turns={sum(turns)/len(turns):.1f} deaths={dict(deaths)}")
        return wins / n


def cosine_restarts_schedule(peaks=(2.5e-4, 1.25e-4, 6.25e-5), floor=1e-5):
    """Cosine LR with warm restarts over training progress (one cycle per
    peak, each decaying peak -> floor). The previous run's anneal-to-0 froze
    whichever basin the policy sat in near the end (it got lucky); restarts
    give the policy chances to leave a bad basin, while the last cycle still
    ends low for the consolidation effect that run demonstrated."""
    n_cycles = len(peaks)

    def schedule(progress_remaining: float) -> float:
        # progress_remaining goes 1.0 (start) -> 0.0 (end)
        p = min(max(1.0 - progress_remaining, 0.0), 1.0 - 1e-9)
        cycle, cycle_progress = divmod(p * n_cycles, 1.0)
        peak = peaks[int(cycle)]
        return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * cycle_progress))

    return schedule


if __name__ == "__main__":
    # 0. Create the (empty) self-play pool dir BEFORE the env workers spawn.
    # From scratch there are no seeds: the pool fills once the eval gate opens.
    ensure_pool_seeded()

    # 1. Initialize and validate the wrapper
    scale = True

    if not scale:
        env = BattlesnakeHisssEnv()
        check_env(env)
    else:
        env = make_vec_env(
            BattlesnakeHisssEnv,
            n_envs=32,          # CPU rollout/IPC is the bottleneck; 64 was 3x worse
            vec_env_cls=SubprocVecEnv,
        )

    print("Initializing RecurrentPPO model (from scratch, no warm start)...")
    # 2. RecurrentPPO (CNN -> LSTM -> heads) with tensorboard logging.
    # Architecture comes from obs_config.build_model so training and inference
    # can never diverge.
    model = build_model(
        env=env,
        device="cuda",
        verbose=1,
        tensorboard_log="./battlesnake_logs/",

        # Rollout: n_steps * n_envs = 32k samples per update, as before
        n_steps=1024,

        # Training
        learning_rate=cosine_restarts_schedule(),
        n_epochs=4,
        batch_size=256,

        # PPO-specific
        clip_range=0.2,
        ent_coef=0.01,         # keep some exploration (RecurrentPPO default is 0.0)
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,        # early-stop an update if policy moves too far (collapse guard)

        # GAE
        gamma=0.99,
        gae_lambda=0.95,
    )

    # Checkpoint at the same 2.5M cadence as the eval callback, so every eval
    # point has a restorable checkpoint (pick the shipped model by eval curve,
    # never blindly the final zip). save_freq counts *per-env* steps.
    n_envs = env.num_envs if hasattr(env, "num_envs") else 1
    checkpoint_callback = CheckpointCallback(
        save_freq=max(2_500_000 // n_envs, 1),
        save_path="./models/",
        name_prefix="ppo_bs_lstm",  # don't clobber earlier runs' checkpoints
    )
    # Snapshot into the opponent pool every 5M steps once the gate is open;
    # the eval callback opens the gate (order matters: eval before snapshot).
    snapshot_callback = SelfPlaySnapshotCallback(snapshot_freq=5_000_000)
    eval_callback = MaskedEvalCallback(eval_freq=2_500_000,
                                       snapshot_cb=snapshot_callback)

    model.learn(
        total_timesteps=40_000_000,
        progress_bar=False,
        reset_num_timesteps=True,
        callback=[checkpoint_callback, eval_callback, snapshot_callback],
    )
    # 4. Save the weights
    model.save("ppo_bs_lstm_final")
    print("Training complete. Model saved to ppo_bs_lstm_final.zip")
