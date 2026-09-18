import os
import sys
import json
import logging
from flask import Flask, jsonify, request

# Ensure local imports work correctly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import hisss
from battlesnake_types import GameState, Direction
from random_agent import RandomAgent
from hungry_agent import HungryAgent
from best_agent import BestAgent
from ppo import PPOAgent
from ppo2 import PPOAgent2

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visualizer_server")

app = Flask(__name__, static_url_path='', static_folder='static')

AGENT_CLASSES = {
    "HungryAgent": HungryAgent,
    "RandomAgent": RandomAgent,
    "BestAgent": BestAgent,
    "PPO2": PPOAgent2,
    "PPO": PPOAgent,
}

DIRECTION_TO_HISSS = {
    Direction.UP: hisss.UP,
    Direction.DOWN: hisss.DOWN,
    Direction.LEFT: hisss.LEFT,
    Direction.RIGHT: hisss.RIGHT
}

def run_simulation(agent_ids, width, height, view_radius):
    # Instantiate agents
    agents = []
    for aid in agent_ids:
        cls = AGENT_CLASSES.get(aid, RandomAgent)
        agents.append(cls())

    # Build hisss configuration
    if view_radius is not None:
        ec = hisss.game.config.BestRestrictedEncodingConfig()
    else:
        ec = hisss.game.config.BestBattleSnakeEncodingConfig()
    ec.compress_enemies = False

    game_cfg = hisss.BattleSnakeConfig(
        w=width,
        h=height,
        num_players=len(agents),
        min_food=1,
        food_spawn_chance=15,
        ec=ec,
        init_snake_len=[3] * len(agents),
        all_actions_legal=True,
        view_radius=view_radius
    )

    env = hisss.BattleSnakeGame(game_cfg)

    # Stable game ID for agents with memory (e.g. HungryAgent)
    stable_game_id = "local-game-session"

    # Call start for each agent
    for idx, agent in enumerate(agents):
        state_json = hisss.to_battlesnake_json(env, idx)
        state_dict = json.loads(state_json)
        state_dict["game"]["id"] = stable_game_id
        agent.start(GameState.model_validate(state_dict))

    history = []

    # Helper to extract turn details
    def get_turn_state(env, turn_actions=None):
        state = env.get_state()
        hazards_arr = env.get_hazards()
        hazards_list = []
        h, w = hazards_arr.shape
        for y in range(h):
            for x in range(w):
                if hazards_arr[y, x]:
                    hazards_list.append({"x": int(x), "y": int(y)})

        snakes_info = []
        for idx in range(env.num_players):
            agent = agents[idx]
            name = agent.get_name() if hasattr(agent, "get_name") else f"Snake {idx}"
            color = agent.get_color() if hasattr(agent, "get_color") else "#FF0000"
            
            is_alive = state.snakes_alive[idx]
            body_coords = state.snake_pos[idx]
            body_list = [{"x": int(p[0]), "y": int(p[1])} for p in body_coords]

            elim_event = None
            if state.elimination_events and idx in state.elimination_events:
                ev = state.elimination_events[idx]
                elim_event = {
                    "cause": str(ev.cause),
                    "turn": int(ev.turn),
                    "by": f"snake-{ev.by}" if ev.by is not None and isinstance(ev.by, int) else (str(ev.by) if ev.by is not None else None)
                }

            visible_tiles = []
            if is_alive and body_coords and view_radius is not None:
                hx, hy = body_coords[0]
                for ty in range(env.cfg.h):
                    for tx in range(env.cfg.w):
                        if abs(tx - hx) + abs(ty - hy) <= view_radius:
                            visible_tiles.append({"x": int(tx), "y": int(ty)})

            food_memory = []
            if hasattr(agent, "agent_states") and stable_game_id in agent.agent_states:
                agent_state = agent.agent_states[stable_game_id]
                if hasattr(agent_state, "possible_food") and agent_state.possible_food:
                    food_memory = [{"x": int(f.x), "y": int(f.y)} for f in agent_state.possible_food]

            snakes_info.append({
                "id": f"snake-{idx}",
                "name": name,
                "color": color,
                "health": int(state.snake_health[idx]) if is_alive else 0,
                "length": int(state.snake_len[idx]),
                "body": body_list,
                "alive": is_alive,
                "elimination_event": elim_event,
                "visible_tiles": visible_tiles,
                "food_memory": food_memory
            })

        return {
            "turn": int(state.turn),
            "width": int(env.cfg.w),
            "height": int(env.cfg.h),
            "food": [{"x": int(f[0]), "y": int(f[1])} for f in state.food_pos],
            "hazards": hazards_list,
            "snakes": snakes_info,
            "view_radius": int(view_radius) if view_radius is not None else None,
            "actions": turn_actions
        }

    # Record initial state
    history.append(get_turn_state(env))

    # Main simulation loop
    max_turns = 2000
    turn = 0
    while not env.is_terminal() and turn < max_turns:
        moves = []
        agent_actions = []
        
        # Get moves from all alive players who are at turn
        for idx, agent in enumerate(agents):
            if not env.is_player_at_turn(idx):
                agent_actions.append(None)
                continue

            # Export game state from player's perspective
            state_json = hisss.to_battlesnake_json(env, idx, include_eliminated=True)
            state_dict = json.loads(state_json)
            state_dict["game"]["id"] = stable_game_id
            game_state = GameState.model_validate(state_dict)

            # Call agent move
            try:
                move_result = agent.move(game_state)
                chosen_dir = move_result.move
                moves.append(DIRECTION_TO_HISSS[chosen_dir])
                agent_actions.append(chosen_dir.value)
            except Exception as e:
                logger.error(f"Agent {idx} crashed on turn {turn}: {e}")
                moves.append(hisss.UP)
                agent_actions.append("up")

        # Step the environment
        env.step(actions=tuple(moves))
        turn += 1

        # Record new state
        history.append(get_turn_state(env, agent_actions))

    # Call end for all agents
    for idx, agent in enumerate(agents):
        state_json = hisss.to_battlesnake_json(env, idx)
        state_dict = json.loads(state_json)
        state_dict["game"]["id"] = stable_game_id
        try:
            agent.end(GameState.model_validate(state_dict))
        except Exception as e:
            logger.error(f"Agent {idx} end() crashed: {e}")

    env.close()
    return history

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/agents', methods=['GET'])
def get_agents():
    agents_list = []
    for aid, cls in AGENT_CLASSES.items():
        # Instantiate agent briefly to read details if needed, or use static info
        try:
            agent_inst = cls()
            name = agent_inst.get_name()
            color = agent_inst.get_color()
            author = agent_inst.get_author() or "Unknown"
        except Exception:
            name = aid
            color = "#FF0000"
            author = "Unknown"
        
        agents_list.append({
            "id": aid,
            "name": name,
            "color": color,
            "author": author
        })
    return jsonify(agents_list)

@app.route('/api/simulate', methods=['POST'])
def simulate():
    data = request.json or {}
    width = int(data.get('width', 15))
    height = int(data.get('height', 15))
    
    view_radius = data.get('view_radius')
    if view_radius is not None:
        view_radius = int(view_radius)
        
    snakes = data.get('snakes', [])
    if not snakes:
        return jsonify({"error": "No snakes specified"}), 400
        
    try:
        history = run_simulation(snakes, width, height, view_radius)
        return jsonify({
            "success": True,
            "history": history
        })
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False)
