from dataclasses import dataclass
import heapq
import numpy as np
from typing import List, Tuple

from battlesnake_types import Food, GameState, MoveAction, Direction, BaseAgent, Point

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def get_obstacle_map(game_state: GameState):
    obstacle_map = np.zeros((game_state.board.height, game_state.board.width), dtype=bool)
    
    for snake in game_state.board.snakes:
        for body_part in snake.body[:-1]:
            if body_part is None:
                # we don't see this body section, could be many parts long
                continue
            obstacle_map[body_part.y, body_part.x] = 1
    
    return obstacle_map

def get_vision_mask(width: int, height: int, center: Point, radius: int) -> np.ndarray:
    y, x = np.ogrid[:height, :width]
    dist_sq = abs(x - center.x) + abs(y - center.y)
    return dist_sq <= radius

# ---------------------------------------------------------
# Battlesnake Agent Implementation
# ---------------------------------------------------------
@dataclass
class AgentState:
    possible_food: list[Food]

class HungryAgent(BaseAgent):
    def __init__(self):
        self.agent_states: dict[str, AgentState] = {}

    def get_name(self):
        return "Hungry Caterpillar"

    def get_color(self):
        return "#FFC13C"

    def get_author(self):
        return "Gluttony"

    def start(self, game_state: GameState):
        """start is called when the battlesnake begins a game"""
        self.agent_states[game_state.game.id] = AgentState(possible_food=[])

    def move(self, game_state: GameState) -> MoveAction:
        """move is called on every turn and returns your next move"""
        # update the food memory (saved across move steps)
        agent_state = self.agent_states[game_state.game.id]
        head = game_state.you.head
        view_radius = game_state.game.ruleset.settings.viewRadius
        assert head is not None
        assert view_radius is not None
        vision_mask = get_vision_mask(width=game_state.board.width, height=game_state.board.height, center=head, radius=view_radius)

        updated_food = []
        # keep food that is not in vision range in memory
        for food in agent_state.possible_food:
            if not vision_mask[food.y, food.x]:
                updated_food.append(food)
        # add visible food
        visible_food = game_state.board.food
        for food in visible_food:
            if food not in updated_food:
                updated_food.append(food)
        agent_state.possible_food = updated_food

        # build an obstacle map
        obstacle_map = get_obstacle_map(game_state)

        # use A* to find closest food and the required move
        result_direction = None
        min_distance = float('inf')
        for food in agent_state.possible_food:
            direction, length = a_star_wrapper(obstacle_map, head, food)
            if length < min_distance:
                result_direction = direction
                min_distance = length
        
        # first fallback (if no food or cannot reach food): follow own tail
        if result_direction is None:
            tail = game_state.you.body[-1]
            assert tail is not None
            result_direction, _ = a_star_wrapper(obstacle_map, head, tail)

        # second fallback (if tail is unreachable): random move
        if result_direction is None:
            result_direction = self.random_fallback_move(game_state, obstacle_map)

        return MoveAction(move=result_direction)
    
    def random_fallback_move(self, game_state: GameState, obstacle_map: np.ndarray) -> Direction:
        head = game_state.you.head
        assert head is not None
        clear_directions = []
        for d in Direction:
            # check out-of-bounds
            if head.x + d.dx < 0 or head.x + d.dx >= game_state.board.width:
                continue
            if head.y + d.dy < 0 or head.y + d.dy >= game_state.board.height:
                continue

            # check for obstacles (i.e., snake parts)
            if obstacle_map[head.y + d.dy, head.x + d.dx]:
                continue

            clear_directions.append(d)
        
        result_direction = clear_directions[np.random.choice(len(clear_directions))] if clear_directions else Direction.UP

        return result_direction

    def end(self, game_state: GameState):
        """end is called when the battlesnake finishes a game"""
        if game_state.game.id in self.agent_states:
            del self.agent_states[game_state.game.id]


# ---------------------------------------------------------
# A* Algorithm
# ---------------------------------------------------------
def a_star_wrapper(grid: np.ndarray, start: Point, goal: Point) -> tuple[Direction | None, int]:
    """Converts from battlesnake x-y coords to i-j index-tuples used by a_star()."""
    path = a_star(grid, (start.y, start.x), (goal.y, goal.x))
    if path is None:
        return None, 9999999

    next_pos = path[1]
    result_direction = Direction.from_board_delta((next_pos[1] - start.x, next_pos[0] - start.y))
    return result_direction, len(path) 

def a_star(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]] | None:
    h, w = grid.shape
    open_set, g_score, came_from = [(0, start)], {start: 0}, {}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not grid[nr, nc]:
                neighbor, new_g = (nr, nc), g_score[current] + 1

                if new_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = new_g
                    f_score = new_g + abs(nr - goal[0]) + abs(nc - goal[1])
                    heapq.heappush(open_set, (f_score, neighbor))

    return None


if __name__ == "__main__":
    import sys
    from battlesnake_server import start_server

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <port>")
        sys.exit(1)

    agent = HungryAgent()
    port = int(sys.argv[1])

    start_server(agent=agent, port=port)
