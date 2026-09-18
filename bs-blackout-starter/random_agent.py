import numpy as np

from battlesnake_types import GameState, MoveAction, Direction, BaseAgent


def get_obstacle_map(game_state: GameState):
    obstacle_map = np.zeros((game_state.board.height, game_state.board.width), dtype=bool)
    
    for snake in game_state.board.snakes:
        for body_part in snake.body[:-1]:
            if body_part is None:
                # we don't see this body section, could be many parts long
                continue
            obstacle_map[body_part.y, body_part.x] = 1
    
    # print(obstacle_map.astype(int)[::-1])
    return obstacle_map


class RandomAgent(BaseAgent):

    def get_name(self):
        return "Mr. Unpredictable"

    def get_color(self):
        return '#32CD32'

    def get_author(self):
        return "Chaos Itself"

    def start(self, game_state: GameState):
        """start is called when the battlesnake begins a game"""
        pass

    def move(self, game_state: GameState) -> MoveAction:
        """move is called on every turn and returns your next move"""
        head = game_state.you.head
        assert head is not None

        # build an obstacle map
        obstacle_map = get_obstacle_map(game_state)

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

        return MoveAction(move=result_direction)

    def end(self, game_state: GameState):
        """end is called when the battlesnake finishes a game"""
        pass


if __name__ == "__main__":
    import sys
    from battlesnake_server import start_server

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <port>")
        sys.exit(1)

    agent = RandomAgent()
    port = int(sys.argv[1])

    start_server(agent=agent, port=port)
