from enum import Enum
from abc import abstractmethod

from pydantic import BaseModel
from typing import List, Optional, Tuple

# ---------------------------------------------------------
# Misc Models
# ---------------------------------------------------------
class Point(BaseModel):
    x: int
    y: int

class Food(Point):
    spawn_turn: int

class EliminatedCause(str, Enum):
    EliminatedByCollision = "snake-collision"
    EliminatedBySelfCollision = "snake-self-collision"
    EliminatedByOutOfHealth = "out-of-health"
    EliminatedByHeadToHeadCollision = "head-collision"
    EliminatedByOutOfBounds = "wall-collision"

class EliminationEvent(BaseModel):
    cause: EliminatedCause
    turn: int
    by: Optional[str] = None

# ---------------------------------------------------------
# Snake Model
# ---------------------------------------------------------
class SnakeCustomizations(BaseModel):
    color: Tuple[int, int, int]
    head: Optional[str]
    tail: Optional[str]

class Snake(BaseModel):
    id: str
    name: str
    length: int
    latency: Optional[str]
    squad: Optional[str]
    health: Optional[int]
    head: Optional[Point]
    body: List[Optional[Point]]
    customizations: SnakeCustomizations
    elimination_event: Optional[EliminationEvent] = None

# ---------------------------------------------------------
# Game & Ruleset Models
# ---------------------------------------------------------
class RoyaleSettings(BaseModel):
    shrinkEveryNTurns: int

class SquadSettings(BaseModel):
    allowBodyCollisions: bool
    sharedElimination: bool
    sharedHealth: bool
    sharedLength: bool

class RulesetSettings(BaseModel):
    foodSpawnChance: int
    hazardDamagePerTurn: int
    minimumFood: int
    viewRadius: Optional[int]
    royale: RoyaleSettings
    squad: SquadSettings

class Ruleset(BaseModel):
    name: str
    version: str
    settings: RulesetSettings

class Game(BaseModel):
    id: str
    source: str
    timeout: int
    ruleset: Ruleset

# ---------------------------------------------------------
# Board & Root GameState Models
# ---------------------------------------------------------
class Board(BaseModel):
    height: int
    width: int
    food: List[Food]
    hazards: List[Point] # Assuming hazards are basic x/y coordinates
    snakes: List[Snake]

class GameState(BaseModel):
    turn: int
    game: Game
    board: Board
    you: Snake

# ---------------------------------------------------------
# Snake Action Model
# ---------------------------------------------------------

class Direction(str, Enum):
    UP = 'up'
    RIGHT = 'right'
    DOWN = 'down'
    LEFT = 'left'

    @property
    def board_delta(self) -> Tuple[int, int]:
        if self is Direction.DOWN:
            return 0, -1
        if self is Direction.UP:
            return 0, 1
        if self is Direction.LEFT:
            return -1, 0
        if self is Direction.RIGHT:
            return 1, 0
    
    @classmethod
    def from_board_delta(cls, delta: Tuple[int, int]):
        for direction in cls:
            if direction.board_delta == delta:
                return direction
    
    @property
    def dx(self) -> int:
        return self.board_delta[0]

    @property
    def dy(self) -> int:
        return self.board_delta[1]

class MoveAction(BaseModel):
    move: Direction

# ---------------------------------------------------------
# Base Agent Interface
# ---------------------------------------------------------

class BaseAgent:

    @abstractmethod
    def get_name(self):
        pass

    def get_color(self):
        return '#32CD32'

    def get_author(self):
        return None

    @abstractmethod
    def start(self, game_state: GameState):
        pass

    @abstractmethod
    def move(self, game_state: GameState) -> MoveAction:
        pass

    @abstractmethod
    def end(self, game_state: GameState):
        pass
