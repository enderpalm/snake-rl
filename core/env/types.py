from enum import IntEnum, Enum
from typing import Tuple, TypedDict


class RenderMode(str, Enum):
    HUMAN = "human"
    RGB_ARRAY = "rgb_array"


class Action(IntEnum):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


DIR_OFFSETS = {
    Direction.UP: (-1, 0),
    Direction.RIGHT: (0, 1),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
}


class GridType(IntEnum):
    EMPTY = 0
    HEAD = 1
    APPLE = 2
    BODY = 3
    OBSTACLE = -1


class ObsType(Enum):
    VECTOR_11 = "11_dim"
    FULL_GRID = "full_grid"
    ALL = "all"


class DeathReason(str, Enum):
    WALL = "wall"
    SELF = "self"
    COMPLETE = "complete"
    TRUNCATED = "truncated"


class RenderOptions(TypedDict, total=False):
    cell_size: int
    render_fps: int
    agent_color: Tuple[int, int, int]


class RewardOptions(TypedDict, total=False):
    reward_apple: float
    reward_step: float
    reward_loop_penalty: float
    reward_complete: float
    reward_death_wall: float
    reward_death_self: float
    reward_shaping_closer: float
    reward_shaping_further: float
