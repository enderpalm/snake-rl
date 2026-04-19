from enum import IntEnum, Enum
from typing import Tuple, TypedDict
from dataclasses import dataclass


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

    @property
    def dir_offset(self) -> Tuple[int, int]:
        return DIR_OFFSETS[self]


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


class ObserveType:
    FULL = -1
    VEC_11 = 0
    # Any positive int > 0 is implicitly treated as a CNN radius


class DeathReason(str, Enum):
    WALL = "wall"
    SELF = "self"
    COMPLETE = "complete"
    TRUNCATED = "truncated"


# -------------------------------- TypedDicts -------------------------------- #


class RenderOptions(TypedDict, total=False):
    cell_size: int
    render_fps: int
    agent_color: Tuple[int, int, int]


@dataclass
class RewardOptions:
    reward_apple: float = 10.0
    reward_step: float = -0.01
    reward_loop_penalty: float = -0.1
    reward_complete: float = 50.0
    reward_death_wall: float = -10.0
    reward_death_self: float = -10.0
    reward_shaping_closer: float = 0.1
    reward_shaping_further: float = -0.2
