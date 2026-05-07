from dataclasses import dataclass
from enum import IntEnum, Enum, auto
from typing import Tuple, Final


class RenderMode(str, Enum):
    HUMAN = "human"
    # Other render modes (e.g. text) could be added here in the future.


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
    EMPTY = auto()
    SNAKE = auto()
    APPLE = auto()
    OBSTACLE = auto()


class ObserveType(Enum):
    FULL_GRID = auto()
    VEC_11 = auto()


class DeathReason(str, Enum):
    WALL = "Wall"
    SELF = "Self"
    COMPLETE = "Complete"
    TRUNCATED = "Truncated"


# ------------------ Shared Dataclasses with default values ------------------ #


@dataclass(frozen=True)
class RewardOptions:
    eats_apple: float = 10.0
    complete: float = 50.0
    penalty_step: float = -0.1
    penalty_loop: float = -1.0
    death_wall: float = -10.0
    death_self: float = -10.0
    shaping_closer: float = 0.5
    shaping_further: float = -0.5


DEFAULT_REWARD: Final = RewardOptions()


@dataclass
class RenderOptions:
    cell_size: int = 40
    render_fps: int = 30
    agent_color: Tuple[int, int, int] = (52, 211, 153)


DEFAULT_RENDER_OPTIONS: Final = RenderOptions()
