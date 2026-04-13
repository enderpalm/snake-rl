from enum import IntEnum, Enum

class RenderMode(str, Enum):
    HUMAN = "human"
    RGB_ARRAY = "rgb_array"

class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @property
    def x(self):
        return {
            Direction.UP: 0,
            Direction.RIGHT: 1,
            Direction.DOWN: 0,
            Direction.LEFT: -1,
        }[self]

    @property
    def y(self):
        return {
            Direction.UP: -1,
            Direction.RIGHT: 0,
            Direction.DOWN: 1,
            Direction.LEFT: 0,
        }[self]

    @property
    def opposite(self):
        return {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }[self]


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
