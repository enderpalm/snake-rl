import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.env.core import SnakeEnv
from core.env.types import DIR_OFFSETS, Direction, GridType


def observe_vec11(env: "SnakeEnv") -> npt.NDArray[np.uint8]:
    # Numpy boolean still uses 1 byte, use uint8 to be unified type with more complex observations
    if not env.snake:
        return np.zeros(11, dtype=np.uint8)
    r, c = env.snake.body[0]
    dy, dx = DIR_OFFSETS[env.snake.dir]
    y, x = next(iter(env.apples)) if env.apples else (r, c)

    def col(pr, pc):
        return not (0 <= pr < env.height and 0 <= pc < env.width) or env.grid[
            pr, pc
        ] in (
            GridType.OBSTACLE,
            GridType.SNAKE,
        )

    return np.array(
        [
            col(r - dx, c + dy),
            col(r + dy, c + dx),
            col(r + dx, c - dy),
            env.snake.dir == Direction.LEFT,
            env.snake.dir == Direction.RIGHT,
            env.snake.dir == Direction.UP,
            env.snake.dir == Direction.DOWN,
            x < c,
            x > c,
            y < r,
            y > r,
        ],
        dtype=np.uint8,
    )


def observe_full_grid(env: "SnakeEnv") -> npt.NDArray[np.float32]:
    """
    Observes the full grid state. Consist of 4 channels:
    - Channel 0: Head (1.0)
    - Channel 1: Body TTL (normalized by Total_Grid_Cells)
    - Channel 2: Apples (1.0)
    - Channel 3: Obstacles/Walls (1.0)
    """
    grid_obs = np.zeros((4, env.height, env.width), dtype=np.float32)
    total_cells = env.height * env.width

    if env.snake and env.snake.body:
        hr, hc = env.snake.body[0]
        grid_obs[0, hr, hc] = 1.0  # Head

        snake_len = len(env.snake.body)
        for i, (r, c) in enumerate(env.snake.body):
            ttl = float(snake_len - i)
            grid_obs[1, r, c] = ttl / total_cells  # Body TTL

    for r, c in env.apples:
        grid_obs[2, r, c] = 1.0

    for r, c in env.obstacles:
        grid_obs[3, r, c] = 1.0

    return grid_obs
