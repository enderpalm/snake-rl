import numpy as np
from typing import Tuple, Dict, Any, Optional
from collections import deque
from core.env.enums import Direction, GridType, DeathReason, DIR_OFFSETS


class Snake:
    def __init__(
        self,
        start_pos: Tuple[int, int],
        length: int = 3,
        dir: Direction = Direction.RIGHT,
    ):
        self.dir = dir
        self.alive = True
        dy, dx = DIR_OFFSETS[dir]
        self.body: deque[Tuple[int, int]] = deque(
            (start_pos[0] - dy * i, start_pos[1] - dx * i) for i in range(length)
        )

    def clone(self) -> "Snake":
        new_snake = Snake((0, 0), 1, self.dir)
        new_snake.body = self.body.copy()
        new_snake.alive = self.alive
        return new_snake


class SnakeEngine:
    REWARD_APPLE = 50.0
    REWARD_STEP = -0.05
    REWARD_LOOP_PENALTY = -0.25
    REWARD_COMPLETE = 100.0
    REWARD_DEATH_WALL = -10.0
    REWARD_DEATH_SELF = -10.0

    # Reward Shaping (for gym wrapper)
    REWARD_SHAPING_CLOSER = 0.05
    REWARD_SHAPING_FURTHER = -0.05

    def __init__(
        self,
        width: int = 15,
        height: int = 15,
        num_apples: int = 1,
        num_obstacles: int = 0,
        seed: Optional[int] = None,
    ):
        self.width, self.height = width, height
        self.num_apples, self.num_obstacles = num_apples, num_obstacles

        self.np_random = np.random.default_rng(seed)
        self.snake: Optional[Snake] = None
        self.apples: set[Tuple[int, int]] = set()
        self.obstacles: set[Tuple[int, int]] = set()
        self.grid = np.full(
            (self.height, self.width), fill_value=GridType.EMPTY, dtype=np.int8
        )
        self.state_step, self.done = 0, False

    def reset(self) -> None:
        self.state_step, self.done = 0, False

        self.snake = Snake(
            start_pos=(self.height // 2, self.width // 2),
            length=3,
            dir=Direction(self.np_random.integers(0, 4)),
        )

        self.apples.clear()
        self.obstacles.clear()
        self.grid.fill(GridType.EMPTY)

        for i, (r, c) in enumerate(self.snake.body):
            self.grid[r, c] = GridType.HEAD if i == 0 else GridType.BODY

        self._spawn_apples(self.num_apples)
        self._spawn_obstacles(self.num_obstacles)

    def _spawn_items(self, count: int, item_set: set, grid_type: GridType):
        empty = list(zip(*np.where(self.grid == GridType.EMPTY)))
        if not empty:
            return
        for idx in self.np_random.choice(
            len(empty), min(count, len(empty)), replace=False
        ):
            r, c = empty[idx]
            item_set.add((r, c))
            self.grid[r, c] = grid_type

    def _spawn_apples(self, count: int) -> None:
        self._spawn_items(count, self.apples, GridType.APPLE)

    def _spawn_obstacles(self, count: int) -> None:
        self._spawn_items(count, self.obstacles, GridType.OBSTACLE)

    def _add_item_dyn(
        self, pos: Tuple[int, int], item_set: set, grid_type: GridType
    ) -> bool:
        r, c = pos
        if (
            0 <= r < self.height
            and 0 <= c < self.width
            and self.grid[r, c] == GridType.EMPTY
        ):
            item_set.add(pos)
            self.grid[r, c] = grid_type
            return True
        return False

    def add_obstacle(self, pos: Tuple[int, int]) -> bool:
        return self._add_item_dyn(pos, self.obstacles, GridType.OBSTACLE)

    def add_apple(self, pos: Tuple[int, int]) -> bool:
        return self._add_item_dyn(pos, self.apples, GridType.APPLE)

    def step(self, action: Optional[int]) -> Dict[str, Any]:
        if self.done or self.snake is None:
            return {"rewards": 0.0, "deaths": None}

        self.state_step += 1
        reward = 0.0
        death: Optional[DeathReason] = None

        snake = self.snake
        if not snake.alive:
            return {"rewards": reward, "deaths": death}

        if action is not None:
            new_dir = Direction(action)
            if (new_dir + 2) % 4 != snake.dir:
                snake.dir = new_dir

        r, c = snake.body[0]
        dy, dx = DIR_OFFSETS[snake.dir]
        new_head = (r + dy, c + dx)

        def _die(reason: DeathReason):
            snake.alive, nonlocal_death[0] = False, reason
            for br, bc in snake.body:
                if self.grid[br, bc] in (GridType.HEAD, GridType.BODY):
                    self.grid[br, bc] = GridType.EMPTY

        nonlocal_death = [death]

        if not (0 <= new_head[0] < self.height and 0 <= new_head[1] < self.width):
            _die(DeathReason.WALL)
        else:
            target = self.grid[new_head[0], new_head[1]]
            if target == GridType.OBSTACLE:
                _die(DeathReason.WALL)
            elif target in (GridType.BODY, GridType.HEAD) and not (
                new_head == snake.body[-1] and new_head not in self.apples
            ):
                _die(DeathReason.SELF)
            else:
                snake.body.appendleft(new_head)
                if len(snake.body) > 1:
                    self.grid[snake.body[1][0], snake.body[1][1]] = GridType.BODY
                self.grid[new_head[0], new_head[1]] = GridType.HEAD

                if new_head in self.apples:
                    self.apples.remove(new_head)
                    if len(self.apples) < self.num_apples:
                        self._spawn_apples(1)
                    reward += self.REWARD_APPLE
                else:
                    tr, tc = snake.body.pop()
                    self.grid[tr, tc] = GridType.EMPTY
                    reward += self.REWARD_STEP

                if len(snake.body) == self.width * self.height - len(self.obstacles):
                    _die(DeathReason.COMPLETE)
                    reward += self.REWARD_COMPLETE

        death = nonlocal_death[0]
        self.done = not snake.alive

        death_penalties = {
            DeathReason.WALL: self.REWARD_DEATH_WALL,
            DeathReason.SELF: self.REWARD_DEATH_SELF,
        }
        if death in death_penalties:
            reward += death_penalties[death]

        return {"rewards": reward, "deaths": death}

    def clone(self) -> "SnakeEngine":
        new_engine = SnakeEngine(
            self.width, self.height, self.num_apples, self.num_obstacles
        )
        new_engine.np_random.bit_generator.state = self.np_random.bit_generator.state
        new_engine.snake = self.snake.clone()
        new_engine.apples = set(self.apples)
        new_engine.obstacles = set(self.obstacles)
        new_engine.grid = self.grid.copy()
        new_engine.state_step = self.state_step
        new_engine.done = self.done
        return new_engine

    def render_grid(self) -> np.ndarray:
        return self.grid.copy()
