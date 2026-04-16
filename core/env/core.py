import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Any, Optional
from collections import deque

from core.env.types import (
    Direction,
    GridType,
    ObsType,
    RenderMode,
    RenderOptions,
    RewardOptions,
    DeathReason,
    DIR_OFFSETS,
)


class Snake:
    def __init__(
        self,
        start_pos: Tuple[int, int],
        length: int = 3,
        dir: Direction = Direction.RIGHT,
    ):
        self.dir, self.alive = dir, True
        dy, dx = DIR_OFFSETS[dir]
        self.body: deque[Tuple[int, int]] = deque(
            (start_pos[0] - dy * i, start_pos[1] - dx * i) for i in range(length)
        )

    def clone(self) -> "Snake":
        new_snake = Snake((0, 0), 1, self.dir)
        new_snake.body, new_snake.alive = self.body.copy(), self.alive
        return new_snake


class SnakeEnv(gym.Env):
    REWARD_APPLE, REWARD_STEP, REWARD_LOOP_PENALTY = 10.0, -0.01, -0.1
    REWARD_COMPLETE, REWARD_DEATH_WALL, REWARD_DEATH_SELF = 50.0, -10.0, -10.0
    REWARD_SHAPING_CLOSER, REWARD_SHAPING_FURTHER = 0.1, -0.2

    def __init__(
        self,
        render_mode: Optional[RenderMode] = None,
        width: int = 15,
        height: int = 15,
        obs_type: ObsType = ObsType.VECTOR_11,
        num_apples: int = 1,
        num_obstacles: int = 0,
        seed: Optional[int] = None,
        render_options: Optional[RenderOptions] = None,
        reward_options: Optional[RewardOptions] = None,
        max_steps: int = 2000,
    ):
        super().__init__()
        self.width, self.height, self.render_mode, self.obs_type, self.max_steps = (
            width,
            height,
            render_mode,
            obs_type,
            max_steps,
        )
        self.num_apples, self.num_obstacles = num_apples, num_obstacles
        self.np_random = np.random.default_rng(seed)

        self.snake: Optional[Snake] = None
        self.apples: set[Tuple[int, int]] = set()
        self.obstacles: set[Tuple[int, int]] = set()
        self.grid = np.full(
            (self.height, self.width), fill_value=GridType.EMPTY, dtype=np.int8
        )

        self.state_step, self.done = 0, False
        self.current_step, self.recent_positions = 0, deque(maxlen=20)
        self.last_death_reason: Optional[DeathReason] = None
        self.total_rewards, self.apples_eaten = 0.0, 0

        if reward_options:
            for k, v in reward_options.items():
                setattr(self, k.upper(), v)

        self.ui = None
        if self.render_mode == RenderMode.HUMAN:
            from core.pygame_ui import PygameUI

            opts = render_options or {}
            self.ui = PygameUI(
                cell_size=opts.get("cell_size", 40),
                fps=opts.get("render_fps", 30),
                agent_color=opts.get("agent_color"),
            )
            self.ui.init_screen(width, height)

        self.action_space = spaces.Discrete(3)
        self.observation_space = (
            spaces.MultiBinary(11)
            if self.obs_type == ObsType.VECTOR_11
            else spaces.Box(low=0, high=255, shape=(3, height, width), dtype=np.uint8)
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, dict]:
        super().reset(seed=seed, options=options)
        if options:
            self.num_apples = options.get("num_apples", self.num_apples)
            self.num_obstacles = options.get("num_obstacles", self.num_obstacles)

        self.state_step, self.done, self.current_step = 0, False, 0
        self.recent_positions.clear()
        self.last_death_reason, self.total_rewards, self.apples_eaten = None, 0.0, 0

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

        self._spawn_items(self.num_apples, self.apples, GridType.APPLE)
        self._spawn_items(self.num_obstacles, self.obstacles, GridType.OBSTACLE)
        return self._get_obs(), {
            "death_reason": None,
            "apples_eaten": 0,
            "engine_state": self.clone(),
        }

    def _spawn_items(self, count: int, item_set: set, grid_type: GridType):
        empty = np.flatnonzero(self.grid == GridType.EMPTY)
        if empty.size == 0:
            return
        for flat_idx in self.np_random.choice(
            empty, min(count, empty.size), replace=False
        ):
            r, c = divmod(flat_idx, self.width)
            item_set.add((r, c))
            self.grid[r, c] = grid_type

    def _min_dist_to_apple(self) -> float:
        if not self.apples or not self.snake:
            return 0.0
        r, c = self.snake.body[0]
        return min(abs(r - ar) + abs(c - ac) for ar, ac in self.apples)

    def _get_obs(self) -> Any:
        if self.obs_type == ObsType.FULL_GRID:
            state = np.zeros((3, self.height, self.width), dtype=np.uint8)
            if self.snake:
                for i, (r, c) in enumerate(self.snake.body):
                    state[0, r, c] = 255 if i == 0 else 128
            for r, c in self.apples:
                state[1, r, c] = 255
            for r, c in self.obstacles:
                state[2, r, c] = 255
            return state

        if not self.snake:
            return np.zeros(11, dtype=np.int8)

        r, c = self.snake.body[0]
        dy, dx = DIR_OFFSETS[self.snake.dir]

        def is_col(pr, pc):
            return not (0 <= pr < self.height and 0 <= pc < self.width) or self.grid[
                pr, pc
            ] in (GridType.OBSTACLE, GridType.BODY, GridType.HEAD)

        apple = next(iter(self.apples)) if self.apples else (r, c)

        return np.array(
            [
                is_col(r - dx, c + dy),
                is_col(r + dy, c + dx),
                is_col(r + dx, c - dy),
                *(
                    self.snake.dir == d
                    for d in (
                        Direction.LEFT,
                        Direction.RIGHT,
                        Direction.UP,
                        Direction.DOWN,
                    )
                ),
                apple[1] < c,
                apple[1] > c,
                apple[0] < r,
                apple[0] > r,
            ],
            dtype=np.int8,
        )

    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]:
        if self.done or not self.snake or not self.snake.alive:
            return (
                self._get_obs(),
                0.0,
                True,
                False,
                {
                    "death_reason": self.last_death_reason,
                    "apples_eaten": self.apples_eaten,
                    "engine_state": self.clone(),
                },
            )

        self.snake.dir = Direction((self.snake.dir + action - 1) % 4)
        dist_before = self._min_dist_to_apple()

        self.state_step += 1
        self.current_step += 1
        reward, death, hit_apple = 0.0, None, False
        snake = self.snake

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
            reward += self.REWARD_DEATH_WALL
        else:
            target = self.grid[new_head[0], new_head[1]]
            if target == GridType.OBSTACLE or (
                target in (GridType.BODY, GridType.HEAD)
                and not (new_head == snake.body[-1] and new_head not in self.apples)
            ):
                _die(
                    DeathReason.WALL
                    if target == GridType.OBSTACLE
                    else DeathReason.SELF
                )
                reward += (
                    self.REWARD_DEATH_WALL
                    if target == GridType.OBSTACLE
                    else self.REWARD_DEATH_SELF
                )
            else:
                snake.body.appendleft(new_head)
                if len(snake.body) > 1:
                    self.grid[snake.body[1][0], snake.body[1][1]] = GridType.BODY
                self.grid[new_head[0], new_head[1]] = GridType.HEAD

                if new_head in self.apples:
                    hit_apple = True
                    self.apples.remove(new_head)
                    if len(self.apples) < self.num_apples:
                        self._spawn_items(1, self.apples, GridType.APPLE)
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

        terminated = not self.snake.alive or self.done
        if not self.snake.alive and not self.last_death_reason:
            self.last_death_reason = death

        pure_movement = not hit_apple and not terminated
        if pure_movement and self.apples:
            reward += (
                self.REWARD_SHAPING_CLOSER
                if self._min_dist_to_apple() < dist_before
                else self.REWARD_SHAPING_FURTHER
            )
            head = snake.body[0]
            if head in self.recent_positions:
                reward += getattr(self, "REWARD_LOOP_PENALTY", -0.25)
            self.recent_positions.append(head)
        elif hit_apple or (reward > 2.0 and not pure_movement):
            self.recent_positions.clear()

        self.total_rewards += reward
        if hit_apple:
            self.apples_eaten += 1

        truncated = self.current_step >= self.max_steps
        if truncated and not self.last_death_reason:
            self.last_death_reason = DeathReason.TRUNCATED

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {
                "death_reason": self.last_death_reason,
                "apples_eaten": self.apples_eaten,
                "engine_state": self.clone(),
            },
        )

    def clone(self) -> "SnakeEnv":
        new_env = SnakeEnv(
            render_mode=None,
            width=self.width,
            height=self.height,
            obs_type=self.obs_type,
            num_apples=self.num_apples,
            num_obstacles=self.num_obstacles,
            max_steps=self.max_steps,
        )
        new_env.np_random.bit_generator.state = self.np_random.bit_generator.state
        if self.snake:
            new_env.snake = self.snake.clone()
        new_env.apples, new_env.obstacles = set(self.apples), set(self.obstacles)
        new_env.grid = self.grid.copy()
        new_env.state_step, new_env.done, new_env.current_step = (
            self.state_step,
            self.done,
            self.current_step,
        )
        new_env.recent_positions = self.recent_positions.copy()
        new_env.last_death_reason, new_env.total_rewards, new_env.apples_eaten = (
            self.last_death_reason,
            self.total_rewards,
            self.apples_eaten,
        )
        return new_env

    def render(self):
        if self.render_mode == RenderMode.HUMAN and self.ui is not None:
            if not self.ui.handle_events(self):
                return sys.exit(0)
            while self.ui.paused:
                if not self.ui.handle_events(self):
                    return sys.exit(0)
                self.ui.render(self, self.total_rewards)
            self.ui.render(self, self.total_rewards)

    def close(self):
        if self.ui:
            self.ui.quit()
            self.ui = None
