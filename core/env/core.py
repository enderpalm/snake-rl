import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import numpy.typing as npt
from typing import SupportsFloat, Tuple, Any
from collections import deque
from dataclasses import replace

import core.env.observations as obs
from core.env.types import (
    DEFAULT_RENDER_OPTIONS,
    DEFAULT_REWARD,
    Action,
    Direction,
    GridType,
    ObserveType,
    RenderMode,
    RenderOptions,
    DeathReason,
    DIR_OFFSETS,
    RewardOptions,
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
        self.body: deque[Tuple[int, int]] = deque((start_pos[0] - dy * i, start_pos[1] - dx * i) for i in range(length))
        self.recent_positions: deque[Tuple[int, int]] = deque(maxlen=20)
        self.last_death_reason: DeathReason | None = None
        self.total_rewards: float = 0.0
        self.apples_eaten: int = 0

    def clone(self) -> "Snake":
        new_snake = Snake((0, 0), 1, self.dir)
        new_snake.body = self.body.copy()
        new_snake.alive = self.alive
        new_snake.recent_positions = self.recent_positions.copy()
        new_snake.last_death_reason = self.last_death_reason
        new_snake.total_rewards = self.total_rewards
        new_snake.apples_eaten = self.apples_eaten
        return new_snake


# --------------- SnakeEnv: Single-agent Gymnasium environment --------------- #




class SnakeEnv(gym.Env):
    """
    A Gymnasium environment for single-agent Snake. The snake starts in the middle of the grid,
    and can move in 3 relative directions (straight, left, right). The goal is to eat apples that spawn randomly on the grid, while avoiding collisions with walls, obstacles, and itself.

    The environment supports two types of observations:
    - Partially observable, 11-dimensional vector (based on this [paper](https://www.researchgate.net/publication/387389306_Comparative_Evaluation_of_Reinforcement_Learning_Algorithms_on_the_Snake_Game))
    - Fully observable 3-channel grid (head/body/tail, apples, obstacles) for CNN-based agents

    Rewards are given for eating apples, taking steps, and penalties for dying or looping.
    The environment can be rendered and interacted with using Pygame for visualization.
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        obs_type: ObserveType | int = ObserveType.VEC_11,
        num_apples: int = 1,
        num_obstacles: int = 0,
        max_steps: int = 2000,
        seed: int | None = None,
        render_mode: RenderMode | None = None,
        render_options: RenderOptions = DEFAULT_RENDER_OPTIONS,
        reward_options: RewardOptions = DEFAULT_REWARD,
        snapshot_engine_state: bool = False,
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
        self._np_random_seed = seed

        self.rewards: RewardOptions = (
            reward_options
            if isinstance(reward_options, RewardOptions)
            else replace(DEFAULT_REWARD, **(reward_options or {}))
        )

        self.snake: Snake | None = None
        self.apples: set[Tuple[int, int]] = set()
        self.obstacles: set[Tuple[int, int]] = set()
        self.grid = np.full((self.height, self.width), fill_value=GridType.EMPTY, dtype=np.uint8)

        # For single-agent env, done is equivalent to 'snake not being alive'
        self.step_count = 0
        self.snapshot_engine_state = snapshot_engine_state

        # Initialize rendering
        self.ui = None
        if self.render_mode == RenderMode.HUMAN:
            from core.pygame_ui import PygameUI

            self.ui = PygameUI(render_options)
            self.ui.init_screen(width, height)

        # Not explicity used, but good to have for gym.env compliance
        self.action_space = spaces.Discrete(3)
        self.observation_space = (
            spaces.MultiBinary(11)
            if self.obs_type == ObserveType.VEC_11
            else spaces.Box(low=0, high=3, shape=(3, height, width), dtype=np.uint8)
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[Any, dict]:
        super().reset(seed=seed)

        self.step_count = 0
        self.snake = Snake(
            start_pos=(self.height // 2, self.width // 2),
            length=3,
            dir=Direction(self.np_random.integers(0, 4)),
        )
        self.apples.clear()
        self.obstacles.clear()
        self.grid.fill(GridType.EMPTY)

        for r, c in self.snake.body:
            self.grid[r, c] = GridType.SNAKE

        self._spawn_items(self.num_apples, self.apples, GridType.APPLE)
        self._spawn_items(self.num_obstacles, self.obstacles, GridType.OBSTACLE)
        return self._get_observation(), self._get_info()

    def _spawn_items(self, count: int, item_set: set, grid_type: GridType):
        empty = np.flatnonzero(self.grid == GridType.EMPTY)
        if empty.size == 0:
            return
        for flat_idx in self.np_random.choice(empty, min(count, empty.size), replace=False):
            r, c = divmod(flat_idx, self.width)
            item_set.add((r, c))
            self.grid[r, c] = grid_type

    def set_item_dyn(self, pos: Tuple[int, int], grid_type: GridType, add: bool = True):
        if self.grid[pos] is GridType.SNAKE:
            return

        target_set = self.apples if grid_type == GridType.APPLE else self.obstacles
        if add:
            self.grid[pos] = grid_type
            target_set.add(pos)
        elif self.grid[pos] is grid_type:
            self.grid[pos] = GridType.EMPTY
            target_set.discard(pos)

    def _min_dist_to_apple(self) -> float:
        if not self.apples or not self.snake:
            return 0.0
        r, c = self.snake.body[0]
        return min(abs(r - ar) + abs(c - ac) for ar, ac in self.apples)  # L1 distance

    def _get_observation(self) -> npt.NDArray[np.uint8]:
        if self.obs_type == ObserveType.FULL_GRID:
            return obs.observe_full_grid(self)
        else:  # Default to vec_11 as self.obs_type default value is ObserveType.VEC_11
            return obs.observe_vec11(self)

    def _get_info(self) -> dict:
        snake = self.snake
        info = {
            "death_reason": snake.last_death_reason if snake else None,
            "apples_eaten": snake.apples_eaten if snake else 0,
        }
        if self.snapshot_engine_state:
            info["engine_state"] = self.clone()
        return info

    def _mark_dead(self, reason: DeathReason):
        if not self.snake:
            return
        self.snake.alive = False
        self.snake.last_death_reason = reason
        for br, bc in self.snake.body:
            if self.grid[br, bc] is GridType.SNAKE:
                self.grid[br, bc] = GridType.EMPTY

    def step(self, action: Action) -> Tuple[npt.NDArray[np.uint8], SupportsFloat, bool, bool, dict]:
        snake = self.snake
        if not snake or not snake.alive:
            return self._get_observation(), 0.0, True, False, self._get_info()

        grid = self.grid
        apples = self.apples
        obstacles = self.obstacles
        rewards = self.rewards

        snake.dir = Direction((snake.dir + action - 1) % 4)
        head_r, head_c = snake.body[0]
        dy, dx = DIR_OFFSETS[snake.dir]
        new_head = (head_r + dy, head_c + dx)

        dist_before = self._min_dist_to_apple()
        reward = 0.0
        hit_apple = False

        # movement & collision handling
        pr, pc = new_head
        if (not (0 <= pr < self.height and 0 <= pc < self.width)) or grid[new_head] == GridType.OBSTACLE:
            # Snake new head is out of bounds or hits an obstacle
            self._mark_dead(DeathReason.WALL)
            reward += rewards.death_wall
        elif grid[new_head] == GridType.SNAKE and not (new_head == snake.body[-1] and new_head not in apples):
            # Snake collides with itself (note: allow moving into the tail if it's not eating an apple,
            # since the tail will move away in the same step)
            self._mark_dead(DeathReason.SELF)
            reward += rewards.death_self
        else:
            # Valid movement, move snake's head
            snake.body.appendleft(new_head)
            grid[new_head] = GridType.SNAKE
            if new_head in apples:
                hit_apple = True
                apples.remove(new_head)
                if len(apples) < self.num_apples:
                    self._spawn_items(1, apples, GridType.APPLE)
                reward += rewards.eats_apple
            else:
                tr, tc = snake.body.pop()
                grid[tr, tc] = GridType.EMPTY
                reward += rewards.penalty_step

            # If snake fills the entire grid (except obstacles), it's a win condition
            if len(snake.body) == self.width * self.height - len(obstacles):
                self._mark_dead(DeathReason.COMPLETE)
                reward += rewards.complete

        # reward shaping & loop penalty
        pure_movement = not hit_apple and snake.alive
        if pure_movement and apples:
            closer = self._min_dist_to_apple() < dist_before
            reward += rewards.shaping_closer if closer else rewards.shaping_further
            if new_head in snake.recent_positions:
                reward += rewards.penalty_loop
            snake.recent_positions.append(new_head)
        elif hit_apple or not pure_movement:
            snake.recent_positions.clear()

        snake.total_rewards += reward
        snake.apples_eaten += 1 if hit_apple else 0

        truncated = self.step_count >= self.max_steps
        if truncated:
            self._mark_dead(DeathReason.TRUNCATED)
        else:
            self.step_count += 1

        return self._get_observation(), reward, not snake.alive, truncated, self._get_info()

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
        new_env.apples, new_env.obstacles = set(self.apples), set(self.obstacles)  # Shallow copy
        new_env.grid = self.grid.copy()
        new_env.step_count = self.step_count
        return new_env

    def render(self):
        if self.render_mode == RenderMode.HUMAN and self.ui is not None:
            if not self.ui.handle_events(self):
                return sys.exit(0)
            while self.ui.paused:
                if not self.ui.handle_events(self):
                    return sys.exit(0)
                self.ui.render(self, self.snake.total_rewards if self.snake else None)
            self.ui.render(self, self.snake.total_rewards if self.snake else None)

    def close(self):
        if self.ui:
            self.ui.quit()
            self.ui = None
