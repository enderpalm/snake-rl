import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Any, Optional
from collections import deque

import core.env.observations as obs
from core.env.types import (
    Direction,
    GridType,
    ObserveType,
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


class SnakeEnv(gym.Env):
    REWARDS: RewardOptions = RewardOptions()

    def __init__(
        self,
        width: int = 15,
        height: int = 15,
        obs_type: ObserveType | int = ObserveType.VEC_11,
        num_apples: int = 1,
        num_obstacles: int = 0,
        seed: int | None = None,
        render_mode: RenderMode | None = None,
        render_options: RenderOptions | None = None,
        reward_options: RewardOptions | dict | None = None,
        max_steps: int = 2000,
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
        self.np_random = np.random.default_rng(seed)

        self.snake: Snake | None = None
        self.apples: set[Tuple[int, int]] = set()
        self.obstacles: set[Tuple[int, int]] = set()
        self.grid = np.full((self.height, self.width), fill_value=GridType.EMPTY, dtype=np.int8)

        self.step_count = 0
        self.done = False
        self.snapshot_engine_state = snapshot_engine_state

        if reward_options:
            # accept a dict-like or a dataclass instance
            if hasattr(reward_options, "items"):
                items = reward_options.items()
            else:
                items = vars(reward_options).items()
            for k, v in items:
                setattr(type(self).REWARDS, k, v)

        self.ui = None
        if self.render_mode == RenderMode.HUMAN:
            from core.pygame_ui import PygameUI

            opts = render_options or {}
            ui_kwargs: dict = {}
            if "cell_size" in opts:
                ui_kwargs["cell_size"] = opts["cell_size"]
            if "render_fps" in opts:
                ui_kwargs["fps"] = opts["render_fps"]
            if "agent_color" in opts:
                ui_kwargs["agent_color"] = opts["agent_color"]

            self.ui = PygameUI(**ui_kwargs)
            self.ui.init_screen(width, height)

        # Not explicity used, but good to have for gym.env compliance
        self.action_space = spaces.Discrete(3)
        self.observation_space = (
            spaces.MultiBinary(11)
            if self.obs_type == ObserveType.VEC_11
            else spaces.Box(low=0, high=255, shape=(3, height, width), dtype=np.uint8)
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, dict]:
        super().reset(seed=seed, options=options)
        if options:
            self.num_apples = options.get("num_apples", self.num_apples)
            self.num_obstacles = options.get("num_obstacles", self.num_obstacles)

        self.step_count = 0
        self.done = False

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
        if self.grid[pos] in (GridType.HEAD, GridType.BODY):
            return

        target_set = self.apples if grid_type == GridType.APPLE else self.obstacles
        if add:
            self.grid[pos] = grid_type
            target_set.add(pos)
        elif self.grid[pos] == grid_type:
            self.grid[pos] = GridType.EMPTY
            target_set.discard(pos)

    def _min_dist_to_apple(self) -> float:
        if not self.apples or not self.snake:
            return 0.0
        r, c = self.snake.body[0]
        return min(abs(r - ar) + abs(c - ac) for ar, ac in self.apples)  # L1 distance

    def _get_observation(self) -> Any:
        if self.obs_type == ObserveType.FULL:
            return obs.observe_full_grid(self)

        elif self.obs_type == ObserveType.VEC_11:
            return obs.observe_vec11(self)

    def _get_info(self) -> dict:
        snake = self.snake
        info = {
            "death_reason": snake.last_death_reason if snake else None,
            "apples_eaten": snake.apples_eaten if snake else 0,
        }
        if getattr(self, "snapshot_engine_state", False):
            info["engine_state"] = self.clone()
        return info

    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]:
        snake = self.snake
        if self.done or not snake or not snake.alive:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # update direction and remember distance to nearest apple
        snake.dir = Direction((snake.dir + action - 1) % 4)
        dist_before = self._min_dist_to_apple()

        # bookkeeping
        self.step_count += 1
        reward = 0.0
        hit_apple = False

        grid = self.grid
        apples = self.apples
        obstacles = self.obstacles
        r_cfg = self.REWARDS

        head_r, head_c = snake.body[0]
        dy, dx = DIR_OFFSETS[snake.dir]
        new_head = (head_r + dy, head_c + dx)

        def _die(reason: DeathReason, penalty: float) -> float:
            snake.alive = False
            snake.last_death_reason = reason
            for br, bc in snake.body:
                if grid[br, bc] in (GridType.HEAD, GridType.BODY):
                    grid[br, bc] = GridType.EMPTY
            return penalty

        # collision / movement
        nr, nc = new_head
        if not (0 <= nr < self.height and 0 <= nc < self.width):
            reward += _die(DeathReason.WALL, r_cfg.reward_death_wall)
        else:
            target = grid[new_head]
            if target == GridType.OBSTACLE:
                reward += _die(DeathReason.WALL, r_cfg.reward_death_wall)
            elif target in (GridType.BODY, GridType.HEAD) and not (
                new_head == snake.body[-1] and new_head not in apples
            ):
                reward += _die(DeathReason.SELF, r_cfg.reward_death_self)
            else:
                snake.body.appendleft(new_head)
                if len(snake.body) > 1:
                    grid[snake.body[1]] = GridType.BODY
                grid[new_head] = GridType.HEAD

                if new_head in apples:
                    hit_apple = True
                    apples.remove(new_head)
                    if len(apples) < self.num_apples:
                        self._spawn_items(1, apples, GridType.APPLE)
                    reward += r_cfg.reward_apple
                else:
                    tr, tc = snake.body.pop()
                    grid[tr, tc] = GridType.EMPTY
                    reward += r_cfg.reward_step

                if len(snake.body) == self.width * self.height - len(obstacles):
                    reward += _die(DeathReason.COMPLETE, r_cfg.reward_complete)

        pure_movement = not hit_apple and snake.alive
        if pure_movement and apples:
            closer = self._min_dist_to_apple() < dist_before
            reward += r_cfg.reward_shaping_closer if closer else r_cfg.reward_shaping_further
            if new_head in snake.recent_positions:
                reward += r_cfg.reward_loop_penalty
            snake.recent_positions.append(new_head)
        else:
            if hit_apple or not pure_movement:
                snake.recent_positions.clear()

        snake.total_rewards += reward
        if hit_apple:
            snake.apples_eaten += 1

        truncated = self.step_count >= self.max_steps
        if truncated and snake.alive:
            snake.last_death_reason = DeathReason.TRUNCATED

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
        new_env.apples, new_env.obstacles = set(self.apples), set(self.obstacles)
        new_env.grid = self.grid.copy()
        new_env.step_count = self.step_count
        new_env.done = self.done
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
