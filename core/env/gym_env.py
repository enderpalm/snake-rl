import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple, TypedDict

from core.env.core import SnakeEngine
from core.env.enums import Direction, GridType, ObsType, RenderMode, DeathReason
from core.pygame_ui import PygameUI


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

class SnakeEnv(gym.Env):

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
    ):

        super().__init__()
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.obs_type = obs_type

        from collections import deque

        self.recent_positions: deque = deque(maxlen=20)

        self.engine = SnakeEngine(
            width=width,
            height=height,
            num_apples=num_apples,
            num_obstacles=num_obstacles,
            seed=seed,
        )

        if reward_options:
            for k, v in reward_options.items():
                setattr(self.engine, k.upper(), v)

        self.ui = None
        if self.render_mode == RenderMode.HUMAN:
            opts = render_options or {}
            self.ui = PygameUI(
                cell_size=opts.get("cell_size", 30),
                fps=opts.get("render_fps", 15),
                agent_color=opts.get("agent_color"),
            )
            self.ui.init_screen(width, height)

        # Action space: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT (Map to Direction enum)
        self.action_space = spaces.Discrete(4)

        # Observation space
        if self.obs_type == ObsType.VECTOR_11:
            self.observation_space = spaces.MultiBinary(11)
        elif self.obs_type == ObsType.FULL_GRID:  # very crude for now :(
            # Image output: shape (height, width, channels) or (channels, h, w) for PyTorch.
            # Using standard (Channels, Height, Width) for CNNs: C=3 (Snake, Foods, Obstacles)
            self.observation_space = spaces.Box(low=0, high=255, shape=(3, height, width), dtype=np.uint8)
        elif self.obs_type == ObsType.ALL:
            self.observation_space = spaces.Dict(
                {
                    "vector": spaces.MultiBinary(11),
                    "grid": spaces.Box(low=0, high=255, shape=(3, height, width), dtype=np.uint8),
                }
            )
        else:
            raise ValueError(f"Unknown observation type: {obs_type}")

        self.last_death_reason: Optional[DeathReason] = None
        self.total_rewards = 0.0
        self.apples_eaten = 0

    def _get_obs(self) -> Any:
        vec, grid = self._extract_11_dim_vector(self.engine), self._extract_full_grid(self.engine)
        if self.obs_type == ObsType.VECTOR_11:
            return vec
        if self.obs_type == ObsType.FULL_GRID:
            return grid
        if self.obs_type == ObsType.ALL:
            return {"vector": vec, "grid": grid}

    def _extract_11_dim_vector(self, engine) -> np.ndarray:
        snake = engine.snake
        r, c = snake.body[0]
        dy, dx = snake.dir.y, snake.dir.x

        def is_collision(pr, pc):
            return not (0 <= pr < engine.height and 0 <= pc < engine.width) or engine.grid[pr, pc] in (
                GridType.OBSTACLE,
                GridType.BODY,
                GridType.HEAD,
            )

        apple = next(iter(engine.apples)) if engine.apples else (r, c)

        return np.array(
            [
                is_collision(r + dy, c + dx),
                is_collision(r + dx, c - dy),
                is_collision(r - dx, c + dy),
                *(snake.dir == d for d in (Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN)),
                apple[1] < c,
                apple[1] > c,
                apple[0] < r,
                apple[0] > r,
            ],
            dtype=np.int8,
        )

    # still very experimental
    def _extract_full_grid(self, engine) -> np.ndarray:
        state = np.zeros((3, engine.height, engine.width), dtype=np.uint8)
        state[0][engine.grid == GridType.HEAD] = 255
        state[0][engine.grid == GridType.BODY] = 128
        state[1][engine.grid == GridType.APPLE] = 255
        state[2][engine.grid == GridType.OBSTACLE] = 255
        return state

    def _get_info(self) -> Dict[str, Any]:
        return {
            "death_reason": self.last_death_reason,
            "apples_eaten": self.apples_eaten,
            "engine_state": self.engine.clone(),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, dict]:
        super().reset(seed=seed, options=options)

        if options:
            self.engine.num_apples = options.get("num_apples", self.engine.num_apples)
            self.engine.num_obstacles = options.get("num_obstacles", self.engine.num_obstacles)

        self.engine.reset()
        self.recent_positions.clear()

        self.last_death_reason = None
        self.total_rewards = 0.0
        self.apples_eaten = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]:
        dist_before = 0
        if len(self.engine.apples) > 0:
            head = self.engine.snake.body[0]
            dist_before = min(abs(head[0] - r) + abs(head[1] - c) for r, c in self.engine.apples)

        res = self.engine.step(action)
        reward, death = res["rewards"], res["deaths"]

        terminated = not self.engine.snake.alive or self.engine.done
        self.last_death_reason = death if not self.engine.snake.alive else self.last_death_reason

        # Apply reward shaping natively
        if reward == 0 and not terminated and len(self.engine.apples) > 0:
            head = self.engine.snake.body[0]
            dist_after = min(abs(head[0] - r) + abs(head[1] - c) for r, c in self.engine.apples)
            reward += (
                self.engine.REWARD_SHAPING_CLOSER if dist_after < dist_before else self.engine.REWARD_SHAPING_FURTHER
            )

            # Loop penalty
            if head in self.recent_positions:
                reward += getattr(self.engine, "REWARD_LOOP_PENALTY", -0.25)
            self.recent_positions.append(head)
        elif reward > 2.0:  # apple eaten
            self.recent_positions.clear()

        self.total_rewards += reward
        self.apples_eaten += reward > 2.0

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _close_ui_and_exit(self):
        print("UI closed by user.")
        self.close()
        sys.exit(0)

    def render(self):
        if self.render_mode == RenderMode.HUMAN and self.ui is not None:
            if not self.ui.handle_events(self.engine):
                self._close_ui_and_exit()

            while self.ui.paused:
                if not self.ui.handle_events(self.engine):
                    self._close_ui_and_exit()
                self.ui.render(self.engine, self.total_rewards)

            self.ui.render(self.engine, self.total_rewards)

    def close(self):
        if self.ui is not None:
            self.ui.quit()
            self.ui = None
