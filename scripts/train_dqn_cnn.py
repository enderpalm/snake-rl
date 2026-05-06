import os
from gymnasium.wrappers import FrameStackObservation
from gymnasium.vector import SyncVectorEnv
from agents.dqn.dqn_grid import DQNCnnAgent
from core.env.core import SnakeEnv
from core.env.types import ObserveType


def make_env(width=15, height=15, num_apples=3, num_obstacles=5):
    def _init():
        env = SnakeEnv(
            width=width,
            height=height,
            obs_type=ObserveType.FULL_GRID,
            num_apples=num_apples,
            num_obstacles=num_obstacles,
        )
        # Apply FrameStackObservation (4 frames)
        env = FrameStackObservation(env, stack_size=4)
        return env

    return _init


def train():
    num_envs = 8
    width, height = 15, 15

    # Use SyncVectorEnv to allow multiple agents to collect data simultaneously
    env = SyncVectorEnv([make_env(width, height) for _ in range(num_envs)])

    agent = DQNCnnAgent(
        in_channels=16,  # 4 frames * 4 channels (head, body, apple, obstacles) = 16
        grid_size=width,
        learning_rate=1e-4,
        buffer_size=100000,  # Replay Buffer
        batch_size=64,
        gamma=0.99,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.5,
        train_freq=4,  # Train step interval
        device="cuda",
    )

    print("Starting DQN CNN training with 4 Frame Stacking...")
    os.makedirs("./artifacts/models", exist_ok=True)
    agent.model_file = "./artifacts/models/dqn_cnn_best.pth"
    agent.train(env, total_timesteps=500000)


if __name__ == "__main__":
    train()
