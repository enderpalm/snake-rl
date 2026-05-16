from agents.dqn.dqn_grid import DQNGridAgent
from core.env.types import ObserveType, RenderMode, RenderOptions, RewardOptions
from core.utils import evaluate_agent


def run_test():
    agent = DQNGridAgent(device="cuda", seed=67, frame_stack=2, grid_shape=(20, 20), in_channels=4)
    agent.load("./artifacts/models/stage_3_10_obstacles.pth")
    evaluate_agent(
        agent,
        num_obstacles=10,
        obs_type=ObserveType.FULL_GRID,
        num_apples=2,
        num_episodes=3,
        render_mode=RenderMode.HUMAN,
        seed=42,
        render_options=RenderOptions(agent_color=(190, 128, 250)),
        frame_stack=2,
        # reward_options=RewardOptions(
        #         eats_apple=15.0,
        #         complete=100.0,
        #         penalty_step=-0.0005,
        #         penalty_loop=-3.0,
        #         death_wall=-8.0,
        #         death_self=-10.0,
        #         shaping_closer=0.5,
        #         shaping_further=-0.2,
        #     ),
    )


if __name__ == "__main__":
    run_test()
