from agents.dqn.dqn_grid import DQNGridAgent
from core.env.types import ObserveType, RenderMode, RenderOptions
from core.utils import evaluate_agent


def run_test():
    agent = DQNGridAgent(device="cuda", seed=67, frame_stack=4, grid_shape=(15, 15))
    agent.load("./artifacts/models/dqn_full_grid.pth")
    evaluate_agent(
        agent,
        num_obstacles=10,
        obs_type=ObserveType.FULL_GRID,
        width=15,
        height=15,
        num_apples=2,
        num_episodes=3,
        render_mode=RenderMode.HUMAN,
        seed=67,
        render_options=RenderOptions(agent_color=(190, 128, 250)),
        frame_stack=4,
    )


if __name__ == "__main__":
    run_test()
