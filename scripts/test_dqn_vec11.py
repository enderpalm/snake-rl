from agents.dqn.dqn_vec11 import DQNVec11Agent
from core.env.types import RenderMode, RenderOptions
from core.utils import evaluate_agent


def run_test():
    agent = DQNVec11Agent(device="cuda", seed=67)
    agent.load("./notebooks/dqn_vec11_dueling.pth")
    evaluate_agent(
        agent,
        num_obstacles=10,
        num_apples=2,
        num_episodes=1000,
        render_mode=None,
        seed=67,
        render_options=RenderOptions(agent_color=(190, 128, 250)),
    )


if __name__ == "__main__":
    run_test()
