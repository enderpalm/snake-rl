from core.env.types import RenderMode
from agents.greedy import GreedyAgent
from core.utils import evaluate_agent


def run_test():
    agent = GreedyAgent()
    evaluate_agent(
        agent,
        num_obstacles=10,
        num_apples=2,
        num_episodes=1000,
        render_mode=None,
        seed=42,
    )


if __name__ == "__main__":
    run_test()
