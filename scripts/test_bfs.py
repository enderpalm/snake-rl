from rich.markup import render

from agents.bfs import BFSAgent
from core.env.types import RenderMode, RenderOptions
from core.utils import evaluate_agent


def run_test():
    agent = BFSAgent()
    evaluate_agent(
        agent,
        num_obstacles=10,
        num_apples=2,
        num_episodes=1000,
        render_mode=None,
        seed=42,
        snapshot_engine_state=True,
        render_options=RenderOptions(agent_color=(190, 128, 250))
    )


if __name__ == "__main__":
    run_test()
