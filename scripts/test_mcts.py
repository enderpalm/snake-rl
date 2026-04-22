import os

from agents.mcts.mcts import MCTSAgent
from core.env.types import RenderMode
from core.utils import evaluate_agent


def run_test():
    agent = MCTSAgent(
        tree_debug_window=True,
        num_workers=max(1, (os.cpu_count() or 2) - 1),
    )
    try:
        evaluate_agent(
            agent,
            num_obstacles=10,
            num_apples=2,
            num_episodes=3,
            render_mode=RenderMode.HUMAN,
            seed=67,
            snapshot_engine_state=True,
        )
    finally:
        agent.close()


if __name__ == "__main__":
    run_test()
