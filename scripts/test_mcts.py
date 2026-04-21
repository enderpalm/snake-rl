from agents.mcts.mcts import MCTSAgent
from core.env.types import RenderMode
from core.utils import evaluate_agent


def run_test():
    agent = MCTSAgent()
    evaluate_agent(
        agent,
        num_obstacles=10,
        num_apples=2,
        num_episodes=3,
        render_mode=RenderMode.HUMAN,
        seed=67,
        snapshot_engine_state=True,
    )


if __name__ == "__main__":
    run_test()
