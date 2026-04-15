from agents.q_learning import QLearningAgent
from core.env.types import RenderMode
from core.utils import evaluate_agent


def run_test():
    agent = QLearningAgent(seed=67)
    agent.load("./artifacts/models/q_learning_snake_best.pkl")
    evaluate_agent(
        agent,
        num_obstacles=10,
        num_apples=2,
        num_episodes=3,
        render_mode=RenderMode.HUMAN,
        seed=67,
        render_options={"agent_color": (255, 100, 100)},
    )


if __name__ == "__main__":
    run_test()
