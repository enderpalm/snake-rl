from core.env.enums import RenderMode
from agents.greedy import GreedyAgent
from core.utils import evaluate_agent

def run_test():
    agent = GreedyAgent()
    evaluate_agent(
        agent,
        num_obstacles=10,
        num_apples=2,
        num_episodes=3,
        render_mode=RenderMode.HUMAN,
        seed=67
    )

if __name__ == "__main__":
    run_test()
