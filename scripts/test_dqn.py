from agents.dqn.dqn_vec11 import DQNVec11Agent
from core.env.types import RenderMode, RenderOptions
from core.utils import evaluate_agent


def run_test():
    agent = DQNVec11Agent(
        hidden_dim=256,
        learning_rate=1e-3,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        eps_init=1.0,
        eps_final=0.01,
        eps_decay=0.6,
        learning_starts=1000,
        train_freq=1,
        device="auto",
        seed=42,
    )
    agent.load("./artifacts/models/dqn_snake_best.pth")
    # Set to fully greedy policy so it doesn't wander randomly
    agent.eps = 0.0

    evaluate_agent(
        agent,
        num_obstacles=10,
        num_apples=2,
        num_episodes=3,
        render_mode=RenderMode.HUMAN,
        seed=67,
        render_options=RenderOptions(agent_color=(190, 128, 250)),
    )


if __name__ == "__main__":
    run_test()
