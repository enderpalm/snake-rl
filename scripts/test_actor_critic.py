from agents.actor_critic import ActorCriticAgent
from core.env.types import RenderMode
from core.utils import evaluate_agent


def run_test():
    ac_agent = ActorCriticAgent(state_dim=11, action_dim=3, seed=42)

    ac_agent.load("./artifacts/models/actor_critic_snake_best.pth")

    evaluate_agent(
        ac_agent,
        num_obstacles=10,
        num_apples=2,
        num_episodes=1000,
        render_mode=None,
        seed=42,
        snapshot_engine_state=True,
    )

if __name__ == "__main__":
    run_test()
