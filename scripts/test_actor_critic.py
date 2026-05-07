from agents.actor_critic import ActorCriticAgent
from core.env.types import RenderMode
from core.utils import evaluate_agent

def run_test():
    ac_agent = ActorCriticAgent(state_dim=11, action_dim=3, seed=67)

    ac_agent.load("./artifacts/models/actor_critic_snake_best.pth")

    evaluate_agent(
        ac_agent,
        num_episodes=1000,
        num_apples=(1, 3),
        num_obstacles=(5, 12),
        seed=67,
        num_envs=16,
        max_steps=10000,
        render_mode=RenderMode.HUMAN,
    )

if __name__ == "__main__":
    run_test()


