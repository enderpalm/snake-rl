from agents.actor_critic import ActorCriticAgent
from core.env.core import RewardOptions
from core.env.types import RenderMode, RenderOptions
from core.utils import evaluate_agent


def run_test():
    ac_agent = ActorCriticAgent(state_dim=11, action_dim=3, seed=67)

    ac_agent.load("./artifacts/models/actor_critic_snake_best.pth")

    evaluate_agent(
        ac_agent,
        num_episodes=3,
        num_apples=(1, 3),
        num_obstacles=(5, 12),
        seed=67,
        num_envs=16,
        max_steps=10000,
        render_mode=RenderMode.HUMAN,
        render_options=RenderOptions(agent_color=(255, 197, 211)),
        reward_options=RewardOptions(
            eats_apple=24.0,
            penalty_step=-0.01,
            penalty_loop=-0.1,
            death_wall=-20.0,
            death_self=-20.0,
            shaping_closer=0.1,
            shaping_further=-0.1,
            complete=100.0,
        ),
    )


if __name__ == "__main__":
    run_test()
