import time
from core.env.gym_env import SnakeEnv
from core.env.enums import ObsType, RenderMode
from agents.greedy import GreedyAgent

def run_test():
    env = SnakeEnv(
        width=15,
        height=15,
        num_apples=2,
        num_obstacles=10,
        obs_type=ObsType.VECTOR_11,
        seed=67,
        render_mode=RenderMode.HUMAN,
        render_options={"cell_size": 40, "render_fps": 15},
    )

    # Initialize model
    agent = GreedyAgent()

    for ep in range(3):
        obs, info = env.reset()
        done = False
        step_count = 0

        while not done:
            action = agent.act(obs)
            next_obs, _, terminated, truncated, info = env.step(action)

            # Draw game state
            env.render()

            obs = next_obs
            done = terminated or truncated
            step_count += 1
            time.sleep(0.05)

        print(
            f"Episode {ep + 1} finished after {step_count} steps. Reward: {env.total_rewards:.2f}. Death: {info.get('death_reason').value if info and info.get('death_reason') else 'None'}"
        )
    env.close()


if __name__ == "__main__":
    run_test()
