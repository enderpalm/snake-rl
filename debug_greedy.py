import numpy as np
from core.env.gym_env import SnakeEnv
from core.env.enums import ObsType
from agents.greedy import GreedyAgent

env = SnakeEnv(width=10, height=10, obs_type=ObsType.VECTOR_11, seed=42)
agent = GreedyAgent()

obs, info = env.reset()
done = False
while not done:
    action = agent.act(obs)
    print("Obs:", obs)
    print("Action chosen:", action)
    obs, reward, term, trunc, info = env.step(action)
    print("New obs collision states:", obs[:3])
    if term or trunc:
        print("Died:", info['death_reason'])
        break
