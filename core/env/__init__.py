from gymnasium.envs.registration import register

register(
    id="Snake-v0",
    entry_point="core.env.core:SnakeEnv",
)
