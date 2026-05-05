from gymnasium.envs.registration import register

register(
    id="Snake-v0",
    entry_point="core.env.core:SnakeEnv",
)

# Make important classes accessible at package level
from core.env.core import SnakeEnv, Snake
from core.env.types import Action, Direction, GridType, ObserveType, RenderMode
