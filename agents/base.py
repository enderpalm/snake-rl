from abc import ABC, abstractmethod
import os
from typing import Optional
import numpy as np
from core.env.types import Action


class Agent(ABC):
    """
    Abstract base class for all Snake RL agents.
    """

    MODEL_PATH = "../artifacts/models/"

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def act(self, state: np.ndarray, info: Optional[dict] = None) -> Action:
        """
        Given the current state (observation), returns the chosen action.
        """
        pass

    # Some Agents (like GreedyAgent) may not need to implement these:

    def train(self, mode: bool = True) -> None:
        """Set agent to training or evaluation mode."""
        pass

    def update(
        self,
        state: np.ndarray,
        action: Action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Hook for PyTorch based custom RL loop updates."""
        pass

    def save(self, path: str) -> None:
        """Save agent model weights to an external path."""
        pass

    def load(self, path: str) -> None:
        """Load agent model weights from an external path."""
        pass

    def _modify_path(self, path: str) -> str:
        path = self.MODEL_PATH + path if not path.startswith(self.MODEL_PATH) else path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
