import numpy as np
from typing import Dict, Any, Optional
from agents.base import Agent
from core.env.enums import Direction


class GreedyAgent(Agent):
    """
    Nerfed greedy agent that only uses the 11-dimensional vector state
    to make decisions, matching the "legally blind" Q-learning agent's
    observation space exactly.
    """

    def __init__(self):
        self.training = False

    def act(self, state: np.ndarray, info: Optional[Dict[str, Any]] = None) -> Direction:
        if len(state) != 11:
            return Direction.UP  # Fallback

        # Extract current direction: [LEFT, RIGHT, UP, DOWN] translates to [3, 1, 0, 2]
        current_dir = [3, 1, 0, 2][np.argmax(state[3:7])]

        # Safe directions (state[0..2] = danger straight, right, left)
        possible_dirs = [current_dir, (current_dir + 1) % 4, (current_dir - 1) % 4]
        safe_dirs = [d for i, d in enumerate(possible_dirs) if not state[i]]

        if not safe_dirs:
            return Direction(current_dir)

        # Apple priority mapping: UP(bit 9), RIGHT(bit 8), DOWN(bit 10), LEFT(bit 7)
        for move, bit in zip([0, 1, 2, 3], [9, 8, 10, 7]):
            if state[bit] and move in safe_dirs:
                return Direction(move)

        return Direction(safe_dirs[0])
