import numpy as np
from typing import Dict, Any, Optional
from agents.base import Agent
from core.env.types import Action, Direction


class GreedyAgent(Agent):
    def __init__(self):
        self.training = False

    def _change_dir(self, current_dir: Direction, action: Action) -> Direction:
        return Direction((current_dir.value + action.value - 1) % 4)

    def act(self, state: np.ndarray, info: Optional[Dict[str, Any]] = None) -> Action:
        if len(state) != 11:
            return Action.STRAIGHT

        safe_actions = [Action(i) for i in range(3) if not state[i]]
        if not safe_actions:
            return Action.STRAIGHT

        current_dir = [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN][
            np.argmax(state[3:7])
        ]
        food_goals = [
            (7, Direction.LEFT),
            (8, Direction.RIGHT),
            (9, Direction.UP),
            (10, Direction.DOWN),
        ]

        for bit, tgt_dir in food_goals:
            if state[bit]:
                for a in safe_actions:
                    if self._change_dir(current_dir, a) == tgt_dir:
                        return a

        for bit, tgt_dir in food_goals:
            if state[bit]:
                perps = (
                    (Direction.UP, Direction.DOWN)
                    if tgt_dir in (Direction.LEFT, Direction.RIGHT)
                    else (Direction.LEFT, Direction.RIGHT)
                )
                for a in safe_actions:
                    if self._change_dir(current_dir, a) in perps:
                        return a

        return Action.STRAIGHT if Action.STRAIGHT in safe_actions else safe_actions[0]
