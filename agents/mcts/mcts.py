from typing import Any, Dict, Optional

import numpy as np

from agents.base import Agent
from agents.mcts.tree import (
    TreeNode,
    backprop,
    best_action,
    expand,
    select_leaf,
    simulate,
)
from core.env.types import Action, Direction


class MCTSAgent(Agent):
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

        root = TreeNode(state, None, {}, 0, 0.0)
        for _ in range(1000):
            node = select_leaf(root)
            expand(node)
            result = simulate(node)
            backprop(node, result)

        return best_action(root)

        # food_goals = [
        #     (7, Direction.LEFT),
        #     (8, Direction.RIGHT),
        #     (9, Direction.UP),
        #     (10, Direction.DOWN),
        # ]

        # for bit, tgt_dir in food_goals:
        #     if state[bit]:
        #         for a in safe_actions:
        #             if self._change_dir(current_dir, a) == tgt_dir:
        #                 return a

        # return Action.STRAIGHT if Action.STRAIGHT in safe_actions else safe_actions[0]
        # for bit, tgt_dir in food_goals:
        #     if state[bit]:
        #         for a in safe_actions:
        #             if self._change_dir(current_dir, a) == tgt_dir:
        #                 return a

        # return Action.STRAIGHT if Action.STRAIGHT in safe_actions else safe_actions[0]
