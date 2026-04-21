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
from core.env.core import SnakeEnv
from core.env.types import Action


class MCTSAgent(Agent):
    def __init__(self, simulations: int = 64):
        self.training = False
        self.simulations = simulations

    def act(self, state: np.ndarray, info: Optional[Dict[str, Any]] = None) -> Action:
        if len(state) != 11:
            return Action.STRAIGHT

        safe_actions = [Action(i) for i in range(3) if not state[i]]
        if not safe_actions:
            return Action.STRAIGHT

        engine = info.get("engine_state") if info else None
        if not isinstance(engine, SnakeEnv):
            return Action.STRAIGHT

        root = TreeNode(engine.clone(), None, {}, 0, 0.0)
        for _ in range(self.simulations):
            leaf = select_leaf(root)
            child = expand(leaf)
            result = simulate(child)
            backprop(child, result)

        chosen = best_action(root)
        return chosen if chosen in safe_actions else safe_actions[0]
