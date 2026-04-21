from typing import Any, Dict, Optional

import numpy as np

from agents.base import Agent
from agents.greedy import GreedyAgent
from agents.mcts.path import bfs_first_action_to_apple
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
    """Planning agent: grid BFS toward an apple (default), optional tree search, greedy fallback."""

    def __init__(
        self,
        simulations: int = 48,
        max_rollout_steps: int = 80,
        seed: int | None = None,
        use_grid_bfs: bool = True,
        run_mcts: bool = False,
    ):
        super().__init__(seed=seed)
        self.training = False
        self.simulations = simulations
        self.max_rollout_steps = max_rollout_steps
        self.use_grid_bfs = use_grid_bfs
        self.run_mcts = run_mcts
        self._greedy = GreedyAgent()

    def act(self, state: np.ndarray, info: Optional[Dict[str, Any]] = None) -> Action:
        if len(state) != 11:
            return Action.STRAIGHT

        safe_actions = [Action(i) for i in range(3) if not state[i]]
        if not safe_actions:
            return Action.STRAIGHT

        engine = info.get("engine_state") if info else None
        if not isinstance(engine, SnakeEnv):
            return self._greedy.act(state, info)

        if self.use_grid_bfs:
            bfs_action = bfs_first_action_to_apple(engine)
            # Prefer BFS whenever a path exists: it uses the full grid; vec11 can disagree
            # (e.g. tail will move) and masking BFS with danger bits brought back looping.
            if bfs_action is not None:
                return bfs_action

        if self.run_mcts and self.simulations > 0:
            root = TreeNode(engine.clone(), None, {}, 0, 0.0)
            for _ in range(self.simulations):
                leaf = select_leaf(root)
                child = expand(leaf)
                result = simulate(child, self.max_rollout_steps)
                backprop(child, result)

            chosen = best_action(root)
            if chosen in safe_actions:
                return chosen
            if Action.STRAIGHT in safe_actions:
                return Action.STRAIGHT
            return safe_actions[int(self.rng.integers(len(safe_actions)))]

        return self._greedy.act(state, info)
