import logging
from typing import Any, Dict, Optional

import numpy as np

from agents.base import Agent
from agents.greedy import GreedyAgent
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

logger = logging.getLogger(__name__)


class MCTSAgent(Agent):
    """Monte Carlo tree search on vec11 + engine snapshots; greedy fallback when MCTS is skipped."""

    def __init__(
        self,
        simulations: int = 48,
        max_rollout_steps: int = 80,
        seed: int | None = None,
        tree_debug_window: bool = False,
    ):
        super().__init__(seed=seed)
        self.training = False
        self.simulations = simulations
        self.max_rollout_steps = max_rollout_steps
        self.tree_debug_window = tree_debug_window
        self.last_mcts_panel: list[str] | None = None
        self._greedy = GreedyAgent()
        self._decision_counts = {"mcts": 0, "greedy": 0, "other": 0}

    def reset_decision_stats(self) -> None:
        for k in self._decision_counts:
            self._decision_counts[k] = 0

    def log_decision_stats(self, reset: bool = False) -> None:
        c = self._decision_counts
        total = sum(c.values())
        if total == 0:
            msg = "MCTSAgent decisions: no act() calls recorded"
            print(msg)
            logger.info(msg)
            return

        def pct(k: str) -> float:
            return 100.0 * c[k] / total

        msg = (
            f"MCTSAgent decisions (n={total}): "
            f"MCTS {pct('mcts'):.1f}% | greedy {pct('greedy'):.1f}% | other {pct('other'):.1f}%"
        )
        print(msg)
        logger.info(msg)
        if reset:
            self.reset_decision_stats()

    def act(self, state: np.ndarray, info: Optional[Dict[str, Any]] = None) -> Action:
        self.last_mcts_panel = None
        if len(state) != 11:
            self._decision_counts["other"] += 1
            return Action.STRAIGHT

        safe_actions = [Action(i) for i in range(3) if not state[i]]
        if not safe_actions:
            self._decision_counts["other"] += 1
            return Action.STRAIGHT

        engine = info.get("engine_state") if info else None
        if not isinstance(engine, SnakeEnv) or self.simulations <= 0:
            self._decision_counts["greedy"] += 1
            return self._greedy.act(state, info)

        root = TreeNode(engine.clone(), None, {}, 0, 0.0)
        for _ in range(self.simulations):
            leaf = select_leaf(root)
            child = expand(leaf)
            result = simulate(child, self.max_rollout_steps)
            backprop(child, result)

        chosen = best_action(root)
        self._decision_counts["mcts"] += 1
        if self.tree_debug_window:
            from agents.mcts.tree_view import mcts_panel_lines

            self.last_mcts_panel = mcts_panel_lines(root, chosen)
        if chosen in safe_actions:
            return chosen
        if Action.STRAIGHT in safe_actions:
            return Action.STRAIGHT

        # random choice in safe_actions
        return safe_actions[int(self.rng.integers(len(safe_actions)))]
