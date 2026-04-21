import logging
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

logger = logging.getLogger(__name__)


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
        self._decision_counts = {"bfs": 0, "mcts": 0, "greedy": 0, "other": 0}

    def reset_decision_stats(self) -> None:
        for k in self._decision_counts:
            self._decision_counts[k] = 0

    def log_decision_stats(self, reset: bool = False) -> None:
        """Print and log shares of BFS vs MCTS vs greedy (and other early exits)."""
        c = self._decision_counts
        total = sum(c.values())
        if total == 0:
            msg = "MCTSAgent decisions: no act() calls recorded"
            print(msg)
            logger.info(msg)
            return

        def pct(k: str) -> float:
            return 100.0 * c[k] / total

        bfs_mcts_denom = c["bfs"] + c["mcts"]
        bfs_vs_mcts = (
            f"BFS {100.0 * c['bfs'] / bfs_mcts_denom:.1f}% vs MCTS {100.0 * c['mcts'] / bfs_mcts_denom:.1f}% (of planner steps)"
            if bfs_mcts_denom > 0
            else "no BFS/MCTS steps"
        )
        msg = (
            f"MCTSAgent decisions (n={total}): "
            f"BFS {pct('bfs'):.1f}% | MCTS {pct('mcts'):.1f}% | greedy {pct('greedy'):.1f}% | other {pct('other'):.1f}% — {bfs_vs_mcts}"
        )
        print(msg)
        logger.info(msg)
        if reset:
            self.reset_decision_stats()

    def act(self, state: np.ndarray, info: Optional[Dict[str, Any]] = None) -> Action:
        if len(state) != 11:
            self._decision_counts["other"] += 1
            return Action.STRAIGHT

        safe_actions = [Action(i) for i in range(3) if not state[i]]
        if not safe_actions:
            self._decision_counts["other"] += 1
            return Action.STRAIGHT

        engine = info.get("engine_state") if info else None
        if not isinstance(engine, SnakeEnv):
            self._decision_counts["greedy"] += 1
            return self._greedy.act(state, info)

        if self.use_grid_bfs:
            bfs_action = bfs_first_action_to_apple(engine)
            # Prefer BFS whenever a path exists: it uses the full grid; vec11 can disagree
            # (e.g. tail will move) and masking BFS with danger bits brought back looping.
            if bfs_action is not None:
                self._decision_counts["bfs"] += 1
                return bfs_action

        if self.run_mcts and self.simulations > 0:
            root = TreeNode(engine.clone(), None, {}, 0, 0.0)
            for _ in range(self.simulations):
                leaf = select_leaf(root)
                child = expand(leaf)
                result = simulate(child, self.max_rollout_steps)
                backprop(child, result)

            chosen = best_action(root)
            self._decision_counts["mcts"] += 1
            if chosen in safe_actions:
                return chosen
            if Action.STRAIGHT in safe_actions:
                return Action.STRAIGHT
            return safe_actions[int(self.rng.integers(len(safe_actions)))]

        self._decision_counts["greedy"] += 1
        return self._greedy.act(state, info)
