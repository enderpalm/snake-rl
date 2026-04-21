import logging
from typing import Any, Dict, Optional

import numpy as np

from agents.base import Agent
from agents.bfs.path import bfs_first_action_to_apple
from agents.greedy import GreedyAgent
from core.env.core import SnakeEnv
from core.env.types import Action

logger = logging.getLogger(__name__)


class BFSAgent(Agent):
    """Grid BFS toward an apple when `engine_state` is in info; greedy vec11 fallback otherwise."""

    def __init__(self, seed: int | None = None):
        super().__init__(seed=seed)
        self.training = False
        self._greedy = GreedyAgent()
        self._decision_counts = {"bfs": 0, "greedy": 0, "other": 0}

    def reset_decision_stats(self) -> None:
        for k in self._decision_counts:
            self._decision_counts[k] = 0

    def log_decision_stats(self, reset: bool = False) -> None:
        c = self._decision_counts
        total = sum(c.values())
        if total == 0:
            msg = "BFSAgent decisions: no act() calls recorded"
            print(msg)
            logger.info(msg)
            return

        def pct(k: str) -> float:
            return 100.0 * c[k] / total

        msg = (
            f"BFSAgent decisions (n={total}): "
            f"BFS {pct('bfs'):.1f}% | greedy {pct('greedy'):.1f}% | other {pct('other'):.1f}%"
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

        bfs_action = bfs_first_action_to_apple(engine)
        if bfs_action is not None:
            self._decision_counts["bfs"] += 1
            return bfs_action

        self._decision_counts["greedy"] += 1
        return self._greedy.act(state, info)
