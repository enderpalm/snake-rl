from typing import Any, Dict, Optional

import numpy as np

from agents.base import Agent
from core.env.types import Action, Direction


class TreeNode:
    state: np.ndarray
    parent: Optional["TreeNode"]
    children: dict[Action, "TreeNode"]
    visits: int
    value: float


def uct_score(node: TreeNode, parent_visits: int, c: float = 1.41) -> float:
    if node.visits == 0:
        return float("inf")
    return node.value / node.visits + c * np.sqrt(np.log(parent_visits) / node.visits)
