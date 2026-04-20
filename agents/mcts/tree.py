from typing import Optional

import numpy as np

from core.env.core import SnakeEnv
from core.env.types import Action


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


def select_leaf(node: TreeNode) -> TreeNode:
    while node.children:
        best_score = -float("inf")
        best_child = None
        for child in node.children.values():
            score = uct_score(child, node.visits)
            if score > best_score:
                best_score = score
                best_child = child
        node = best_child
    return node


def expand(node: TreeNode) -> TreeNode:
    if not node.children:
        node.children = {
            Action(i): TreeNode(node.state, node, {}, 0, 0.0) for i in range(3)
        }
    return node.children[Action(np.argmax(node.state[3:7]))]


def simulate(node: TreeNode) -> float:
    env = SnakeEnv()
    env.reset()
    while not env.done:
        action = Action(np.argmax(node.state[3:7]))
        env.step(action)
    return env.snake.total_rewards


def backprop(node: TreeNode, result: float) -> None:
    while node:
        node.visits += 1
        node.value += result
        node = node.parent


def best_action(node: TreeNode) -> Action:
    return Action(
        max(node.children.values(), key=lambda child: child.value / child.visits)
    )
