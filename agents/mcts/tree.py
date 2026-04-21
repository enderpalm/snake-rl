from typing import Optional

import numpy as np

import core.env.observations as obs_mod
from core.env.core import SnakeEnv
from core.env.types import Action


def _is_terminal(env: SnakeEnv) -> bool:
    if env.snake is None or not env.snake.alive:
        return True
    return env.step_count >= env.max_steps


class TreeNode:
    def __init__(
        self,
        env: SnakeEnv,
        parent: Optional["TreeNode"],
        children: dict[Action, "TreeNode"],
        visits: int,
        value: float,
    ):
        self.env = env
        self.parent = parent
        self.children = children
        self.visits = visits
        self.value = value


def uct_score(node: TreeNode, parent_visits: int, c: float = 1.41) -> float:
    if node.visits == 0:
        return float("inf")
    if parent_visits <= 0:
        return node.value / node.visits
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
    if _is_terminal(node.env):
        return node
    obs = obs_mod.observe_vec11(node.env)
    legal = [Action(i) for i in range(3) if not obs[i]]
    if not legal:
        legal = [Action(0), Action(1), Action(2)]
    for a in legal:
        e = node.env.clone()
        e.step(int(a))
        node.children[a] = TreeNode(e, node, {}, 0, 0.0)
    return node.env.np_random.choice(list(node.children.values()))


def simulate(node: TreeNode) -> float:
    env = node.env.clone()
    start = env.snake.total_rewards if env.snake else 0.0
    if _is_terminal(env):
        return 0.0
    while True:
        action = int(env.np_random.integers(0, 3))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    end = env.snake.total_rewards if env.snake else 0.0
    return end - start


def backprop(node: TreeNode, result: float) -> None:
    while node:
        node.visits += 1
        node.value += result
        node = node.parent


def best_action(node: TreeNode) -> Action:
    if not node.children:
        return Action.STRAIGHT
    return max(node.children.items(), key=lambda item: item[1].visits)[0]
