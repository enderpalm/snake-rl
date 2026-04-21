from typing import Optional

import numpy as np

import core.env.observations as obs_mod
from core.env.core import SnakeEnv
from core.env.types import Action, Direction


def _greedy_action_from_obs(obs: np.ndarray) -> Action:
    """Rollout policy: prefer moving toward the apple among locally safe moves (GreedyAgent logic)."""
    safe_actions = [Action(i) for i in range(3) if not obs[i]]
    if not safe_actions:
        return Action.STRAIGHT
    current_dir = [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN][
        int(np.argmax(obs[3:7]))
    ]

    def heading_after(a: Action) -> Direction:
        return Direction((current_dir.value + a.value - 1) % 4)

    food_goals = [
        (7, Direction.LEFT),
        (8, Direction.RIGHT),
        (9, Direction.UP),
        (10, Direction.DOWN),
    ]
    for bit, tgt_dir in food_goals:
        if obs[bit]:
            for a in safe_actions:
                if heading_after(a) == tgt_dir:
                    return a
    return Action.STRAIGHT if Action.STRAIGHT in safe_actions else safe_actions[0]


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
        candidates: list[TreeNode] = []
        for child in node.children.values():
            score = uct_score(child, node.visits)
            if score > best_score:
                best_score = score
                candidates = [child]
            elif score == best_score:
                candidates.append(child)
        node = node.env.np_random.choice(candidates)
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


def simulate(node: TreeNode, max_rollout_steps: int = 80) -> float:
    """Greedy rollout, capped for speed (full-episode rollouts are far too slow per simulation)."""
    env = node.env.clone()
    if _is_terminal(env):
        return 0.0
    total = 0.0
    for _ in range(max_rollout_steps):
        obs = obs_mod.observe_vec11(env)
        action = int(_greedy_action_from_obs(obs))
        _, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if terminated or truncated:
            break
    return total


def backprop(node: TreeNode, result: float) -> None:
    while node:
        node.visits += 1
        node.value += result
        node = node.parent


def best_action(node: TreeNode) -> Action:
    if not node.children:
        return Action.STRAIGHT
    # Avoid Python max() tie-breaking on dict order (often biased toward LEFT).
    return max(
        node.children.items(),
        key=lambda item: (
            item[1].visits,
            item[1].value / item[1].visits if item[1].visits else -1e30,
            item[0] == Action.STRAIGHT,
        ),
    )[0]
