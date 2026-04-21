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
        step_reward: float = 0.0,
    ):
        self.env = env
        self.parent = parent
        self.children = children
        self.visits = visits
        self.value = value
        # Reward incurred transitioning from parent -> this node (0.0 at root).
        self.step_reward = step_reward


def uct_score(node: TreeNode, parent_visits: int, c: float = 1.41) -> float:
    """Q(parent, a) = r(parent -> node) + gamma * V(node), plus UCB exploration bonus."""
    if node.visits == 0:
        return float("inf")
    q = node.step_reward + GAMMA * (node.value / node.visits)
    if parent_visits <= 0:
        return q
    return q + c * np.sqrt(np.log(parent_visits) / node.visits)


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
    # Expand all 3 actions and let SnakeEnv.step decide legality. vec11 marks
    # tail-chase as unsafe but snake usually survives it (tail moves out), and
    # pruning those moves traps MCTS in loops near the apple.
    for a in (Action(0), Action(1), Action(2)):
        e = node.env.clone()
        _, r, _, _, _ = e.step(int(a))
        node.children[a] = TreeNode(e, node, {}, 0, 0.0, step_reward=float(r))
    return node.env.np_random.choice(list(node.children.values()))


# Discount factor for returns. With long greedy rollouts, any starting state looks
# roughly equally good (rollout always eats apples eventually), which swamps the
# small per-step reward signal and lets MCTS loop near the apple. Discounting
# pushes far-future rollout gains down so that near-term r(parent -> child)
# dominates Q, which is exactly what we want.
GAMMA: float = 0.9


def simulate(node: TreeNode, max_rollout_steps: int = 80) -> float:
    """Greedy rollout, capped for speed, discounted so immediate reward dominates."""
    env = node.env.clone()
    if _is_terminal(env):
        return 0.0
    total = 0.0
    discount = 1.0
    for _ in range(max_rollout_steps):
        obs = obs_mod.observe_vec11(env)
        action = int(_greedy_action_from_obs(obs))
        _, reward, terminated, truncated, _ = env.step(action)
        total += discount * reward
        discount *= GAMMA
        if terminated or truncated:
            break
    return total


def backprop(node: TreeNode, result: float) -> None:
    """Credit each ancestor with the discounted return from its state onwards.

    Return from node N = step_reward(child_on_path) + gamma * return(child_on_path).
    Ascending the tree, we fold in this relation step by step.
    """
    g = result
    while node:
        node.visits += 1
        node.value += g
        # Move up: this node's return becomes its parent's next-step return, so the
        # parent sees step_reward(node) + gamma * g.
        g = node.step_reward + GAMMA * g
        node = node.parent


def best_action(node: TreeNode) -> Action:
    if not node.children:
        return Action.STRAIGHT
    # Prefer most visits, then best mean return; break ties randomly (no STRAIGHT bias).
    best_key: tuple[int, float] | None = None
    candidates: list[Action] = []
    for action, child in node.children.items():
        q = (child.step_reward + GAMMA * child.value / child.visits) if child.visits else -1e30
        key = (child.visits, q)
        if best_key is None or key > best_key:
            best_key = key
            candidates = [action]
        elif key == best_key:
            candidates.append(action)
    rng = next(iter(node.children.values())).env.np_random
    return Action(int(rng.choice(candidates)))
