"""Shortest-path action on the real grid (needs engine_state snapshot)."""

from collections import deque

from core.env.core import SnakeEnv
from core.env.types import Action, Direction, GridType


def _delta_to_direction(dr: int, dc: int) -> Direction:
    if (dr, dc) == (-1, 0):
        return Direction.UP
    if (dr, dc) == (1, 0):
        return Direction.DOWN
    if (dr, dc) == (0, -1):
        return Direction.LEFT
    if (dr, dc) == (0, 1):
        return Direction.RIGHT
    raise ValueError(f"not a unit step: {(dr, dc)}")


def _direction_to_action(current: Direction, want: Direction) -> Action:
    """Relative {left, straight, right} that matches absolute heading `want` after one step."""
    for a in Action:
        new_dir = (current.value + a.value - 1) % 4
        if new_dir == want.value:
            return a
    best: Action = Action.LEFT
    best_dist = 4
    for a in Action:
        new_dir = (current.value + a.value - 1) % 4
        diff = (want.value - new_dir) % 4
        dist = min(diff, 4 - diff)
        if dist < best_dist:
            best_dist = dist
            best = a
    return best


def _cell_blocked(
    env: SnakeEnv,
    r: int,
    c: int,
    head: tuple[int, int],
    tail: tuple[int, int],
    body_set: set[tuple[int, int]],
) -> bool:
    if not (0 <= r < env.height and 0 <= c < env.width):
        return True
    if env.grid[r, c] == GridType.OBSTACLE:
        return True
    if (r, c) == head:
        return False
    if (r, c) == tail:
        return False
    if (r, c) in body_set:
        return True
    return False


def bfs_first_action_to_apple(env: SnakeEnv) -> Action | None:
    """First relative action on a shortest grid path to any apple, or None if unreachable."""
    if not env.snake or not env.apples:
        return None

    head = (env.snake.body[0][0], env.snake.body[0][1])
    body = list(env.snake.body)
    tail = (body[-1][0], body[-1][1])
    body_set = {(p[0], p[1]) for p in body}
    apples = env.apples

    q: deque[tuple[int, int, Action | None]] = deque([(head[0], head[1], None)])
    visited = {head}

    while q:
        r, c, first_action = q.popleft()
        if first_action is not None and (r, c) in apples:
            return first_action

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) in visited:
                continue
            if _cell_blocked(env, nr, nc, head, tail, body_set):
                continue
            visited.add((nr, nc))
            fa = first_action
            if fa is None:
                tgt = _delta_to_direction(dr, dc)
                fa = _direction_to_action(env.snake.dir, tgt)
            q.append((nr, nc, fa))

    return None
