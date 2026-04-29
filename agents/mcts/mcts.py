import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Optional

import numpy as np

from agents.base import Agent
from agents.greedy import GreedyAgent
from agents.mcts.tree import (
    GAMMA,
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


def _mcts_worker(
    env: SnakeEnv, sims: int, max_rollout_steps: int, seed: int
) -> list[tuple[int, int, float]]:
    """Build an independent MCTS tree in a worker process.

    Returns one tuple per root child: ``(action_int, visits, mean_q)`` where
    ``mean_q = step_reward + gamma * V(child)``.
    """
    env.np_random = np.random.default_rng(seed)
    root = TreeNode(env, None, {}, 0, 0.0)
    for _ in range(sims):
        leaf = select_leaf(root)
        child = expand(leaf)
        result = simulate(child, max_rollout_steps)
        backprop(child, result)
    out: list[tuple[int, int, float]] = []
    for a, c in root.children.items():
        mean_q = (c.step_reward + GAMMA * c.value / c.visits) if c.visits else 0.0
        out.append((int(a), int(c.visits), float(mean_q)))
    return out


class MCTSAgent(Agent):
    """Monte Carlo tree search with optional root parallelization across processes.

    When ``num_workers > 0``, simulations are split across an internal
    ``ProcessPoolExecutor`` (root parallelization): each worker builds an
    independent tree and the agent picks the action with the highest aggregated
    visit count, following Chaslot et al. (2008).
    """

    def __init__(
        self,
        simulations: int = 200,
        max_rollout_steps: int = 100,
        seed: int | None = None,
        tree_debug_window: bool = False,
        num_workers: int = 0,
    ):
        super().__init__(seed=seed)
        self.training = False
        self.simulations = simulations
        self.max_rollout_steps = max_rollout_steps
        self.tree_debug_window = tree_debug_window
        self.num_workers = max(0, num_workers)
        self.last_mcts_panel: list[str] | None = None
        self._greedy = GreedyAgent()
        self._decision_counts = {"mcts": 0, "greedy": 0, "other": 0}
        self._pool: ProcessPoolExecutor | None = None

    def _ensure_pool(self) -> None:
        if self.num_workers <= 0 or self._pool is not None:
            return
        # Spawned once; first submit() pays worker startup cost, later calls are cheap.
        self._pool = ProcessPoolExecutor(max_workers=self.num_workers)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

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

        engine = info.get("engine_state") if info else None
        if not isinstance(engine, SnakeEnv) or self.simulations <= 0:
            self._decision_counts["greedy"] += 1
            return self._greedy.act(state, info)

        if self.num_workers <= 0:
            chosen = self._act_sequential(engine)
        else:
            chosen = self._act_parallel(engine)

        self._decision_counts["mcts"] += 1
        return chosen

    def _act_sequential(self, engine: SnakeEnv) -> Action:
        root = TreeNode(engine.clone(), None, {}, 0, 0.0)
        for _ in range(self.simulations):
            leaf = select_leaf(root)
            child = expand(leaf)
            result = simulate(child, self.max_rollout_steps)
            backprop(child, result)
        chosen = best_action(root)
        if self.tree_debug_window:
            from agents.mcts.tree_view import mcts_panel_lines

            self.last_mcts_panel = mcts_panel_lines(root, chosen)
        return chosen

    def _act_parallel(self, engine: SnakeEnv) -> Action:
        self._ensure_pool()
        assert self._pool is not None
        k = self.num_workers
        base, rem = divmod(self.simulations, k)
        sims_per_worker = [base + (1 if i < rem else 0) for i in range(k)]
        sims_per_worker = [s for s in sims_per_worker if s > 0]
        seeds = [int(self.rng.integers(1, 2**31 - 1)) for _ in sims_per_worker]

        futures = [
            self._pool.submit(
                _mcts_worker, engine.clone(), s, self.max_rollout_steps, seed
            )
            for s, seed in zip(sims_per_worker, seeds)
        ]

        # Aggregate visits per action and a visit-weighted mean Q.
        agg_visits: dict[int, int] = {}
        agg_qv: dict[int, float] = {}
        for fut in futures:
            for a_int, visits, mean_q in fut.result():
                agg_visits[a_int] = agg_visits.get(a_int, 0) + visits
                agg_qv[a_int] = agg_qv.get(a_int, 0.0) + mean_q * visits

        # Terminal root (e.g. snake already dead on this tick): no children expanded.
        if not agg_visits:
            return Action.STRAIGHT

        # Pick most visits; tie-break by aggregated mean Q, then random (no STRAIGHT bias).
        best_key: tuple[int, float] | None = None
        candidates: list[int] = []
        for a_int, v in agg_visits.items():
            mean_q = (agg_qv[a_int] / v) if v else -1e30
            key = (v, mean_q)
            if best_key is None or key > best_key:
                best_key = key
                candidates = [a_int]
            elif key == best_key:
                candidates.append(a_int)
        chosen = Action(int(self.rng.choice(candidates)))

        if self.tree_debug_window:
            from agents.mcts.tree_view import mcts_panel_lines_from_summary

            summary: dict[Action, tuple[int, float]] = {
                Action(a_int): (v, (agg_qv[a_int] / v) if v else 0.0)
                for a_int, v in agg_visits.items()
            }
            total_visits = sum(agg_visits.values())
            self.last_mcts_panel = mcts_panel_lines_from_summary(
                summary, chosen, total_visits, len(futures)
            )
        return chosen
