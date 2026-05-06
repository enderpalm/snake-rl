import csv
import numpy as np
import random
from pathlib import Path
from rich.box import ROUNDED
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from typing import Any, Tuple

from gymnasium.vector import SyncVectorEnv
from agents.base import Agent
from core.env.core import RenderOptions, RewardOptions, SnakeEnv
from core.env.types import (
    DEFAULT_RENDER_OPTIONS,
    DEFAULT_REWARD,
    ObserveType,
    RenderMode,
)

from gymnasium.wrappers import FrameStackObservation

METRIC_PATH = "../artifacts/metrics/"


def _info_for_env_idx(
    infos: dict[str, Any] | None, env_idx: int, num_envs: int
) -> dict[str, Any] | None:
    """Extract the info dict for a single sub-env from SyncVectorEnv batched infos."""
    if not infos:
        return None
    out: dict[str, Any] = {}
    for k, v in infos.items():
        if isinstance(v, (list, tuple)):
            if len(v) == num_envs:
                out[k] = v[env_idx]
            else:
                out[k] = v
        elif isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == num_envs:
            out[k] = v[env_idx]
        else:
            out[k] = v
    return out


def _sample_range(v: int | Tuple[int, int]) -> int:
    """Sample from a range if a tuple is provided, else return the value."""
    return random.randint(*v) if isinstance(v, tuple) else v


def _aggregate_metrics(lst: list[float]) -> dict[str, float]:
    """Compute statistical aggregates for a list of metrics."""
    return {
        "Average": float(np.mean(lst)) if lst else 0.0,
        "Median": float(np.median(lst)) if lst else 0.0,
        "Min": float(np.min(lst)) if lst else 0.0,
        "Max": float(np.max(lst)) if lst else 0.0,
        "Std Dev": float(np.std(lst)) if lst else 0.0,
    }


def _print_summary_table(
    stats: dict[str, dict[str, float]], death_dist: dict[str, int]
) -> None:
    console = Console()

    # Create a table for metrics
    metrics_table = Table(title="Evaluation Metrics", box=ROUNDED, title_justify="left")
    metrics_table.add_column("Metric", justify="left", no_wrap=True)

    for key in next(iter(stats.values())).keys():
        metrics_table.add_column(key, justify="right")

    for name, stat in stats.items():
        metrics_table.add_row(
            name.capitalize(),
            f"{stat['Average']:.2f}",
            f"{stat['Median']:.2f}",
            f"{stat['Min']:.2f}",
            f"{stat['Max']:.2f}",
            f"{stat['Std Dev']:.2f}",
        )

    death_table = Table(title="Death Distribution", box=ROUNDED, title_justify="left")
    death_table.add_column("Reason", justify="left", no_wrap=True)
    death_table.add_column("Count", justify="right")
    death_table.add_column("Percentage", justify="right")

    total_deaths = sum(death_dist.values())
    for reason, count in sorted(death_dist.items(), key=lambda x: -x[1]):
        percentage = (count / total_deaths) * 100 if total_deaths > 0 else 0
        death_table.add_row(reason, str(count), f"{percentage:.1f}%")

    console.print(metrics_table)
    console.print(death_table)


def evaluate_agent(
    agent: Agent,
    num_episodes: int = 100,
    width: int = 20,
    height: int = 20,
    obs_type: ObserveType = ObserveType.VEC_11,
    num_apples: int | Tuple[int, int] = (1, 3),
    num_obstacles: int | Tuple[int, int] = (0, 10),
    num_envs: int = 1,
    render_mode: RenderMode | None = None,
    render_options: RenderOptions = DEFAULT_RENDER_OPTIONS,
    reward_options: RewardOptions = DEFAULT_REWARD,
    max_steps: int = 2000,
    seed: int | None = None,
    snapshot_engine_state: bool = False,
    frame_stack: int = 1,
) -> tuple[dict, dict, dict, dict]:
    # determinism
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    def make_env(env_id: int):
        env_seed = (seed + env_id) if seed is not None else None

        def _init():
            env = SnakeEnv(
                render_mode=render_mode,
                width=width,
                height=height,
                obs_type=obs_type,
                num_apples=_sample_range(num_apples),
                num_obstacles=_sample_range(num_obstacles),
                seed=env_seed,
                render_options=render_options,
                reward_options=reward_options,
                max_steps=max_steps,
                snapshot_engine_state=snapshot_engine_state,
            )
            if frame_stack > 1:
                env = FrameStackObservation(env, stack_size=frame_stack)
            return env

        return _init

    if render_mode == RenderMode.HUMAN:
        print("Rendering enabled. Single environment will be run at a time.")
        num_envs = 1

    env = SyncVectorEnv([make_env(i) for i in range(num_envs)])
    obs, infos = env.reset(seed=seed)

    if hasattr(agent, "reset_decision_stats"):
        agent.reset_decision_stats()

    metrics = {"rewards": [], "apples": [], "steps": []}
    death_dist: dict = {}
    ep_rewards = np.zeros(num_envs, dtype=float)
    ep_steps = np.zeros(num_envs, dtype=int)
    completed = 0

    with tqdm(
        total=num_episodes,
        desc="Evaluating Agent",
        disable=(render_mode is RenderMode.HUMAN),
    ) as pbar:
        while completed < num_episodes:
            # collect actions for each sub-env
            actions = [
                agent.act(obs[i], _info_for_env_idx(infos, i, num_envs))
                for i in range(num_envs)
            ]

            if render_mode is RenderMode.HUMAN and num_envs >= 1:
                setattr(
                    env.envs[0], "mcts_panel", getattr(agent, "last_mcts_panel", None)
                )

            next_obs, rewards_arr, dones, truncs, infos = env.step(actions)
            ep_rewards += rewards_arr
            ep_steps += 1

            # handle finished sub-environments
            for i, (done_i, trunc_i) in enumerate(zip(dones, truncs)):
                if not (done_i or trunc_i):
                    continue
                if completed >= num_episodes:
                    # still need to reset counters for env slot but don't record extra episodes
                    ep_rewards[i] = ep_steps[i] = 0
                    continue

                info_i = _info_for_env_idx(infos, i, num_envs) or {}
                metrics["rewards"].append(float(ep_rewards[i]))
                metrics["steps"].append(int(ep_steps[i]))
                metrics["apples"].append(int(info_i.get("apples_eaten", 0)))

                death_reason = info_i.get("death_reason")
                if death_reason:
                    death_dist[death_reason] = death_dist.get(death_reason, 0) + 1

                # re-sample environment boundaries for the slot
                env.envs[i].num_apples = _sample_range(num_apples)
                env.envs[i].num_obstacles = _sample_range(num_obstacles)

                completed += 1
                if render_mode is RenderMode.HUMAN:
                    print(
                        f"Episode {completed}/{num_episodes} - "
                        f"Reward: {ep_rewards[i]:.2f}, Steps: {ep_steps[i]}, "
                        f"Apples: {info_i.get('apples_eaten', 0)}, "
                        f"Death: {getattr(death_reason, 'value', death_reason) or 'None'}"
                    )
                else:
                    pbar.update(1)

                ep_rewards[i] = ep_steps[i] = 0

            if render_mode is RenderMode.HUMAN:
                env.envs[0].render()

            obs = next_obs

    env.close()

    stats = {k: _aggregate_metrics(v) for k, v in metrics.items()}

    _print_summary_table(stats, death_dist)

    if hasattr(agent, "log_decision_stats"):
        agent.log_decision_stats()

    return stats["rewards"], stats["apples"], stats["steps"], death_dist


def save_metrics(logs: list[dict], filepath: str) -> None:
    """Save a list of log dictionaries to a CSV file.

    Args:
        logs: List of dictionaries containing metric data
        filepath: Path where the CSV file will be saved
    """
    if not logs:
        return

    filepath = (
        METRIC_PATH + filepath if not filepath.startswith(METRIC_PATH) else filepath
    )
    filepath_obj = Path(filepath)
    filepath_obj.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = logs[0].keys()

    with open(filepath, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logs)
