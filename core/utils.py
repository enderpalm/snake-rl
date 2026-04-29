import csv
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from typing import Any, Tuple

from gymnasium.vector import SyncVectorEnv
from agents.base import Agent
from core.env.core import RenderOptions, RewardOptions, SnakeEnv
from core.env.types import ObserveType, RenderMode

METRIC_PATH = "../artifacts/metrics/"
BREAKER_WIDTH = 60


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


def evaluate_agent(
    agent: Agent,
    num_episodes: int = 100,
    width: int = 20,
    height: int = 20,
    obs_type: ObserveType | int = ObserveType.VEC_11,
    num_apples: int | Tuple[int, int] = (1, 3),
    num_obstacles: int | Tuple[int, int] = (0, 10),
    num_envs: int = 1,
    render_mode: RenderMode | None = None,
    render_options: RenderOptions | None = None,
    reward_options: RewardOptions = RewardOptions(),
    max_steps: int = 2000,
    seed: int | None = None,
    snapshot_engine_state: bool = False,
) -> tuple[dict, dict, dict, dict]:

    # For the env to be exactly the same, both num_envs and seed must be the same.
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    def sample(val):
        return random.randint(*val) if isinstance(val, tuple) else val

    def env_factory(env_id: int):
        return lambda: SnakeEnv(
            render_mode=render_mode,
            width=width,
            height=height,
            obs_type=obs_type,
            num_apples=sample(num_apples),
            num_obstacles=sample(num_obstacles),
            seed=seed + env_id if seed else None,
            render_options=render_options,
            reward_options=reward_options,
            max_steps=max_steps,
            snapshot_engine_state=snapshot_engine_state,
        )

    if render_mode == RenderMode.HUMAN:
        print("Rendering enabled. Single environment will be run at a time.")
        num_envs = 1

    env = SyncVectorEnv([env_factory(i) for i in range(num_envs)])
    obs, infos = env.reset(seed=seed)

    if hasattr(agent, "reset_decision_stats"):
        agent.reset_decision_stats()

    metrics = {"rewards": [], "apples": [], "steps": []}
    death_dist = {}
    ep_rewards, ep_steps = np.zeros(num_envs), np.zeros(num_envs)
    completed = 0

    with tqdm(
        total=num_episodes,
        desc="Evaluating Agent",
        disable=(render_mode == RenderMode.HUMAN),
    ) as pbar:
        while completed < num_episodes:
            actions = [
                agent.act(
                    obs[i],
                    _info_for_env_idx(infos, i, num_envs),
                )
                for i in range(num_envs)
            ]
            if render_mode == RenderMode.HUMAN and num_envs >= 1:
                setattr(
                    env.envs[0],
                    "mcts_panel",
                    getattr(agent, "last_mcts_panel", None),
                )
            next_obs, rewards, done, truncs, infos = env.step(actions)
            ep_rewards += rewards
            ep_steps += 1

            for i in range(num_envs):
                if done[i] or truncs[i]:
                    if completed < num_episodes:
                        metrics["rewards"].append(ep_rewards[i])
                        metrics["steps"].append(ep_steps[i])
                        metrics["apples"].append(
                            infos.get("apples_eaten", [0] * num_envs)[i]
                        )

                        death_reason = infos.get("death_reason", [None] * num_envs)[i]
                        if death_reason:
                            death_dist[death_reason] = (
                                death_dist.get(death_reason, 0) + 1
                            )

                        # Re-sample boundaries for internal env
                        env.envs[i].num_apples = sample(num_apples)
                        env.envs[i].num_obstacles = sample(num_obstacles)

                        completed += 1

                        if render_mode == RenderMode.HUMAN:
                            print(
                                f"Episode {completed}/{num_episodes} - Reward: {ep_rewards[i]:.2f}, Steps: {ep_steps[i]}, Apples: {infos.get('apples_eaten', [0] * num_envs)[i]}, Death: {death_reason.value if death_reason else 'None'}"
                            )
                        else:
                            pbar.update(1)

                    ep_rewards[i] = ep_steps[i] = 0

            if render_mode == RenderMode.HUMAN:
                env.envs[0].render()

            obs = next_obs
    env.close()

    def agg(lst):
        return {
            "avg": float(np.mean(lst)) if lst else 0.0,
            "max": float(np.max(lst)) if lst else 0.0,
            "sd": float(np.std(lst)) if lst else 0.0,
        }

    stats = {k: agg(v) for k, v in metrics.items()}

    # Print Table

    print("\n" + "=" * BREAKER_WIDTH)
    print(f"{'Metric':<15} | {'Average':<8} | {'Max':<8} | {'Std Dev':<8}")
    print("-" * BREAKER_WIDTH)
    for name, st in stats.items():
        print(
            f"{name.capitalize():<15} | {st['avg']:<8.2f} | {st['max']:<8.2f} | {st['sd']:<8.2f}"
        )

    print("\nDeath Distribution:")
    for reason, count in sorted(death_dist.items(), key=lambda x: -x[1]):
        print(
            f" - {reason.value if hasattr(reason, 'value') else reason}: {count} ({(count / num_episodes) * 100:.1f}%)"
        )
    print("=" * BREAKER_WIDTH + "\n")

    if hasattr(agent, "log_decision_stats"):
        agent.log_decision_stats()

    return stats["rewards"], stats["apples"], stats["steps"], death_dist
