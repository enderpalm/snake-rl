import numpy as np
from typing import Callable, Any


def log_and_save_progress(
    step: int,
    log_step: int,
    episode_rewards: list[float],
    episode_lengths: list[float],
    best_reward: float,
    pbar: Any,
    eps: float,
    save_callback: Callable[[], None],
) -> float:
    """
    Periodically logs the best reward to a progress bar and saves the model.
    Returns the updated best_reward.
    """
    if step > 0 and step % log_step == 0:
        if len(episode_rewards) >= 100:
            recent_avg = np.mean(episode_rewards[-100:])
            recent_len = np.mean(episode_lengths[-100:])
            if recent_avg > best_reward:
                best_reward = recent_avg
                save_callback()
            pbar.set_postfix(
                {
                    "Avg Rwd (100)": f"{recent_avg:.2f}",
                    "Avg Len (100)": f"{recent_len:.2f}",
                    "Best": f"{best_reward:.2f}",
                    "Eps": f"{eps:.3f}",
                }
            )
        elif len(episode_rewards) > 0:
            pbar.set_postfix(
                {
                    "Last Rwd": f"{episode_rewards[-1]:.2f}",
                    "Last Len": f"{episode_lengths[-1]:.2f}",
                    "Eps": f"{eps:.3f}",
                }
            )
    return best_reward
