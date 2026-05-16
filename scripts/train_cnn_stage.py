"""
Snake CNN DQN — curriculum training to 30+ apples.
Run: python train.py
"""

import os, time
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import FrameStackObservation

# ── adjust these two imports to match your project ──
from agents.dqn.dqn_grid import DQNGridAgent
from core.env.core import SnakeEnv
from core.env.types import ObserveType, RenderMode, RenderOptions, RewardOptions
import core.env.observations as _obs

# ────────────────────────── constants ────────────────────────────────────── #

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ENVS  = 12
OUT_DIR   = "artifacts/models"
os.makedirs(OUT_DIR, exist_ok=True)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ────────────────────────── observation ──────────────────────────────────── #

def _binary_obs(env):
    g = np.zeros((4, env.height, env.width), dtype=np.float32)
    if env.snake and env.snake.body:
        hr, hc = env.snake.body[0]
        g[0, hr, hc] = 1.0
        for r, c in env.snake.body:
            g[1, r, c] = 1.0
    for r, c in env.apples:    g[2, r, c] = 1.0
    for r, c in env.obstacles: g[3, r, c] = 1.0
    return g

_obs.observe_full_grid = _binary_obs   # patch before any env is created

# ────────────────────────── curriculum ───────────────────────────────────── #

def _r(death):
    return RewardOptions(
        eats_apple=20.0, complete=100.0,
        penalty_step=0.0, penalty_loop=-3.0,
        death_wall=death, death_self=death + 2,
        shaping_closer=1.0, shaping_further=0.0,
    )

STAGES = [
    # (obstacles, target_apples, timesteps, eps_start, death_penalty, label)
    # (0,  30, 700_000, 1.00, -3.0, "stage_1_no_obstacles"),
    (5,  27, 400_000, 0.40, -4.0, "stage_2_5_obstacles"),
    (10, 25, 400_000, 0.30, -5.0, "stage_3_10_obstacles"),
    # (15, 20, 400_000, 0.25, -5.0, "stage_4_15_obstacles"),
]

# ────────────────────────── helpers ──────────────────────────────────────── #

def make_env(seed, obstacles, reward):
    def _init():
        env = SnakeEnv(
            width=20, height=20,
            obs_type=ObserveType.FULL_GRID,
            num_apples=3, num_obstacles=obstacles,
            max_steps=10000, seed=seed,
            reward_options=reward,
        )
        return FrameStackObservation(env, stack_size=2)
    return _init


def make_agent(path):
    agent = DQNGridAgent(
        in_channels=4, grid_shape=(20, 20), frame_stack=2,
        learning_rate=2e-4, weight_decay=0.01,
        device=DEVICE, model_file=path,
    )
    # try:
    #     agent.q_net      = torch.compile(agent.q_net,      mode="reduce-overhead")
    #     agent.target_net = torch.compile(agent.target_net, mode="reduce-overhead")
    # except Exception:
    #     pass
    return agent

# ────────────────────────── training ─────────────────────────────────────── #

prev_model = None

for idx, (obstacles, target, timesteps, eps, death, label) in enumerate(STAGES):
    stage_num = idx + 1
    out_path  = f"{OUT_DIR}/{label}.pth"
    reward    = _r(death)

    print(f"\n{'='*55}")
    print(f"  Stage {stage_num}/4 — {obstacles} obstacles — target {target}+ apples")
    print(f"  {timesteps:,} steps × {NUM_ENVS} envs = {timesteps*NUM_ENVS:,} effective")
    print(f"{'='*55}")

    envs  = AsyncVectorEnv([make_env(42 + i, obstacles, reward) for i in range(NUM_ENVS)])
    agent = make_agent(out_path)

    if prev_model and os.path.exists(prev_model):
        agent.load(prev_model)
        agent.eps           = eps
        agent.replay_buffer = None          # clear — old layout misleads new stage
        print(f"Loaded {prev_model} | eps={eps} | buffer cleared")

    t0 = time.time()
    stats = agent.train(
        env=envs,
        total_timesteps=timesteps,
        log_step=10_000,
        buffer_size=250_000,
        batch_size=256,
        gamma=0.997,
        tau=0.002,
        eps_init=eps,
        eps_final=0.03,
        eps_decay=0.25,
        learning_starts=5_000,
        train_freq=4,
        gradient_steps=2,
    )
    envs.close()

    elapsed = (time.time() - t0) / 60
    if stats["episode_lengths"]:
        recent = stats["episode_lengths"][-500:]
        avg    = np.mean(recent)
        peak   = int(np.max(recent))
        p75    = np.percentile(recent, 75)
        print(f"\n  {elapsed:.1f} min | avg={avg:.1f} | p75={p75:.1f} | max={peak} | target={target}+ {'OK' if avg >= target else 'Fail'}")

    prev_model = out_path
    print(f"  Saved: {out_path}")

print(f"\nDone. Best model: {prev_model}")
print("Evaluate: python -c \"\nimport torch\nfrom train import make_agent, make_env\nfrom core.env.types import RenderMode\nagent = make_agent('artifacts/models/stage_4_15_obstacles.pth')\nagent.load('artifacts/models/stage_4_15_obstacles.pth')\n\"")