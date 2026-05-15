from gymnasium.vector import VectorEnv
from gymnasium import Wrapper
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict
from tqdm import tqdm
from agents.base import Agent
from agents.dqn.replay_buffer import ReplayBuffer
from util import log_and_save_progress


class BaseDQNAgent(Agent):
    """
    Base PyTorch DQN Agent with custom ReplayBuffer.
    Designed for vectorized and frame-stacked gymnasium-compatible environment.
    """

    def __init__(
        self,
        q_net: nn.Module,
        target_net: nn.Module,
        optimizer: optim.Optimizer,
        device: str = "auto",
        model_file: str = "dqn_base.pth",
        seed: int | None = None,
    ):
        super().__init__(seed)
        self.q_net = q_net.to(device)
        self.target_net = target_net.to(device)
        self.optimizer = optimizer

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        if device not in ["cpu", "cuda"]:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.scaler = torch.amp.GradScaler()
        self.seed = seed
        self.model_file = model_file
        self.replay_buffer: ReplayBuffer | None = None
        self.eps = 1.0

    def act(self, state: np.ndarray, info: dict | None = None) -> np.ndarray:
        is_unbatched = state.ndim in (1, 3)
        if state.ndim == 4:
            # Distinguish between batched grid (B, C, H, W) and unbatched framestacked (F, C, H, W)
            for m in self.q_net.modules():
                if isinstance(m, nn.Conv2d):
                    # Unbatched framestacked grid has C at dim 1, while network expects F*C channels
                    if state.shape[1] != m.in_channels:
                        is_unbatched = True
                    break

        n_envs = 1 if is_unbatched else state.shape[0]

        if self.rng.random() < self.eps:
            return self.rng.integers(0, 3, size=(n_envs,))

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            if is_unbatched:
                state_tensor = state_tensor.unsqueeze(0)

            q_values = self.q_net(state_tensor)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        return actions

    def train(
        self,
        env: VectorEnv | Wrapper | Any,
        total_timesteps: int,
        log_step: int = 5000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 0.005,  # for soft update of target network
        eps_init: float = 1.0,
        eps_final: float = 0.05,
        eps_decay: float = 0.1,
        learning_starts: int = 1000,
        train_freq: int = 1,
        gradient_steps: int = 1,
    ) -> Dict[str, list]:
        """
        Synchronous training loop adapted from Stable-Baselines3 paradigm.
        Takes a VectorEnv (SyncVectorEnv or AsyncVectorEnv) and steps through it.
        """

        if env.single_observation_space is None:
            raise ValueError(
                "Environment must have a single_observation_space defined for replay buffer initialization."
            )

        if not self.replay_buffer:
            self.replay_buffer = ReplayBuffer(
                buffer_size,
                observe_shape=env.single_observation_space.shape,
                n_envs=env.num_envs,
                device=self.device,
                seed=self.seed,
            )

        self.eps = eps_init

        obs, _ = env.reset()

        episode_rewards = []
        episode_lengths = []
        losses = []

        current_rewards = np.zeros(env.num_envs)
        current_lengths = np.zeros(env.num_envs)

        best_reward = -np.inf
        decay_steps = max(1, int(total_timesteps * eps_decay))

        with tqdm(total=total_timesteps, desc="Parallel Training") as pbar:
            for step in range(total_timesteps):
                actions = self.act(obs)
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                dones = np.logical_or(terminations, truncations)

                real_next_obs = next_obs.copy()
                if "final_observation" in infos:
                    for idx, is_done in enumerate(dones):
                        if is_done and infos["_final_observation"][idx]:
                            real_next_obs[idx] = infos["final_observation"][idx]

                self.replay_buffer.add(obs, actions, rewards, real_next_obs, dones)
                obs = next_obs

                current_rewards += rewards
                current_lengths += 1

                for idx in dones.nonzero()[0]:
                    episode_rewards.append(float(current_rewards[idx]))
                    # Use the final snake length from the info dict
                    snake_len = infos["snake_length"][idx] if "snake_length" in infos else 0
                    episode_lengths.append(float(snake_len))
                    current_rewards[idx] = 0
                    current_lengths[idx] = 0

                # Update exploration rate
                decay_progress = min(1.0, step / decay_steps)
                self.eps = eps_init - (eps_init - eps_final) * decay_progress

                # Train the network
                if step > learning_starts and step % train_freq == 0:
                    for _ in range(gradient_steps):
                        loss = self._optimize(batch_size, gamma, tau)
                        if loss is not None:
                            losses.append(loss)

                pbar.update(1)

                best_reward = log_and_save_progress(
                    step=step,
                    log_step=log_step,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    best_reward=best_reward,
                    pbar=pbar,
                    eps=self.eps,
                    save_callback=lambda: self.save(self.model_file),
                )

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "losses": losses,
        }

    def _optimize(self, batch_size: int, gamma: float, tau: float) -> float | None:
        if not self.replay_buffer or self.replay_buffer.pos < batch_size and not self.replay_buffer.full:
            return None

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(batch_size)

        # Cast observations to float32 for PyTorch layers
        obs = obs.float()
        next_obs = next_obs.float()

        # Ensure target shapes align to (Batch, 1) to avoid (B, B) broadcast bugs
        actions = actions.long().unsqueeze(1) if actions.ndim == 1 else actions.long()
        rewards = rewards.unsqueeze(1) if rewards.ndim == 1 else rewards
        dones = dones.unsqueeze(1) if dones.ndim == 1 else dones

        with torch.autocast(device_type=self.device, dtype=torch.float16):
            q_values = self.q_net(obs)
            q_sa = q_values.gather(1, actions)

            # Double DQN (DDQN) computation for better stability over standard DQN
            with torch.no_grad():
                next_actions = self.q_net(next_obs).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_obs)
                max_next_q_sa = next_q_values.gather(1, next_actions)
                target_q_values = rewards.float() + (~dones).float() * gamma * max_next_q_sa

            # Huber Loss (Smooth L1) handles large penalties (-20) much better than MSE
            loss = F.smooth_l1_loss(q_sa, target_q_values)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update target network with Polyak soft update, borrowed from DDPG
        # Inspired by Stable-Baselines3 and this paper: https://arxiv.org/pdf/1509.02971
        for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
            target_param.data.lerp_(param.data, tau)

        return loss.item()

    def save(self, path: str) -> None:
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.eps = 0.0
        self.q_net.load_state_dict(torch.load(path, map_location=str(self.device)))
        self.target_net.load_state_dict(self.q_net.state_dict())
