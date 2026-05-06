from gymnasium.vector import VectorEnv
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict
from tqdm import tqdm
from agents.base import Agent
from agents.dqn.replay_buffer import ReplayBuffer


class BaseDQNAgent(Agent):
    """
    Base PyTorch DQN Agent with VectorEnv training support and integrated ReplayBuffer.
    """

    def __init__(
        self,
        q_net: nn.Module,
        target_net: nn.Module,
        optimizer: optim.Optimizer,
        buffer_size: int = 100000,
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 0.005,  # for soft update of target network
        eps_init: float = 1.0,
        eps_final: float = 0.05,
        eps_decay: float = 0.1,
        learning_starts: int = 1000,
        train_freq: int = 1,
        log_step: int = 5000,
        gradient_steps: int = 1,
        device: str = "cpu",
        seed: int | None = None,
        model_file: str = "dqn_base.pth",
    ):
        super().__init__(seed)
        self.q_net = q_net.to(device)
        self.target_net = target_net.to(device)
        self.optimizer = optimizer

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.log_step = log_step

        self.eps = eps_init
        self.eps_init = eps_init
        self.eps_final = eps_final
        self.eps_decay = eps_decay

        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps

        self.device = device
        self.seed = seed
        self.model_file = model_file

    def act(self, state: np.ndarray, info: dict | None = None) -> np.ndarray:
        n_envs = state.shape[0] if state.ndim > 1 and state.shape[0] > 1 else 1

        if self.rng.random() < self.eps:
            return self.rng.integers(0, 3, size=(n_envs,))

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.uint8, device=self.device)
            if n_envs == 1 and state_tensor.ndim in [1, 3]:
                state_tensor = state_tensor.unsqueeze(0)

            q_values = self.q_net(state_tensor)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        return actions

    def train(self, env: VectorEnv, total_timesteps: int) -> Dict[str, list]:
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
                buffer_size=self.buffer_size,
                observe_shape=env.single_observation_space,
                n_envs=env.num_envs,
                device=self.device,
                seed=self.seed,
            )

        obs, _ = env.reset()

        episode_rewards = []
        episode_lengths = []
        losses = []

        current_rewards = np.zeros(env.num_envs)
        current_lengths = np.zeros(env.num_envs)

        best_reward = -np.inf

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

                for idx, is_done in enumerate(dones):
                    if is_done:
                        episode_rewards.append(float(current_rewards[idx]))
                        episode_lengths.append(float(current_lengths[idx]))
                        current_rewards[idx] = 0
                        current_lengths[idx] = 0

                # Update exploration rate
                decay_steps = int(total_timesteps * self.eps_decay)
                if step < decay_steps:
                    self.eps = self.eps_init - (self.eps_init - self.eps_final) * (
                        step / decay_steps
                    )
                else:
                    self.eps = self.eps_final

                # Train the network
                if step > self.learning_starts and step % self.train_freq == 0:
                    for _ in range(self.gradient_steps):
                        loss = self._optimize()
                        if loss is not None:
                            losses.append(loss)

                pbar.update(1)

                # Periodically log best reward and save model
                if step > 0 and step % self.log_step == 0:
                    if len(episode_rewards) >= 100:
                        recent_avg = np.mean(episode_rewards[-100:])
                        if recent_avg > best_reward:
                            best_reward = recent_avg
                            self.save(self.model_file)
                        pbar.set_postfix(
                            {
                                "Avg Rwd (100)": f"{recent_avg:.2f}",
                                "Best": f"{best_reward:.2f}",
                                "Eps": f"{self.eps:.3f}",
                            }
                        )
                    elif len(episode_rewards) > 0:
                        pbar.set_postfix(
                            {
                                "Last Rwd": f"{episode_rewards[-1]:.2f}",
                                "Eps": f"{self.eps:.3f}",
                            }
                        )

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "losses": losses,
        }

    def _optimize(self) -> Optional[float]:
        if (
            not self.replay_buffer
            or self.replay_buffer.pos < self.batch_size
            and not self.replay_buffer.full
        ):
            return None

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(
            self.batch_size
        )

        q_values = self.q_net(obs)
        q_sa = q_values.gather(1, actions)

        # Compute max Q'(s', a') with Target Network
        with torch.no_grad():
            next_q_values = self.target_net(next_obs)
            max_next_q_sa = next_q_values.max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_sa

        # Loss & Backprop
        loss = F.mse_loss(q_sa, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network with Polyak soft update
        for param, target_param in zip(
            self.q_net.parameters(), self.target_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return loss.item()

    def save(self, path: str) -> None:
        path = self._modify_path(path)
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.q_net.load_state_dict(torch.load(path, map_location=str(self.device)))
        self.target_net.load_state_dict(self.q_net.state_dict())
