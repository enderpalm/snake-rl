import numpy as np
import torch
from typing import Tuple


class ReplayBuffer:
    """
    A vectorized Replay Buffer that stores experiences from multiple parallel environments.
    """

    def __init__(
        self,
        buffer_size: int,
        observe_shape: tuple,
        n_envs: int = 1,
        device: str = "cpu",
        seed: int | None = None,
    ):
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.n_envs = n_envs
        self.device = torch.device(device)
        self.pos = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

        # Ring buffer: storage for observations, actions, rewards, dones
        # Use next_observe and observe in single buffer with indexing
        self.observe = np.zeros(
            (self.buffer_size, self.n_envs, *observe_shape), dtype=np.uint8
        )
        self.terminal_obs = {}

        self.actions = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=bool)

    def add(
        self,
        observe: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_observe: np.ndarray,
        done: np.ndarray,
    ):
        """
        Add a new batch of transitions to the buffer.
        """
        # Ensure dimensions match expected (n_envs, ...).
        # Safely cast float32 Box outputs (like 0.0, 1.0) to uint8 to fit our optimized buffer.
        np.copyto(self.observe[self.pos], observe.astype(np.uint8, copy=False))
        np.copyto(self.actions[self.pos], action)
        np.copyto(self.rewards[self.pos], reward)
        np.copyto(self.dones[self.pos], done)

        # Update sparse terminal observations
        for env_idx in range(self.n_envs):
            # Clean up old terminal state at this position if the ring buffer looped around
            # If the episode ended here, save the true final frame
            self.terminal_obs.pop((self.pos, env_idx), None)
            if done[env_idx]:
                self.terminal_obs[(self.pos, env_idx)] = (
                    next_observe[env_idx].astype(np.uint8, copy=False).copy()
                )

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions. Returns batched tensors.
        """
        upper_bound = self.buffer_size if self.full else self.pos

        # Cannot sample the absolute latest transition if the buffer isn't full,
        # because its next_obs doesn't exist yet (unless it was terminal).
        # Safe trick: push the upper_bound back by 1 if not full.
        max_bound = max(1, upper_bound - 1) if not self.full else upper_bound
        batch_inds = self.rng.integers(0, max_bound, size=batch_size)
        env_inds = self.rng.integers(0, self.n_envs, size=batch_size)

        # Retrieve data
        obs = self.observe[batch_inds, env_inds, :]
        actions = self.actions[batch_inds, env_inds]
        rewards = self.rewards[batch_inds, env_inds]
        dones = self.dones[batch_inds, env_inds]

        # Calculate next_obs dynamically
        next_inds = (batch_inds + 1) % self.buffer_size
        next_obs = self.observe[next_inds, env_inds, :].copy()

        # Patch in the true terminal states where necessary
        for i in range(batch_size):
            if dones[i]:
                key = (batch_inds[i], env_inds[i])
                if key in self.terminal_obs:
                    next_obs[i] = self.terminal_obs[key]

        observe_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.uint8)
        actions_tensor = torch.as_tensor(actions, device=self.device).unsqueeze(-1)
        rewards_tensor = torch.as_tensor(rewards, device=self.device).unsqueeze(-1)
        next_observe_tensor = torch.as_tensor(
            next_obs, device=self.device, dtype=torch.uint8
        )
        dones_tensor = torch.as_tensor(
            dones, device=self.device, dtype=torch.bool
        ).unsqueeze(-1)

        return (
            observe_tensor,
            actions_tensor,
            rewards_tensor,
            next_observe_tensor,
            dones_tensor,
        )
