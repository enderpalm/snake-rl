import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from agents.dqn.dqn import BaseDQNAgent


class CNNQNet(nn.Module):
    """
    CNN Q-Network for Full Grid observations (with optional framestacking).
    Uses AdaptiveAvgPool2d(4, 4) for better generalization across grid sizes.
    """

    def __init__(self, in_channels: int, output_dim: int = 3, grid_size: int = 10):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        flatten_dim = 64 * 4 * 4
        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 256), nn.ReLU(), nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If using FrameStack (num_envs, num_frames, channels, height, width), flatten frames and channels
        if x.ndim == 5:
            # x shape: (B, F, C, H, W) -> (B, F * C, H, W)
            x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))

        # Ensure tensor is float32 for PyTorch layers, even if retrieved as uint8 from ReplayBuffer
        x = x.float()

        cnn_out = self.conv(x)
        flattened = cnn_out.view(cnn_out.size(0), -1)
        return self.fc(flattened)


class DQNCnnAgent(BaseDQNAgent):
    """
    DQN Agent configured specifically for Grid-based observations with CNN.
    """

    def __init__(
        self,
        in_channels: int = 4,  # Changed for framestacking
        grid_size: int = 10,
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 0.005,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        exploration_fraction: float = 0.1,
        learning_starts: int = 1000,
        train_freq: int = 4,
        gradient_steps: int = 1,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        q_net = CNNQNet(in_channels=in_channels, output_dim=3, grid_size=grid_size)
        target_net = CNNQNet(in_channels=in_channels, output_dim=3, grid_size=grid_size)
        optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

        super().__init__(
            q_net=q_net,
            target_net=target_net,
            optimizer=optimizer,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            eps_init=exploration_initial_eps,
            eps_final=exploration_final_eps,
            eps_decay=exploration_fraction,
            learning_starts=learning_starts,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            device=device,
            seed=seed,
        )
