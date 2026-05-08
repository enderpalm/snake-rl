import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn.dqn import BaseDQNAgent


class CNNQNet(nn.Module):
    """
    CNN Q-Network for Full Grid observations (with optional framestacking).
    Adapted from Nature DQN CNN architecture: https://arxiv.org/pdf/2201.07211

    Uses AdaptiveMaxPool2d((5, 5)) for better generalization across grid sizes.
    """

    def __init__(
        self, in_channels: int, grid_shape: tuple[int, int], output_dim: int = 3, hidden_dim: int = 256
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        flatten_dim = 64 * grid_shape[0] * grid_shape[1]

        self.value_stream = nn.Sequential(nn.Linear(flatten_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        self.advantage_stream = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
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

        values = self.value_stream(flattened)
        advantages = self.advantage_stream(flattened)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


class DQNGridAgent(BaseDQNAgent):
    """
    DQN Agent configured specifically for Grid-based observations with CNN.
    """

    def __init__(
        self,
        in_channels: int = 4,
        grid_shape: tuple[int, int] = (20, 20),
        frame_stack: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.1,
        device: str = "auto",
        model_file: str = "dqn_grid.pth",
        seed: int | None = None,
    ):
        total_channels = in_channels * frame_stack

        q_net = CNNQNet(in_channels=total_channels, grid_shape=grid_shape, output_dim=3)
        target_net = CNNQNet(in_channels=total_channels, grid_shape=grid_shape, output_dim=3)
        optimizer = optim.AdamW(q_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        super().__init__(
            q_net,
            target_net,
            optimizer,
            device,
            model_file,
            seed,
        )
