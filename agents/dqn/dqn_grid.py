import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn.dqn import BaseDQNAgent

class CoordConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, _, H, W = x.shape
        device = x.device
        dtype = x.dtype

        x_coord = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        y_coord = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y_coord, x_coord, indexing="ij")

        xx = xx.unsqueeze(0).expand(B, -1, -1)
        yy = yy.unsqueeze(0).expand(B, -1, -1)
        coords = torch.stack([xx, yy], dim=1)
        return torch.cat([x, coords], dim=1)


class CNNQNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        grid_shape: tuple[int, int],
        output_dim: int = 3,
        hidden_dim: int = 256,
        use_coord_conv: bool = True,
    ):
        super().__init__()

        self.use_coord_conv = use_coord_conv
        conv_in_channels = in_channels + 2 if use_coord_conv else in_channels

        if use_coord_conv:
            self.coord_conv = CoordConv()

        # No normalization — snake obs is sparse, 1px gaps must be preserved
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        h_after = (grid_shape[0] + 1) // 2
        w_after = (grid_shape[1] + 1) // 2
        flatten_dim = 128 * h_after * w_after  # 128 * 10 * 10 = 12800

        # Projection: compress before streams, LayerNorm safe on dense vector
        self.projection = nn.Sequential(
            nn.Linear(flatten_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))

        x = x.float()

        if self.use_coord_conv:
            x = self.coord_conv(x)

        cnn_out = self.conv(x)
        flattened = cnn_out.view(cnn_out.size(0), -1)

        projected = self.projection(flattened)
        values = self.value_stream(projected)
        advantages = self.advantage_stream(projected)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


class DQNGridAgent(BaseDQNAgent):
    def __init__(
        self,
        in_channels: int = 4,
        grid_shape: tuple[int, int] = (20, 20),
        frame_stack: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,      # fixed from 0.1
        device: str = "auto",
        model_file: str = "dqn_grid.pth",
        seed: int | None = None,
        use_coord_conv: bool = True,
    ):
        total_channels = in_channels * frame_stack

        q_net = CNNQNet(
            in_channels=total_channels,
            grid_shape=grid_shape,
            output_dim=3,
            use_coord_conv=use_coord_conv,
        )
        target_net = CNNQNet(
            in_channels=total_channels,
            grid_shape=grid_shape,
            output_dim=3,
            use_coord_conv=use_coord_conv,
        )
        optimizer = optim.AdamW(
            q_net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        super().__init__(q_net, target_net, optimizer, device, model_file, seed)