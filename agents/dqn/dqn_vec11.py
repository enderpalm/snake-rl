import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn.dqn import BaseDQNAgent


class LinearQNet(nn.Module):
    def __init__(self, input_dim: int = 11, output_dim: int = 3, hidden_dim: int = 256):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Dueling DQN Aggregation
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


class DQNVec11Agent(BaseDQNAgent):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.1,
        device: str = "auto",
        model_file: str = "dqn_vec11.pth",
        seed: int | None = None,
    ):

        q_net = LinearQNet(input_dim=11, output_dim=3, hidden_dim=256)
        target_net = LinearQNet(input_dim=11, output_dim=3, hidden_dim=256)
        optimizer = optim.AdamW(q_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        super().__init__(
            q_net,
            target_net,
            optimizer,
            device,
            model_file,
            seed,
        )
