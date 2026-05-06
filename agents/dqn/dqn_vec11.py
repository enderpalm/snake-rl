import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn.dqn import BaseDQNAgent


class LinearQNet(nn.Module):
    def __init__(self, input_dim: int = 11, output_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNVec11Agent(BaseDQNAgent):
    def __init__(
        self,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        buffer_size: int = 100000,
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 0.005,
        eps_init: float = 1.0,
        eps_final: float = 0.05,
        eps_decay: float = 0.1,
        learning_starts: int = 1000,
        train_freq: int = 1,
        gradient_steps: int = 1,
        device: str = "cpu",
        seed: int | None = None,
        model_file: str = "dqn_vec11.pth",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        q_net = LinearQNet(input_dim=11, output_dim=3, hidden_dim=hidden_dim)
        target_net = LinearQNet(input_dim=11, output_dim=3, hidden_dim=hidden_dim)
        optimizer = optim.AdamW(
            q_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        super().__init__(
            q_net=q_net,
            target_net=target_net,
            optimizer=optimizer,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            eps_init=eps_init,
            eps_final=eps_final,
            eps_decay=eps_decay,
            learning_starts=learning_starts,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            device=device,
            seed=seed,
            model_file=model_file,
        )
