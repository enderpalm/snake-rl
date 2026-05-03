import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Optional

from agents.base import Agent
from core.env.types import Action

class ActorCriticNet(nn.Module):
    def __init__(self, input_dim=11, action_dim=3):
        super(ActorCriticNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return probs, state_value

class ActorCriticAgent(Agent):
    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 3,
        lr: float = 0.0003,
        gamma: float = 0.99,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCriticNet(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma

    def act(self, state: np.ndarray, info: Optional[dict] = None) -> Action:
        # Move observation to GPU
        state_t = torch.from_numpy(state).float().to(self.device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
            
        with torch.no_grad():
            probs, _ = self.model(state_t)
        
        # Sampling handles exploration naturally
        dist = Categorical(probs)
        action_idx = dist.sample().item()
        return Action(int(action_idx))

    def update(
        self,
        state: np.ndarray, # These can be batches for your loop
        action: int,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        log_probs: Optional[torch.Tensor] = None, # Extra args for parallel efficiency
        values: Optional[torch.Tensor] = None
    ) -> None:
        # Prepare Tensors on GPU
        next_states_t = torch.from_numpy(next_state).float().to(self.device)
        rewards_t = torch.from_numpy(reward).float().to(self.device).unsqueeze(1)
        dones_t = torch.from_numpy(done).float().to(self.device).unsqueeze(1)

        _, next_values = self.model(next_states_t)
        
        # TD Target and Advantage
        td_targets = rewards_t + (1 - dones_t) * self.gamma * next_values
        advantages = td_targets - values

        # Actor-Critic Loss
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, td_targets.detach())

        loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, mode: bool = True) -> None:
        self.model.train() if mode else self.model.eval()

    def save(self, path: str) -> None:
        path = self._modify_path(path)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        path = self._modify_path(path)
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()