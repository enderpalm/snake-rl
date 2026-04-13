import numpy as np
import os
import pickle
from typing import Optional
from agents.base import Agent, Direction


class QLearningAgent(Agent):
    def __init__(
        self,
        state_dim=2048,
        action_dim=4,
        lr=0.05,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.q_table = np.zeros((state_dim, action_dim))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def _obs_to_state(self, obs: np.ndarray) -> int:  # 11-bit -> int for index in q-table
        return int("".join(obs.astype(int).astype(str)), 2)

    def act(self, state: np.ndarray, info: Optional[dict] = None) -> Direction:
        if self.rng.random() < self.epsilon:
            action_idx = int(self.rng.integers(4))  # Explore Default 4 directions
        else:
            state_idx = self._obs_to_state(state)
            q_values = self.q_table[state_idx]
            max_q = np.max(q_values)
            action_idx = int(self.rng.choice(np.flatnonzero(q_values == max_q)))
        return list(Direction)[action_idx]

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        state_idx = self._obs_to_state(state)
        next_state_idx = self._obs_to_state(next_state)

        best_next_action = np.argmax(self.q_table[next_state_idx])
        td_target = reward + (0 if done else self.gamma * self.q_table[next_state_idx][best_next_action])
        td_error = td_target - self.q_table[state_idx][action]
        self.q_table[state_idx][action] += self.lr * td_error

    def train(self, mode: bool = True) -> None:
        """Called at end of episode to decay exploration rate if mode=True."""
        if mode:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            self.epsilon = self.epsilon_min

    def save(self, path: str) -> None:
        path = self._modify_path(path)
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path: str) -> None:
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)
            self.epsilon = self.epsilon_min  # Keep epsilon low during testing/evaluation
            print(f"Loaded Q-table from {path}")
        else:
            print(f"Model file not found at {path}. Starting with untrained Q-table.")
