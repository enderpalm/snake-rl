import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os
from collections import deque
from agents.base import Agent
from typing import Optional
from agents.base import Agent
from core.env.core import SnakeEnv
from core.env.types import Action

MAX_MEMORY = 100_000
BATCH_SIZE = 1000


class DQNAgent(Agent):
    def __init__(self, lr, hidden_size, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995, gamma=0.9, seed=None):
        super().__init__(seed)
        self.number_games = 0
        self.epsilon = epsilon  # randomness
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, hidden_size, 3)
        self.trainer = DQNTrainer(self.model, lr=lr, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def act(self, state: np.ndarray, info: Optional[dict] = None) -> Action:
        if self.rng.random() < self.epsilon:
            action_idx = int(self.rng.integers(3))
        else:
            prediction = self.model(torch.tensor(state, dtype=torch.float))
            action_idx = int(torch.argmax(prediction).item())
        return Action(action_idx)

    def train(self, mode: bool = True) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, file_name="model.pth"):
        self.model.save(file_name)

    def load(self, path):
        if os.path.isfile(path):
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            self.epsilon = 0
            print(f"Loaded model from {path}")
        else:
            print(f"Model file not found at {path}. Starting with untrained model.")


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # ensure input is on same device as model parameters
        device = next(self.parameters()).device
        x = x.to(device)
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, file_name="model.pth"):
        model_folder_path = "../models"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        # save state dict on CPU for portability
        state_dict_cpu = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(state_dict_cpu, file_name)


class DQNTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        # use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # move tensors to device
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)

        if len(state.shape) == 1:
            # only one parameter to predict
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)
        target = pred.clone().detach()
        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                # compute max Q for next state
                next_pred = self.model(next_state[idx])
                q_new = reward[idx] + self.gamma * torch.max(next_pred)
            target[idx][torch.argmax(action[idx]).item()] = q_new
        # 2: q_new = r + y * max(next_predicted Q value) -> only do this if not done
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


def train():
    total_score = 0
    record = 0
    agent = DQNAgent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state()
        # get move
        final_move = agent.get_action(state_old)
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state()
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory (experience replay)
            game.reset()
            agent.number_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
            print("Game", agent.number_games, "Score", score, "Record:", record)


if __name__ == "__main__":
    train()
