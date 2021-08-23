import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        with torch.no_grad():
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
        return x


class DQN_COPY(object):
    def __init__(self, state_dim, action_dim, hidden_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_dim
        self.critic_net = Critic(self.state_dim, self.action_dim, self.hidden_size)

    def choose_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
        action = torch.argmax(self.critic_net(observation)).item()
        return action

    def load(self, run_dir, episode):
        base_path = os.path.join(run_dir, 'trained_model')
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        self.critic_net.load_state_dict(torch.load(model_critic_path))
