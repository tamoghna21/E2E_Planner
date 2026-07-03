"""State-vector -> [steering, throttle_brake] policy for behavior cloning."""
import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim=2, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return torch.tanh(self.net(obs))  # actions live in [-1, 1]
