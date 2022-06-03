from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import torch
from torch import nn as nn


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]

        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.net(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.net(observations))
