import gym
import stable_baselines3.common.logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import envs
import numpy as np
import torch
import torch.nn as nn


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]

        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.net(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.net(observations))


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)
env = gym.make('SnakeEnv-v0', window_size=256, block_size=8)
eval_env = gym.make('SnakeEnv-v0', window_size=256, block_size=8)

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='/dqn/chck/', name_prefix='dqn_chck')
eval_callback = EvalCallback(eval_env, best_model_save_path='/dqn/best', log_path='/dqn/eval, eval_freq=500')
callback = CallbackList([checkpoint_callback, eval_callback])

model = DQN('MlpPolicy', env, verbose=1, buffer_size=10000, policy_kwargs=policy_kwargs, tensorboard_log='./dqn/logs')
model.learn(total_timesteps=1400, log_interval=4, callback=callback, tb_log_name='first_run', eval_log_path='./dqn/eval_logs')
print('Learned Model')

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(mean_reward, std_reward)
print('Model Evaluation Done')
model.save('dqn_model')

obs = env.reset()
# setting up the seed for reproducibility
env.seed(42)
env.action_space.seed(42)
print('Beginning working wih trained agent')
for _ in range(50):
    env.render()
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

    if done:
        obs = env.reset()
env.close()
print('Rendering done')
