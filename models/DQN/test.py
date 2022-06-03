import gym
import envs
import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

env = Monitor(gym.make('SnakeEnv-v0', window_size=256, block_size=8))
print(os.path.exists('dqn.zip'))
model = DQN.load('dqn', env=env)
env.seed(42)
env.action_space.seed(42)

obs = env.reset()
for _ in range(50):
    env.render()
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

    if done:
        obs = env.reset()

env.close()