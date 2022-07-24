import gym
import envs
import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


env = Monitor(gym.make('SnakeEnv-v0', window_size=256, block_size=16))

model = DQN.load('dqn4', env=env)

env.seed(42)
env.action_space.seed(42)

obs = env.reset()
count = 0
total_score, total_steps = 0, 0
while count < 5:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

    if done:
        score, steps = env.get_logging_details()
        obs = env.reset()
        count += 1
        total_score += score
        total_steps += steps

print('Average Score: {}\tAverage Steps: {}'.format(total_score / count, total_steps / count))
env.close()
