import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import envs
import numpy as np

env = make_vec_env('SnakeEnv-v0', n_envs=1)

model = PPO('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
print('Learned Model')

# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# print('Model Evaluation Done')


obs = env.reset()
env.seed(42)
env.action_space(42)
print('Beginning working wih trained agent')
for i in range(1000):
    env.render()
    action, _states = model.predict(obs, deterministic=True)
    # print(i, type(action))
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs, info = env.reset()
env.close()
print('Rendering done')