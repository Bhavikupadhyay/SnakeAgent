import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DQN
import envs
import numpy as np

env = make_vec_env('SnakeEnv-v0')

model = DQN('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
print('Learned Model')

#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#print('Model Evaluation Done')


obs = env.reset()
# setting up the seed for reproducibility
env.seed(42)
env.action_space.seed(42)
print('Beginning working wih trained agent')
for _ in range(100):
    env.render()
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

    if done:
        obs = env.reset()
env.close()
print('Rendering done')
