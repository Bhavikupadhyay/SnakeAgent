import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines.deepq import DQN
import envs
import numpy as np
import tensorflow as tf

env = gym.make('SnakeEnv-v0', window_size=256, block_size=8)
policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[256, 128, 64, 32, 1])
model = DQN('MlpPolicy', env, verbose=1, buffer_size=10000, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=14000, log_interval=4)
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
