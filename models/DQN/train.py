import gym
import envs
import os

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from utils.CustomBaseFeaturesExtractor import CustomCNN
from utils.CustomCallback import TensorboardCallback


env = Monitor(gym.make('SnakeEnv-v0', window_size=256, block_size=8))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs = dict(features_dim=256)
)

model = DQN(
    policy='CnnPolicy',
    env=env,
    learning_rate=0.0005,
    verbose=1,
    buffer_size=10000,
    policy_kwargs=policy_kwargs,
    tensorboard_log='../../logs/dqn/log',
    optimize_memory_usage=True,
    exploration_initial_eps = 0.5,
    exploration_final_eps = 0.1,
    create_eval_env=True
)
if os.path.exists('dqn.zip'):
    model = DQN.load('dqn', env=env)

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='../../logs/dqn/chck/')
eval_callback = EvalCallback(env, best_model_save_path="../../logs/dqn/best/", eval_freq=500)
tensorboard_callback = TensorboardCallback(env)
callback_list = CallbackList([checkpoint_callback, eval_callback, tensorboard_callback])

model.learn(
    total_timesteps=100000,
    log_interval=4,
    callback=callback_list,
    tb_log_name='run',
    eval_log_path='../../logs/dqn/eval/',
    reset_num_timesteps=False
)

model.save('dqn_2')
print('Model Saved')

mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    n_eval_episodes=10
)
