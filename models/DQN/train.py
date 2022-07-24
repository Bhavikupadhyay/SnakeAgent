import gym
import torch.cuda

import envs
import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from utils.CustomBaseFeaturesExtractor import CustomCNN
from utils.CustomCallback import TensorboardCallback


env = Monitor(gym.make('SnakeEnv-v0', window_size=256, block_size=16))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=64)
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DQN(
    policy='CnnPolicy',
    env=env,
    learning_rate=5e-5,
    verbose=1,
    policy_kwargs=policy_kwargs,
    learning_starts=100000,
    tensorboard_log='../../logs/dqn5/log',
    optimize_memory_usage=True,
    exploration_fraction=1,
    exploration_initial_eps=0.6,
    exploration_final_eps=0.4,
    create_eval_env=True,
    device=device
)
if os.path.exists('dqn5.zip'):
    model = DQN.load(
        'dqn5',
        env=env,
        device=device,
        custom_objects={
            'learning_starts': 0,
            'exploration_final_eps': 0.2,
            'exploration_fraction': 1
        }
    )

checkpoint_callback = CheckpointCallback(save_freq=500000, save_path='../../logs/dqn5/chck/')
eval_callback = EvalCallback(env, best_model_save_path="../../logs/dqn5/best/", eval_freq=500)
tensorboard_callback = TensorboardCallback(env)
callback_list = CallbackList([checkpoint_callback, eval_callback, tensorboard_callback])

model.learn(
    total_timesteps=1000000,
    log_interval=100,
    callback=callback_list,
    tb_log_name='run',
    eval_log_path='../../logs/dqn5/eval/',
    reset_num_timesteps=False
)

model.save('dqn5')
print('Model Saved')

mean_reward, std_reward = evaluate_policy(
    model=model,
    env=model.get_env(),
    n_eval_episodes=10
)

print(mean_reward, std_reward)
