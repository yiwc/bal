import sys
import os
import pickle
import numpy as np
import gym
import time
import env
from gym import envs, spaces, core
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Binocular_Extractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(Binocular_Extractor, self).__init__(observation_space, features_dim)

        std_n=64
        act=nn.Tanh

        self.R_extractor_linear = nn.Sequential(
             nn.Linear(2, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim/2)),
             act()
        )
        self.L_extractor_linear = nn.Sequential(
             nn.Linear(2, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, std_n),
             act(),
             nn.Linear(std_n, int(features_dim/2)),
             act()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print(observations)
        obs_left = observations[:,:,0:2]
        obs_right = observations[:,:,2:4]

        # with torch.no_grad():
        #     l1=l1+th.rand_like(l1)*0.00001
        #     l2=l2+th.rand_like(l2)*0.00001

        L = self.L_extractor_linear(obs_left.flatten(1))
        R = self.R_extractor_linear(obs_right.flatten(1))
        extracted_obs = th.cat([L,R],1)
        return extracted_obs

policy_kwargs = dict(
    features_extractor_class = Binocular_Extractor,
    features_extractor_kwargs = dict(features_dim=256),
    net_arch = [256,256,256],
)
env = gym.make('BAL_env-v1')
model = PPO("MlpPolicy", env=env, learning_rate=0.0008, tensorboard_log="/root/BAL/tensorboard/BAL_env-v1/", policy_kwargs=policy_kwargs,\
     verbose=1,n_steps=64, batch_size=32, create_eval_env=True)
model.learn(100000)
model.save("/root/BAL/model/env_bal-v1.pkl")