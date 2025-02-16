'''
---DQNの動作を簡単に確かめるための環境---
目標 : エージェントをゴールのある角度と一致させること
行動 : 角度を1度づつ増やすか,減らすか
報酬 : ゴールとエージェントの角度差
'''
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import math
import json
import matplotlib.pyplot as plt
import torch
from gym import logger, spaces

class AngleMatchEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()

        # 状態空間 (角度と目標差分、ステップ数)
        self.observation_space = gym.spaces.Box(
            low=np.array([-180.0, -180.0, 0]), high=np.array([180.0, 180.0, 100]), dtype=np.float32
        )
        # 行動空間
        self.action_space = spaces.Discrete(2)
        self.angle = 0.0
        self.goal_angle = 0.0
        self.step_p = 0

    def reset(self, seed=None):
        self.angle = 0.0
        self.step_p = 0
        observation = np.array([self.angle, self.goal_angle - self.angle, self.step_p], dtype=np.float32)
        return observation, {}

    def step(self, action):
        # 行動に応じた角度更新
        if action == 0:
            self.angle += 1
        elif action == 1:
            self.angle -= 1

        # 報酬計算
        reward = - abs(self.goal_angle - self.angle) * 0.1
        if abs(self.goal_angle - self.angle) < 1.0:
            reward += 10

        # 終了条件
        terminated = abs(self.goal_angle - self.angle) < 1.0 or self.step_p >= 360

        # 次の観測値
        observation = np.array([self.angle, self.goal_angle - self.angle, self.step_p], dtype=np.float32)
        self.step_p += 1
        print(reward)
        return observation, reward, terminated, False, {}


