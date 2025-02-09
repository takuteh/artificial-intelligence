import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import simulator
import math
import json
import matplotlib.pyplot as plt
import torch
def load_action_list(filename="action_list.json"):
    with open(filename, "r") as file:
        actions = json.load(file)
    return actions

class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()

        #シミュレータをインスタンス化
        self.sim = simulator.Simulate()
        self.actions=load_action_list()
        # 状態空間（連続的な観測データ）
        self.observation_space = gym.spaces.Box(
            low=np.array([-180,-180,-180,-180,-100]), high=np.array([180,180,180,180,100]), dtype=np.float32
        )

        # 離散的な行動空間 (4つの選択肢)
        self.action_space = gym.spaces.Discrete(len(load_action_list()))

        # ロボットとゴールの初期位置
        self.robot_x=0
        self.robot_y=0
        self.old_x=0
        self.old_y=0
        self.goal_x=0
        self.goal_y=0
        self.robot_angle = 0.0  # ロボットの向き（ラジアン）
        self.step_p=0
        self.old_distance=np.sqrt((self.goal_x - self.robot_x) ** 2 + (self.goal_y - self.robot_y) ** 2)
    def reset(self, seed=None):
        # 状態初期化
        #遺伝子を元にロボットを移動
        self.sim.reset()
        self.robot_x,self.robot_y,self.robot_angle = self.sim.simulation_twewheel([1,0.1,1,0.1])
        print("reset:x,y,theta->",self.robot_x,self.robot_y,self.robot_angle)
        # 初期状態の計算
        distance = self._calculate_distance(self.goal_x,self.goal_y,self.robot_x,self.robot_y)
        angle_diff = self._calculate_angle_diff(self.goal_x,self.goal_y,self.robot_x,self.robot_y,self.robot_angle)

        observation = np.array([(self.robot_x),(self.robot_y),(self.robot_angle),(angle_diff),(distance)], dtype=np.float32)
        return observation, {}

    def step(self, action):
        terminated=False
        reward=0
        #遺伝子を元にロボットを移動
        self.robot_x,self.robot_y,self.robot_angle = self.sim.simulation_twewheel(self.actions[action])

        # ゴールとの距離と角度差を取得
        distance_to_goal = self._calculate_distance(self.goal_x,self.goal_y,self.robot_x,self.robot_y)
        angle_diff = self._calculate_angle_diff(self.goal_x,self.goal_y,self.robot_x,self.robot_y,self.robot_angle)
    
        #報酬設定
        #if abs(self.goal_x - self.robot_x) < abs(self.goal_x - self.old_x) or abs(self.goal_y - self.robot_y) < abs(self.goal_y - self.old_y):
        #    reward = 1    
        
        reward -= abs(angle_diff)*0.1+distance_to_goal
        #reward = 100/abs(self.goal_x-self.robot_x+0.01) + 100/abs(self.goal_y - self.robot_y+0.01) #-abs(0.01*angle_diff)
        #if (self.robot_x-self.old_x)==0 and (self.robot_y-self.old_y)==0:
        #    reward -= 10 
       
        self.old_distance=distance_to_goal
        self.old_x=self.robot_x
        self.old_y=self.robot_y
        # 終了条件
        if bool(distance_to_goal < 0.5) :
            #reward += 1000
            terminated =True
            self.step_p=0
        # if self.robot_x>10 or self.robot_x <0 or self.robot_y>10 or self.robot_y<0:
        #     reward -= 100
        if self.step_p>=200:
            # reward -=100
            terminated=True
            self.step_p=0
            #terminated = True
            #self.step_p=0
        print(f"goal:({self.goal_x,self.goal_y})")
        print("selected_action:",self.actions[action])
        print("angle_diff:",angle_diff)
        print("reward:",reward)
        self.step_p += 1 
        return np.array([(self.robot_x),(self.robot_y),(self.robot_angle),(angle_diff),(distance_to_goal)], dtype=np.float32), reward, terminated, False, {}

    def _calculate_distance(self,goal_x,goal_y,robot_x,robot_y):
        #ゴールとの距離
        distance_to_goal = np.sqrt((goal_x - robot_x) ** 2 + (goal_y - robot_y) ** 2)
        print("distance_to_goal:",distance_to_goal)
        return distance_to_goal

    def _calculate_angle_diff(self,goal_x,goal_y,robot_x,robot_y,robot_angle):
        # ゴール方向の角度を計算
        dx_goal = goal_x - robot_x
        dy_goal = goal_y - robot_y
        goal_angle = math.degrees(math.atan2(dy_goal, dx_goal))  # ゴール方向の角度を度数法に変換
        print("goal_angle (degrees):", goal_angle)
        # ロボットの向きとゴール方向の角度差を計算
        angle_diff = goal_angle - robot_angle
        return angle_diff

