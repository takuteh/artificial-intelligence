
'''
ゴール到達用の報酬設定
状態に障害物の状態を含めていない
'''
import sys
sys.path.append("..")
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import robot_simulator
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
        self.sim = robot_simulator.Simulate()
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
        self.obstacles=[]
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
        gene=self.actions[action]
        repeat=round(gene[1]/0.1)
        correct_action=[gene[0],0.1,gene[2],0.1]
        
        for i in range(repeat):
            self.robot_x,self.robot_y,self.robot_angle = self.sim.simulation_twewheel(correct_action)
            obstacle_detection,distance_to_obstacle,angle_to_obstacle=self._get_obstacle(self.obstacles,self.robot_x,self.robot_y,self.robot_angle)
            print(f"障害物の座標:({self.obstacles}),ロボットの座標({self.robot_x},{self.robot_y})")
            print("障害物まで:",distance_to_obstacle)
            print("障害物との角度差:",angle_to_obstacle)
            #障害物に衝突した場合 
            if distance_to_obstacle < 0.3:
                print("衝突！！")
                reward -= 200
                terminated =True
                self.collision=1
                self.step_p=0
                break
        # ゴールとの距離と角度差を取得
        distance_to_goal = self._calculate_distance(self.goal_x,self.goal_y,self.robot_x,self.robot_y)
        angle_diff = self._calculate_angle_diff(self.goal_x,self.goal_y,self.robot_x,self.robot_y,self.robot_angle)
    
        #報酬設定
        reward -= abs(angle_diff)*0.1+distance_to_goal
       
        self.old_distance=distance_to_goal
        self.old_x=self.robot_x
        self.old_y=self.robot_y
        # 終了条件
        if bool(distance_to_goal < 0.5) :
            #reward += 1000
            terminated =True
            self.step_p=0
        
        if self.step_p>=200:
            terminated=True
            self.step_p=0
            
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

    def _get_obstacle(self,obstacles,robot_x,robot_y,robot_angle):
        obstacle_detection=False
        distance_to_obstacle=100
        angle_to_obstacle=180
        distance_min=float('inf')
        angle_min=180
        min_x=100
        min_y=100
        #最も近い障害物を見つける
        for obs in obstacles:
            distance_tmp= self._calculate_distance(obs[0],obs[1],robot_x,robot_y)
            if distance_min > distance_tmp:
                distance_min = distance_tmp
                min_x=obs[0]
                min_y=obs[1]
                angle_min=self._calculate_angle_diff(min_x,min_y,robot_x,robot_y,robot_angle)
            distance_to_obstacle=distance_min
            angle_to_obstacle=angle_min
        if distance_min < 1 :
            print("障害物発見!!")
            print(f"最も近い障害物の座標:({min_x},{min_y})")
            obstacle_detection=True
            distance_to_obstacle=distance_min
            angle_to_obstacle=angle_min

        return obstacle_detection,distance_to_obstacle,angle_to_obstacle