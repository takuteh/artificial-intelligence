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
            low=np.array([-180,-180,-180,-180,-100,-180,-100,0]), high=np.array([180,180,180,180,100,180,100,1]), dtype=np.float32
        )

        # 離散的な行動空間 (4つの選択肢)
        self.action_space = gym.spaces.Discrete(len(load_action_list()))

        # ロボットとゴールの初期位置
        self.collision=0
        self.detect_flag=False
        self.robot_x=0
        self.robot_y=0
        self.old_x=0
        self.old_y=0
        self.goal_x=0
        self.goal_y=0
        self.obstacles=[]
        self.robot_angle = 0.0  # ロボットの向き（ラジアン）
        self.step_p=0
        self.old_distance=np.sqrt((self.goal_x - self.robot_x) ** 2 + (self.goal_y - self.robot_y) ** 2)
    def reset(self, seed=None):
        # 状態初期化
        self.collision=0
        #遺伝子を元にロボットを移動
        self.sim.reset()
        self.robot_x,self.robot_y,self.robot_angle = self.sim.simulation_twewheel([1,0.1,1,0.1])
        print("reset:x,y,theta->",self.robot_x,self.robot_y,self.robot_angle)
        # 初期状態の計算
        distance_to_goal = self._calculate_distance(self.goal_x,self.goal_y,self.robot_x,self.robot_y)
        angle_diff = self._calculate_angle_diff(self.goal_x,self.goal_y,self.robot_x,self.robot_y,self.robot_angle)
        obstacle_detection,distance_to_obstacle,angle_to_obstacle=self._get_obstacle(self.obstacles,self.robot_x,self.robot_y)
        observation = np.array([(self.robot_x),(self.robot_y),(self.robot_angle),(angle_diff),(distance_to_goal),(angle_to_obstacle),(distance_to_obstacle),(obstacle_detection)], dtype=np.float32)
        return observation, {}

    def step(self, action):
        terminated=False
        reward=0
        #遺伝子を元にロボットを移動
        gene=self.actions[action]
        repeat=round(gene[1]/0.1)
        print(f"{repeat}回繰り返し")
        correct_action=[gene[0],0.1,gene[2],0.1]

        for i in range(repeat):
            self.robot_x,self.robot_y,self.robot_angle = self.sim.simulation_twewheel(correct_action)
            obstacle_detection,distance_to_obstacle,angle_to_obstacle=self._get_obstacle(self.obstacles,self.robot_x,self.robot_y)
            #障害物に衝突した場合 
            if distance_to_obstacle < 0.5:
                print("衝突！！")
                reward -= 200
                terminated =True
                self.collision=1
                self.step_p=0
                break

        # ゴールとの距離と角度差を取得
        distance_to_goal = self._calculate_distance(self.goal_x,self.goal_y,self.robot_x,self.robot_y)
        angle_to_goal = self._calculate_angle_diff(self.goal_x,self.goal_y,self.robot_x,self.robot_y,self.robot_angle)
        
        print("ゴールまで:",distance_to_goal)
        print("ゴールとの角度差:",angle_to_goal)
        print("障害物まで:",distance_to_obstacle)
        print("障害物との角度差:",angle_to_obstacle)
        #報酬設定
        #if abs(self.goal_x - self.robot_x) < abs(self.goal_x - self.old_x) or abs(self.goal_y - self.robot_y) < abs(self.goal_y - self.old_y):
        #    reward = 1   
        

        reward -= abs(angle_to_goal)*0.1+distance_to_goal
       
        self.old_distance=distance_to_goal
        self.old_x=self.robot_x
        self.old_y=self.robot_y
        # 終了条件
        if bool(distance_to_goal < 0.5) :
            reward += 100
            terminated =True
            self.step_p=0

        if obstacle_detection == True:
            self.detect_flag=True
        #
        if self.detect_flag == True and obstacle_detection == False:
            reward += 10
        #     self.detect_flag=False
        if abs(self.old_x-self.robot_x) <0.1 and abs(self.old_y-self.robot_y) <0.1:
            reward -=10
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
       # print("angle_diff:",angle_diff)
        print("reward:",reward)
        self.step_p += 1 
        return np.array([(self.robot_x),(self.robot_y),(self.robot_angle),(angle_to_goal),(distance_to_goal),(angle_to_obstacle),(distance_to_obstacle),(obstacle_detection)], dtype=np.float32), reward, terminated, False, {}

    def _calculate_distance(self,target_x,target_y,robot_x,robot_y):
        #対象との距離
        distance_to_goal = np.sqrt((target_x - robot_x) ** 2 + (target_y - robot_y) ** 2)
        #print("distance_to_target:",distance_to_goal)
        return distance_to_goal

    def _calculate_angle_diff(self,target_x,target_y,robot_x,robot_y,robot_angle):
        # ロボットから見た対象の角度を計算
        dx_target = target_x - robot_x
        dy_target = target_y - robot_y
        target_angle = math.degrees(math.atan2(dy_target, dx_target))  #度数法に変換
        #print("target_angle (degrees):", target_angle)
        # ロボットの向いている方向と対象との角度差を計算
        angle_diff = target_angle - robot_angle
        return angle_diff
    
    def _get_obstacle(self,obstacles,robot_x,robot_y):
        obstacle_detection=False
        distance_to_obstacle=100
        angle_to_obstacle=180
        for obs in obstacles:
            distance_to_obstacle= self._calculate_distance(obs[0],obs[1],self.robot_x,self.robot_y)
            angle_to_obstacle=self._calculate_angle_diff(obs[0],obs[1],self.robot_x,self.robot_y,self.robot_angle)
            if distance_to_obstacle < 2 :#距離が0.5以内
                print("obstacle detection!!")
                obstacle_detection=True
                break
            
        return obstacle_detection,distance_to_obstacle,angle_to_obstacle
