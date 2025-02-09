import numpy as np
from math import cos, sin, pi
import math
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import time

class Simulate():
    def __init__(self):
        #アニメーショングラフ描画のため
        self.fig = plt.figure()
        self.ims=[]
        self.state=[0,0,0]
        #計算データ（座標）の格納
        self.st_x = []
        self.st_y = []
        self.st_theta = []
 
        self.st_vec=[0,0,0]
    def reset(self):
        #計算データ（座標）の格納
        self.st_x = []
        self.st_y = []
        self.st_theta = []
        self.st_vec=[0,0,0]
    def twe_wheel_fuc(self,v, state, delta, factor=1, td=2):
        """
        Equation of state
        Args:
            v (tuple or list): velocity of each wheel unit m/s,(right velocity,left velocity)
            state (list):　state [x, y, thita] , x, y 
            delta (float): update time unit s
            factor (float):velocity factor Defaults to 1
            td (float): tread length between wheel unit m Defaults to 2.

        Returns:
            [list]: next state
        """
        # vr: right wheel velocity, vl:left wheel velocity
        # vel: Center of gravity velocity
        # omega: Rotation angular velocity
        speed_r=1
        speed_l=1
        exe_time_r=v[1]
        exe_time_l=v[3]
        vr = v[0] * speed_r * exe_time_r #factor
        vl = v[2] * speed_l * exe_time_l #factor
        vel = (vr + vl)/2
        omega = (vr - vl)/(td)

        x_ = vel*delta*cos(state[2]+omega*delta/2)
        y_ = vel*delta*sin(state[2]+omega*delta/2)

        xt = state[0] + x_
        yt = state[1] + y_
        thetat = state[2]+omega*delta
        # thetatの更新
        # thetat の更新と正規化 (度数法)
        # ラジアンでの角度更新と正規化
        thetat = (thetat+pi) % (2 * pi)-pi
        #if thetat > math.pi:
        #    thetat -= 2 * math.pi
        print(thetat)
        update_state = [xt, yt, thetat]
        return update_state


    def simulation_twewheel(self,data,ini_state=[0,0,0],factor=1,td=6.36,anime=False):
        """
        data: list On/OFF data
        
        """
        self.st_vec = self.twe_wheel_fuc(data, self.st_vec, delta=1,factor=factor,td=td)
        xt, yt, thetat = self.st_vec
        #print("step:",step)
        print("State:",self.st_vec)
        print("Direction angle: ",math.degrees(thetat))
        self.st_x.append(xt)
        self.st_y.append(yt)
        self.st_theta.append(thetat)
    
        # ロボットの向きに基づいて dx, dy を計算
        dx = np.cos(thetat) * 0.5  # 矢印の長さ
        dy = np.sin(thetat) * 0.5
        
        if anime:
            #Plotのための設定
            plt.grid(True)
            plt.axis("equal")
            plt.xlabel("X")
            plt.ylabel("Y")

            im=plt.plot(xt,yt,'o',self.st_x,self.st_y, '--', color='red',markersize=10, linewidth = 2)        
            self.ims.append(im)

        return xt,yt,math.degrees(thetat)

    def exec_animation(self):
        # アニメーション作成
        anim = ArtistAnimation(self.fig, self.ims, interval=200, blit=True,repeat=False) 
        plt.show()

if __name__ == '__main__':
    sim=Simulate()
    #スイッチON/OFFとして速度は一定とする。正回転：1、逆回転：-1、停止：0
    #(右方向，右秒数，左方向，左秒数)

    #一つの遺伝子（右車輪回転方向,右車輪回転秒数,左車輪回転方向,左車輪回転秒数）
    input_lists =[1,2.0,0,0.1]
    
    input_lists2 =[(1,1,1,1),(1,5,-1,5),(1,1,1,1),(1,1,1,1),(1,1,1,1),(1,5,-1,5),(1,5,-1,5),(1,5,-1,5),(1,5,-1,5),(1,5,-1,5)]
    
    #ゴールのx座標とロボットのx座標を比較し，ロボットの方が大きくなるまで繰り返し
    goal_x, goal_y = 5,3
    plt.plot(goal_x, goal_y, 'o', markersize=10, color='blue')  # ゴールをプロット
    #y座標を比較，差をGAに渡す
    x,y,thetat=sim.simulation_twewheel([1,0.1,1,0.1],anime=True)
    for i in input_lists2:
        x,y,thetat=sim.simulation_twewheel(i,anime=True)
        print(f"x:{x},y:{y},theta:{thetat}")
        # ゴール方向の角度を計算
        dx_goal = goal_x - x
        dy_goal = goal_y - y
        theta_goal = math.degrees(math.atan2(dy_goal, dx_goal))  # ゴール方向の角度を度数法に変換
        print("goal_angle (degrees):", theta_goal)
        # if angle_diff > 180:
        #     angle_diff -= 360
        # elif angle_diff < -180:
        #     angle_diff += 360
        print("angle_diff:",theta_goal-thetat)
    sim.exec_animation()