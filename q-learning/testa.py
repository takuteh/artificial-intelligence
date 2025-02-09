import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
import csv

GRID_WIDTH = 5
GRID_HEIGHT = 8
# 定数
ALPHA = 0.1  # 学習率
GAMMA = 0.9  # 割引率
EPSILON = 0.1  # ε-greedy法の探索率
MAX_STEPS = 2000  # 最大ステップ数
NUM_EPISODES = 500  # エピソード数

# 状態数（9パターン）
NUM_STATES = 6  # 状態数（センサー情報による）
# 方策数（4パターン）
NUM_ACTIONS = 2  # 行動数（音認識＋音方策など）

CSV="episode_log.csv"

class Test:
    def __init__(self) -> None:
        # ロボットの初期位置 (y, x) => (縦, 横)
        self.robot_position = [5,3]
        # 光源の位置
        self.light_source = [3, 2]  # (y, x)
        # 音源の位置
        self.sound_sources = [(2, 1), (2, 3), (5, 2),(7,0),(7,4)]  # [(y, x), (y, x), ...]
         # 光源・音源の全座標を統合
        self.blocked_positions = set(self.light_source + self.sound_sources)
        # 音源方向を格納するリストを初期化
        self.sound_poss = []
        self.light_poss = []
        self.state=None
        # Qテーブルの初期化
        self.Q_table = np.zeros((NUM_STATES,NUM_ACTIONS)) #q_tableを乱数で初期化 #np.zeros((NUM_STATES, NUM_ACTIONS))

        self.previous_state=None
        self.previous_action=None
        self.state = random.randint(0, NUM_STATES - 1)  # 初期状態の設定
        self.total_reward = 0
        self.next = "nodata"
        self.episode=0


    def generate_valid_positions(self,blocked_positions):
        """許容される全ての座標を生成"""
        valid_positions = [
            (x, y)
            for x in range(0, GRID_HEIGHT )
            for y in range(0, GRID_WIDTH )
            if (x, y) not in blocked_positions
        ]
        return valid_positions


    # 光認識モジュール
    def light_recognition(self, position):
        light_y, light_x = self.light_source
        if abs(light_y - position[0]) + abs(light_x - position[1]) <= 2:
            if light_x > position[1]:
                lr = "right"
            elif light_x == position[1]:
                lr = "center"
            else:
                lr = "left"

            if light_y < position[0]:
                ud = "up"
            elif light_y == position[0]:
                ud = "center"
            else:
                ud = "down"

            if lr == "right" and ud == "up":
                self.light_poss.append("up_right")
                return 11
            elif lr == "right" and ud == "center":
                self.light_poss.append("right")
                return 12
            elif lr == "right" and ud == "down":
                self.light_poss.append("down_right")
                return 13
            elif lr == "center" and ud == "up":
                self.light_poss.append("up")
                return 14
            elif lr == "center" and ud == "center":
                self.light_poss.append("clear")
                return 19
            elif lr == "center" and ud == "down":
                self.light_poss.append("down")
                return 15
            elif lr == "left" and ud == "up":
                self.light_poss.append("up_left")
                return 16
            elif lr == "left" and ud == "center":
                self.light_poss.append("left")
                return 17
            elif lr == "left" and ud == "down":
                self.light_poss.append("down_left")
                return 18
        else:
            return 10
            
    # 音認識モジュール
    def sound_recognition(self, position):
        for sound_pos in self.sound_sources:
            sound_y, sound_x = sound_pos
            if abs(sound_y - position[0]) + abs(sound_x - position[1]) <= 1:
                if sound_x > position[1]:
                    lr = "right"
                elif sound_x == position[1]:
                    lr = "center"
                else:
                    lr = "left"

                if sound_y < position[0]:
                    ud = "up"
                elif sound_y == position[0]:
                    ud = "center"
                else:
                    ud = "down"

                if lr == "right" and ud == "center":
                    self.sound_poss.append("right")
                    #return "right"
                elif lr == "center" and ud == "up":
                    self.sound_poss.append("up")
                    #return "up"
                elif lr == "center" and ud == "center":
                    self.sound_poss.append("clear")
                elif lr == "center" and ud == "down":
                    self.sound_poss.append("down")
                    #return "down"
                elif lr == "left" and ud == "center":
                    self.sound_poss.append("left")
                    #return "left"      

    # 光方策（接近モジュール）
    def light_move(self, light_poss):
        next_move = ["up", "down", "right", "left"]
        diagonal=["up_right","up_left","down_light","down_left"]

        if "up" in light_poss or "down" in light_poss or "left" in light_poss or "right" in light_poss:  # 上下左右が存在する場合，反応がある方向をランダムで出力
            remaining_moves = [move for move in light_poss if move not in diagonal]#斜め方向は取り除く
            print(remaining_moves, "に反応あり")
            print("接近します")
            return random.choice(remaining_moves)
        elif light_poss == []:  # 反応無しの場合
            print("反応無し")
            return random.choice(next_move)
        elif diagonal in light_poss:  # 斜めしか存在しない場合，どちらかランダムで選択
            print(light_poss, "に反応あり")
            print("接近します")
            remaining_moves = [part for item in light_poss for part in item.split('_')]
            return random.choice(remaining_moves)
        
    # 音方策（退避モジュール）
    def sound_move(self, sound_poss):
        next_move = ["up", "down", "right", "left"]
        diagonal=["up_right","up_left","down_light","down_left"]

        sound_poss = [move for move in sound_poss if move not in diagonal]#斜め方向は取り除く
        if sound_poss == []:  # 反応無しの場合
            print("反応無し")
            return random.choice(next_move)
        else:  # 反応がある方向以外をランダムで出力
            print(sound_poss, "に反応あり")
            print("退避します")
            remaining_moves = [move for move in next_move if move not in sound_poss]#sound_possに含まれる物は取り除く
            return random.choice(remaining_moves)

    # ロボットの行動
    def move_robot(self, next, position):
        new_position = [None] * 2
        new_position[0], new_position[1] = position
        if next == "up":
            print("上に移動")
            new_position[0] -= 1
        elif next == "down":
            print("下に移動")
            new_position[0] += 1
        elif next == "right":
            print("右に移動")
            new_position[1] += 1
        elif next == "left":
            print("左に移動")
            new_position[1] -= 1

        # グリッド範囲を超えないよう制限
        new_position[0] = max(0, min(GRID_HEIGHT - 1, new_position[0]))
        new_position[1] = max(0, min(GRID_WIDTH - 1, new_position[1]))

        return new_position


    # グリッド描画
    def draw_grid(self, step):
        # カスタムカラーマップを作成
        custom_cmap = ListedColormap(["white", "yellow", "green","red"])  # 色順に背景, 光源, 音源
        
        # グリッド初期化
        grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        grid[self.robot_position[0], self.robot_position[1]] = 3  # 光源
        grid[self.light_source[0], self.light_source[1]] = 1  # 光源
        for ss in self.sound_sources:
            grid[ss[0], ss[1]] = 2  # 音源

        # グリッドのカラーマップ表示
        plt.imshow(grid, cmap=custom_cmap, extent=[0, GRID_WIDTH, 0, GRID_HEIGHT], vmin=0, vmax=3)
        
        # 軸の設定
        plt.xticks(range(GRID_WIDTH))
        plt.yticks(range(GRID_HEIGHT))
        plt.grid(which="both", color="gray", linestyle="--", linewidth=0.5)
        
        # タイトルやラベルの設定
        plt.title(f"グリッド環境 - Step {step}")
        plt.xlabel("X")
        plt.ylabel("Y")

        # 表示
        plt.show()

    def get_sound_state(self):
        # 状態番号を決定するマッピング
        if "clear" in self.sound_poss:
            return 32  # 障害物に衝突
        
        if "up" in self.sound_poss and "down" in self.sound_poss and "left" in self.sound_poss and "right" in self.sound_poss:
            return 21  # 上下左右全て
        elif "up" in self.sound_poss and "down" in self.sound_poss:
            return 22  # 上下
        elif "left" in self.sound_poss and "right" in self.sound_poss:
            return 23  # 左右
        elif "up" in self.sound_poss and "right" in self.sound_poss:
            return 24  # 上右
        elif "up" in self.sound_poss and "left" in self.sound_poss:
            return 25  # 上左
        elif "down" in self.sound_poss and "right" in self.sound_poss:
            return 26  # 下右
        elif "down" in self.sound_poss and "left" in self.sound_poss:
            return 27  # 下左
        elif "right" in self.sound_poss:
            return 28  # 右
        elif "left" in self.sound_poss:
            return 29  # 左
        elif "up" in self.sound_poss:
            return 30  # 上
        elif "down" in self.sound_poss:
            return 31  # 下
        else:
            return 20  # 音源なし
    def get_all_state(self,light_state,sound_state):
        state=None
        if light_state==10 and sound_state==20:#光反応なし・音反応なし
            state=0
            print("光・音反応なし")
        elif light_state==10 and (sound_state >= 21 and sound_state <= 31):#光反応なし・音反応あり
            state=1
            print("光反応なし・音反応あり")
        elif (light_state >=11 and light_state <=18) and (sound_state >= 21 and sound_state <= 31):#光反応あり・音反応あり
            state=2
            print("光反応あり・音反応あり")
        elif (light_state >=11 and light_state <=18) and sound_state == 20:#光反応あり・音反応なし
            state=3
            print("光反応あり・音反応なし")
        elif light_state==19:#ゴール
            state=4
        elif sound_state==32:#衝突
            state==5

        return state

    def choose_action(self,state):
        """ε-greedy法で行動を選択"""
        self.a=None
        #徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.5 * (1 / (self.episode + 1))
        if random.uniform(0, 1) < epsilon:
            # ランダム行動
            self.a="random"
            return random.randint(0, NUM_ACTIONS-1)
            
        else:
            # Q値が最大の行動
            self.a="Q"
            return np.argmax(self.Q_table[state])
            
    def update_q_table(self,state, action, reward, next_state):
        """Qテーブルを更新"""
        #max_next_q = np.max(self.Q_table[next_state])
        #self.Q_table[state, action] += ALPHA * (reward + GAMMA * max_next_q - self.Q_table[state, action])
        q = self.Q_table[state][action]  # Q(s, a)
        max_q = max(self.Q_table[state])  # max Q(s')
        # Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        self.Q_table[state][action] = q + (ALPHA * (reward + (GAMMA * max_q) - q))
    # シミュレーションの実行
    def simulate(self, step):
        print("------------------------------------------")
        state=None
        self.light_poss = []
        self.sound_poss = []  # リセット
        
        #現在の状態を取得
        light_state=self.light_recognition(self.robot_position)#光認識
        self.sound_recognition(self.robot_position)#音認識
        sound_state=self.get_sound_state()
        
        current_state=self.get_all_state(light_state,sound_state)

        #現在の状態をもとに方策を選択
        action = self.choose_action(current_state)
        
        #選択された方策を実行し，移動する方向を決定する
        if action==0:#光光
            print("光方策")
            self.next = self.light_move(self.light_poss)
        elif action==1:#光音
            print("音方策")
            self.next = self.sound_move(self.sound_poss)

        #ロボットを動かす
        self.robot_position = self.move_robot(self.next, self.robot_position)
        reward=-1
        result="running"
        end=0
        #行動の結果，ゴールに到達したか障害物に衝突してないか判定
        a=self.light_recognition(self.robot_position)
        self.sound_recognition(self.robot_position)
        b=self.get_sound_state()
        if a== 19:#ゴールに到達したら
            print("Goal !!")
            reward=10
            result="success"
            end=1
        if  b == 32:#障害物に衝突したら
            print("Collision !!!")
            reward=-50
            result="fault"
            end=1
        #行動前の状態を過去の状態とする
        self.previous_state=current_state
        self.previous_action=action

        #選択した方策を実行した現在の状態
        new_state=self.get_all_state(a,b)
        #self.draw_grid(step)
        
        # Qテーブルの更新
        #if self.previous_state is not None:
        print("q->旧状態:",self.previous_state,"行動:",self.previous_action,"報酬:",reward,"新状態:",new_state)
        self.update_q_table(self.previous_state, self.previous_action, reward, new_state)

        return end,result#step終了
if __name__ == '__main__':
    test = Test()
    with open(CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(['Episode', 'Step', 'State','Action', 'Result'])  # ヘッダー行
        writer.writerow(['Episode','Result'])  # ヘッダー行
    # 学習
    for test.episode in range(NUM_EPISODES):
        step=0
        test.robot_position=random.choice(test.generate_valid_positions(test.blocked_positions))
        print(test.robot_position)
        while True:
            step += 1
            fb,result = test.simulate(step)
            with open(CSV, mode='a', newline='') as file:  # 追記モード ('a')
                writer = csv.writer(file)
                #writer.writerow([test.episode+1, step, test.previous_state,test.previous_action, result])
                writer.writerow([test.episode+1,result])
            if fb == 1:
                break
