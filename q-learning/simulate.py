import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap

GRID_WIDTH = 8
GRID_HEIGHT = 10

class Test:
    def __init__(self) -> None:
        # ロボットの初期位置 (y, x) => (縦, 横)
        self.robot_position = [2, 2]
        # 光源の位置
        self.light_source = [1, 4]  # (y, x)
        # 音源の位置
        self.sound_sources = [(2, 1), (2, 3), (5, 2),(8,3),(4,6),(6,7)]  # [(y, x), (y, x), ...]
        # 音源方向を格納するリストを初期化
        self.sound_poss = []

    # 光認識モジュール
    def light_recognition(self, position):
        print("光認識モジュールが選択")
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
                return "up_right"
            elif lr == "right" and ud == "center":
                return "right"
            elif lr == "right" and ud == "down":
                return "down_right"
            elif lr == "center" and ud == "up":
                return "up"
            elif lr == "center" and ud == "center":
                return "clear"
            elif lr == "center" and ud == "down":
                return "down"
            elif lr == "left" and ud == "up":
                return "up_left"
            elif lr == "left" and ud == "center":
                return "left"
            elif lr == "left" and ud == "down":
                return "down_left"
        else:
            return "nodata"
            
    # 音認識モジュール
    def sound_recognition(self, position):
        print("音認識モジュールが選択")
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
                #elif lr == "center" and ud == "center":
                    #return "clear"
                elif lr == "center" and ud == "down":
                    self.sound_poss.append("down")
                    #return "down"
                elif lr == "left" and ud == "center":
                    self.sound_poss.append("left")
                    #return "left"
        print(self.sound_poss)        

    # 光方策モジュール
    def light_move(self, light_pos):
        print("光方策モジュールが選択")
        next_move = ["up", "down", "right", "left"]
        if "_" in light_pos:  # 斜めの場合
            print(light_pos, "に反応あり")
            split_text = light_pos.split("_")
            return random.choice(split_text)
        elif light_pos == "nodata":  # 反応無しの場合
            print("反応無し")
            return random.choice(next_move)
        else:  # 上下左右の場合
            print(light_pos, "に反応あり")
            return light_pos
        
    # 音方策モジュール
    def sound_move(self, sound_poss):
        print("音方策モジュールが選択")
        next_move = ["up", "down", "right", "left"]
        if sound_poss == []:  # 反応無しの場合
            print("反応無し")
            return random.choice(next_move)
        else:  # 音源方向以外をランダムで出力
            print(self.sound_poss, "に音源あり")
            remaining_moves = [move for move in next_move if move not in self.sound_poss]
            return random.choice(remaining_moves)

    # ロボットの行動
    def move_robot(self, next, position):
        new_position = [None] * 2
        new_position[0], new_position[1] = position
        if next == "up":
            print("up")
            new_position[0] -= 1
        elif next == "down":
            print("down")
            new_position[0] += 1
        elif next == "right":
            print("right")
            new_position[1] += 1
        elif next == "left":
            print("left")
            new_position[1] -= 1

        # グリッド範囲を超えないよう制限
        new_position[0] = max(0, min(GRID_HEIGHT - 1, new_position[0]))
        new_position[1] = max(0, min(GRID_WIDTH - 1, new_position[1]))

        return new_position

    # シミュレーションの実行
    def simulate(self, steps=10):
        light_pos = "nodata"
        self.sound_poss = []  # リセット
        print("-----------------------------------------")
        print("光",self.light_recognition(self.robot_position))
        self.sound_recognition(self.robot_position)
 

        recognition_list = ["light", "sound"]
        select = random.choice(recognition_list)
        if select == "light":#光認識モジュールが選択
            light_pos = self.light_recognition(self.robot_position)
        else:#音認識モジュールが選択
            self.sound_recognition(self.robot_position)

        sensor_list = ["light", "sound"]
        select = random.choice(sensor_list)
        if select == "light":#光方策モジュール
            next = self.light_move(light_pos)
        else:#音方策モジュール
            next = self.sound_move(self.sound_poss)  # 修正

        self.robot_position = self.move_robot(next, self.robot_position)
        self.draw_grid(steps)

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

        # ロボットを丸で描画
        # plt.scatter(self.robot_position[1], GRID_HEIGHT - self.robot_position[0] - 0.5, 
        #             s=600, c="red", label="Robot")  # `s`はサイズ, `c`は色

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

if __name__ == '__main__':
    test = Test()
    for step in range(20):
        test.simulate(step)
