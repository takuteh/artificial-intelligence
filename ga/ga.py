import numpy as np
from math import cos, sin, pi
import math
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import time
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
#ここからGA
#-------------------------------------------------------------------------------------------

# パラメータ設定
NGEN = 300  # 世代数
POP_SIZE = 10  # 集団サイズ
CXPB = 0.7  # 交叉確率
MUTPB = 0.2  # 突然変異確率

# 遺伝子生成のカスタム関数
def generate_gene():
    gene1 = random.choice([-1, 0, 1])   #右車輪方向
    gene2 = round(random.uniform(0.1, 2.0), 1)#右車輪秒数
    gene3 = random.choice([-1, 0, 1])   #左車輪方向
    gene4 = round(random.uniform(0.1, 2.0), 1)#左車輪秒数
    return [gene1, gene2, gene3, gene4]

# 評価関数
def evaluate(individual):
    ini_x=0
    ini_y=0
    #ゴールのx座標とロボットのx座標を比較し，ロボットの方が大きくなるまで繰り返し
    #y座標を比較，差をGAに渡す
    x,y,step=simulation_twewheel(data=individual,ini_state=[ini_x,ini_y,0],factor=1,td=6.36)
    #fitness_score = (x-ini_x) - abs(y-ini_y) - step
    fitness_score = (y-ini_y) - abs(x-ini_x) - step  
 # 適当な目的関数例
    return fitness_score,  

# カスタム交叉関数
def custom_crossover(ind1, ind2):
    """遺伝子の要素の2番目と3番目で区切って交叉を行う"""
    right_ind1 = ind1[:2]#親1の右車輪の遺伝子
    left_ind1 = ind1[2:]#親1の左車輪の遺伝子
    right_ind2 = ind2[:2]#親2の右車輪の遺伝子
    left_ind2 = ind2[2:]#親2の左車輪の遺伝子

    ind1[:2], ind1[2:] = right_ind2,left_ind1
    ind2[:2], ind2[2:] = right_ind1,left_ind2
    return ind1, ind2


# 突然変異: すべての遺伝子要素にランダムな変異
def custom_mutate(ind):
    ind[0] = random.choice([-1, 0, 1])
    ind[1] = round(random.uniform(0.1, 2.0), 1)
    ind[2] = random.choice([-1, 0, 1])
    ind[3] = round(random.uniform(0.1, 2.0), 1)
    return ind,

def exec_ga():
    # 1. Fitness クラスの定義
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))  # 最大化問題
    creator.create("Individual", list, fitness=creator.FitnessMin)  # 個体クラスを定義

    toolbox = base.Toolbox()

    #関数等の設定
    toolbox.register("mutate", custom_mutate)#突然変異関数
    toolbox.register("mate", custom_crossover)#交叉関数
    toolbox.register("evaluate", evaluate)#評価関数
    toolbox.register("select", tools.selTournament, tournsize=3)# トーナメント方式

    # 初期個体生成
    toolbox.register("individual", tools.initIterate, creator.Individual, generate_gene)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # # 前回の最適解を引き継ぎたい場合
    # best_ind = creator.Individual([1, 2.0, 1, 2.0])  # best_indをcreator.Individualで作成
    # best_ind.fitness.values = (1.0,)  # best_indに適切なfitness値を設定

    # # 新しい集団に最良個体を追加
    # new_population = [best_ind]  # best_indを新しい集団の最初に追加

    # # 残りの個体を生成し、new_populationに追加
    # new_population.extend(toolbox.population(n=POP_SIZE - 1))  # ここで展開して追加

    # ログの初期化
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'avg', 'std', 'min', 'max']
    # # populationにnew_populationを設定
    # population = new_population
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    population = toolbox.population(n=POP_SIZE)

    # eaSimpleの実行
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, verbose=True)

    # 最良個体の表示
    best_ind = tools.selBest(population, 1)[0]
    print(f"最良の個体: {best_ind}")
    print(f"適応度: {best_ind.fitness.values[0]}")

    # 世代数、平均適応度、最小適応度、最大適応度のリストを抽出
    generations = logbook.select("gen")
    max_fitness = [np.array(val)[0] for val in logbook.select("max")]

    # グラフをプロット
    plt.figure(figsize=(10, 6))
    plt.plot(generations, max_fitness, label="Maximum Fitness", color="red", linestyle=":")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Over Generations")
    plt.legend()
    plt.grid()
    plt.show()
#-------------------------------------------------------------------------------------------
#ここまでGA

#シミュレータ
#-------------------------------------------------------------------------------------------
def twe_wheel_fuc(v, state, delta, factor=1, td=2):
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
    # state[2]: theta
    x_ = vel*delta*cos(state[2]+omega*delta/2)
    y_ = vel*delta*sin(state[2]+omega*delta/2)
    # x_ = vel*delta*cos(state[2])
    # y_ = vel*delta*sin(state[2])
    xt = state[0] + x_
    yt = state[1] + y_
    thetat = state[2]+omega*delta
    update_state = [xt, yt, thetat]
    return update_state


def simulation_twewheel(data,ini_state=[0,0,0],factor=1,td=6.36):
    """
    data: list On/OFF data
    
    """
    # simulation
    #アニメーショングラフ描画のため
    fig = plt.figure()
    ims = [] 
    #計算データ（座標）の格納
    st_x = []
    st_y = []
    st_theta = []
    st_vec = ini_state
    step = 0

    while True:   
        step += 1
        st_vec = twe_wheel_fuc(data, st_vec, delta=1,factor=factor,td=td)
        xt, yt, thetat = st_vec
        #print("State:",st_vec)
        #print("Direction angle: ",math.degrees(thetat))
        st_x.append(xt)
        st_y.append(yt)
        st_theta.append(thetat)

        #Plotのための設定
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("X")
        plt.ylabel("Y")
        
        #　時刻tにおける位置だけならば
        # im=plt.plot(xt,yt,'o', color='red',markersize=10, linewidth = 2)
        
        # 時刻tにおける位置と，時刻tに至るまでの軌跡の二つの絵を作成
        #plt.annotate('', xy=(xt+cos(thetat),yt+sin(thetat)), xytext=(xt,yt),
                    # arrowprops=dict(shrink=0, width=1, headwidth=2, 
                    # headlength=10, connectionstyle='arc3',
                    # facecolor='blue', edgecolor='blue'))
        im=plt.plot(xt,yt,'o',st_x,st_y, '--', color='red',markersize=10, linewidth = 2)
                    
        ims.append(im)
        if xt >= 12 or step >= 20:
            break
    # アニメーション作成
    # anim = ArtistAnimation(fig, ims, interval=200, blit=True,repeat=False) 
    # plt.show()
    return xt,yt,step
    # plt.pause(10)

if __name__ == '__main__':
    #y座標を比較，差をGAに渡す
    exec_ga()
