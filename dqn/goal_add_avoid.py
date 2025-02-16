import numpy as np
import tensorflow as tf
import copy
import tqdm
import gym
import matplotlib.pyplot as plt
import random
from collections import deque
import sys
sys.path.append("custom_env")
import goal_add_avoid_env
import random

class EpsGreedyQPolicy:#εグリーディー法
    def __init__(self, eps=0.1, eps_decay_rate=0.99, min_eps=0.1):
        self.eps = eps
        self.eps_decay_rate = eps_decay_rate
        self.min_eps = min_eps

    def select_action(self, q_values, is_training=True):
        #各行動のQ値が格納された配列
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]#行動数をリストの要素数から取得

        if is_training:#トレーニング中はランダム行動も取る
            if np.random.uniform() < self.eps:
                action = np.random.randint(0, nb_actions)
            else:
                action = np.argmax(q_values)
        else:
            action = np.argmax(q_values)

        return action

    def decay_eps_rate(self):
        self.eps = max(self.eps * self.eps_decay_rate, self.min_eps)


class RandomMemory:
    def __init__(self, limit):#経験メモリのキューを作成，最大数はlimit
        self.experiences = deque(maxlen=limit)

    def sample(self, batch_size):#経験をランダムにサンプリング
        assert batch_size > 0
        batch_size = min(batch_size, len(self.experiences))
        mini_batch = random.sample(self.experiences, batch_size)

        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = [], [], [], [], []
        for state, action, reward, next_state, done in mini_batch:
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            terminal_batch.append(0. if done else 1.)

        return (
            np.array(state_batch),
            np.array(action_batch),
            np.array(reward_batch),
            np.array(next_state_batch),
            np.array(terminal_batch),
        )

    def append(self, state, action, reward, next_state, terminal=False):#新しい経験を追加(FIFO方式)
        self.experiences.append((state, action, reward, next_state, terminal))


class DQNAgent:
    def __init__(self, training=True, policy=None, epochs=32, gamma=0.99, actions=None,
                 memory=None, memory_interval=1, model=None, target_model=None, update_interval=100,
                 train_interval=1, batch_size=32, warmup_steps=200, observation=None, loss_fn=None,
                 optimizer=None, is_ddqn=False):
        self.training = training
        self.policy = policy
        self.actions = actions
        self.gamma = gamma
        self.observation = observation
        self.prev_observation = None
        self.memory = memory
        self.memory_interval = memory_interval
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.model = model
        self.target_model = target_model
        self.train_interval = train_interval
        self.update_interval = update_interval
        self.is_ddqn = is_ddqn
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.step = 0

    def act(self):#QネットワークからQ値を予想し，行動を選択
        q_values = tf.squeeze(self.target_model(np.array([self.observation])))
        #εグリーディー法で行動を選択する
        action_id = self.policy.select_action(q_values=q_values, is_training=self.training)
        self.recent_action_id = action_id
        #ランダム行動の確率を下げる
        self.policy.decay_eps_rate()
        return action_id

    def observe(self, observation, reward=None, is_terminal=None):#状態と報酬の記録
        self.prev_observation = copy.deepcopy(self.observation)
        self.observation = observation
        if self.training and reward is not None:
            self.memory.append(self.prev_observation, self.recent_action_id, reward, observation, terminal=is_terminal)
        self.step += 1

    def train(self):#学習
        for _ in range(self.epochs):
            self._experience_replay()#経験再生

    def update_target_hard(self):#ターゲットネットワークの更新
        self.target_model.set_weights(self.model.get_weights())

    def _experience_replay(self):#経験再生
        #経験メモリから，状態や報酬を取得
        state0_batch, action_batch, reward_batch, state1_batch, terminal_batch = self.memory.sample(self.batch_size)
        reward_batch = reward_batch.reshape(-1, 1)
        terminal_batch = terminal_batch.reshape(-1, 1)

        #ターゲットQ値を計算（教師データ）
        target_q_values = self.target_model(state1_batch)
        discounted_reward_batch = self.gamma * target_q_values * terminal_batch
        targets = reward_batch + discounted_reward_batch
        targets_one_hot = np.zeros((len(targets), len(self.actions)))

        for idx, action in enumerate(action_batch):
            targets_one_hot[idx][action] = max(targets[idx])

        mask = tf.one_hot(action_batch, len(self.actions))
        state0_batch = tf.convert_to_tensor(state0_batch)
        
        #学習モデルがQ値を計算し，ターゲットQ値との誤差からネットワークを更新
        self._train_on_batch(state0_batch, mask, targets_one_hot)

    @tf.function
    def _train_on_batch(self, states, masks, targets):
        with tf.GradientTape() as tape:
            y_preds = self.model(states)
            y_preds = tf.math.multiply(y_preds, masks)
            loss_value = self.loss_fn(targets, y_preds)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))


def build_q_network(input_shape, nb_output):#ニューラルネットワークを構築
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(input_layer)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(nb_output, activation='linear')(x)
    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

def gym_env_step(env, action):
    obs, reward, terminated, truncated, info = env.step(action)
    return obs, reward, bool(terminated), bool(truncated), info

#env = gym.make('CartPole-v1')
env = goal_add_avoid_env.RobotEnv()
env.reset(seed=123)
nb_actions = env.action_space.n#行動の種類
actions = np.arange(nb_actions)#その状態で取れる行動
policy = EpsGreedyQPolicy(eps=1., eps_decay_rate=0.999, min_eps=0.01)#εグリーディー法の設定
memory = RandomMemory(limit=50000)#経験メモリのサンプル数
initial_observation = env.reset()[0]
loss_fn = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam()
model = build_q_network(input_shape=[len(initial_observation)], nb_output=nb_actions)#学習モデル
target_model = build_q_network(input_shape=[len(initial_observation)], nb_output=nb_actions)#ターゲットネットワーク

agent = DQNAgent(actions=actions, memory=memory, update_interval=200, train_interval=1, batch_size=32,
                 observation=initial_observation, model=model, target_model=target_model, policy=policy,
                 loss_fn=loss_fn, optimizer=optimizer, is_ddqn=False)

step_history = []

#実行エピソード数
nb_episodes = 100

# for i in range(20):
#     obs_x=random.randint(-10,10)
#     obs_y=random.randint(-10,10)
#     env.obstacles.append((obs_x,obs_y))
# print(env.obstacles)  

# env.obstacles=[(-1, -1), (-2, -4), (8, -9), (9, -6), (-8, 5), (-5, -10), (5, 6), (-5, 3), (2, 4), (3, -8), (-7, 0),(8,0)]
with tqdm.trange(nb_episodes) as t:
    for episode in t:#環境リセット
        observation, _ = env.reset()
        agent.observe(observation)
        done = False
        step = 0
        episode_reward_history = []
        env.goal_x=random.randint(-10,10)
        env.goal_y=random.randint(-10,10)
        #env.goal_angle=90#random.randint(-180,180)
        #env.goal_x=5
        #env.goal_y=5
        print(f"goal:({env.goal_x,env.goal_y})")
        while not done:#1エピソードのループ
            #DQNから次の行動を取得
            action = agent.act()
            #環境に行動を入力し，報酬や状態等を取得
            observation, reward, terminated, truncated, info = gym_env_step(env, action)
            #エピソード終了フラグを格納
            done = terminated or truncated
            step += 1
            episode_reward_history.append(reward)
            #DQNに状態と報酬を入力，この段階ではメモリに記録するだけ
            agent.observe(observation, reward, done)

            if done:#エピソード終了時
                t.set_description(f'Episode {episode}: {step} steps')
                t.set_postfix(episode_reward=np.sum(episode_reward_history))
                #学習モデルのトレーニング
                agent.train()
                if episode % 5 == 0: #５エピソードに１回ターゲットネットワークに学習モデルの重みをコピー
                    agent.update_target_hard()
                #step_history.append(step)#各エピソードのステップ数を記録
                step_history.append(env.collision)
                #ステップ数が少ないほどいい
                #座標が10*10を超えたら強制終了
                break

env.close()
model.save("goal_add_avoid_model.h5")
x = np.arange(len(step_history))
plt.ylabel('step')
plt.xlabel('episode')
plt.plot(x, step_history)
plt.savefig('result.png')
#print(env.obstacles)