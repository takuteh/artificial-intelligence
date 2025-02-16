import sys
sys.path.append("..")
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm
import json
import robot_simulator
import sys
sys.path.append("custom_env")
import goal_only_env


def gym_env_step(env, action):
    obs, reward, terminated, truncated, info = env.step(action)
    return obs, reward, bool(terminated), bool(truncated), info

class EpsGreedyQPolicy:
    def __init__(self, eps=0.1):
        self.eps = eps

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        if np.random.uniform() < self.eps:
            return np.random.randint(0, nb_actions)
        else:
            return np.argmax(q_values)

class TestAgent:
    def __init__(self, model, policy, observation):
        self.model = model
        self.policy = policy
        self.observation = observation

    def act(self):
        q_values = tf.squeeze(self.model(np.array([self.observation])))
        action_id = self.policy.select_action(q_values=q_values)
        return action_id

    def observe(self, observation):
        self.observation = observation

def load_action_list(filename="action_list.json"):
    with open(filename, "r") as file:
        actions = json.load(file)
    return actions

# 環境の作成
env = goal_only_env.RobotEnv()
env.reset(seed=123)
env.goal_x=9
env.goal_y=9

# モデルのロード
model = tf.keras.models.load_model("goal_only_model.h5")

# 初期観測の取得
initial_observation = env.reset()[0]
policy = EpsGreedyQPolicy(eps=0.0)  # 学習なしなので完全なグリーディー方策

agent = TestAgent(model=model, policy=policy, observation=initial_observation)

# 実行エピソード数
nb_episodes = 1
step_history = []

sim=robot_simulator.Simulate(anime=True)
actions=load_action_list()
sim.simulation_twewheel(data=[1,0.1,1,0.1],ini_state=[0,0,0],factor=1,td=6.36)
with tqdm.trange(nb_episodes) as t:
    for episode in t:
        observation, _ = env.reset()
        agent.observe(observation)
        done = False
        step = 0
        episode_reward_history = []
        #env.obstacles=[(-2,4)]
        
        while not done:
            action = agent.act()
            sim.simulation_twewheel(data=actions[action],ini_state=[0,0,0],factor=1,td=6.36)
            observation, reward, terminated, truncated, info = gym_env_step(env, action)
            done = terminated or truncated
            step += 1
            episode_reward_history.append(reward)
            agent.observe(observation)

            if done:
                t.set_description(f'Episode {episode}: {step} steps')
                t.set_postfix(episode_reward=np.sum(episode_reward_history))
                step_history.append(step)
                break

env.close()
plt.plot(env.goal_x,env.goal_y, 'o', markersize=10, color='blue')  # ゴールをプロット
for obs in env.obstacles:
   plt.plot(obs[0],obs[1], 'o', markersize=10, color='green')  # ゴールをプロット
sim.exec_animation()