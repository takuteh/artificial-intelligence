import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
filename = "episode_log.csv"  # 適宜ファイル名を変更してください
data = pd.read_csv(filename)

# 100エピソードごとにグループ化するための新しい列を作成
data['EpisodeGroup'] = (data['Episode'] - 1) // 100

# 100エピソードごとにActionが2である割合を計算
grouped = data.groupby('EpisodeGroup')['Action']
action_2_count = grouped.apply(lambda x: (x == 0).sum())
total_actions = grouped.count()

# 割合を計算
action_2_ratio = action_2_count / total_actions

# グラフ作成
plt.figure(figsize=(12, 6))

# 割合をプロット
plt.plot(action_2_ratio.index * 100, action_2_ratio.values, color='blue', marker='o', label='Action 2 Ratio per 100 Episodes')

# グラフの装飾
plt.title('Ratio of Action 2 Every 100 Episodes', fontsize=16)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Action 2 Ratio', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# グラフ表示
plt.show()
