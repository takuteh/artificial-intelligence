import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルの読み込み
file_path = "episode_log.csv"  # CSVファイルのパスを指定してください
data = pd.read_csv(file_path)

# エピソードごとにアクションの割合を計算
data['Batch'] = (data['Episode'] - 1) // 100  # 100エピソードごとにバッチ分け
action_counts = data.groupby(['Batch', 'Action']).size().unstack(fill_value=0)

# 割合を計算
action_ratios = action_counts.div(action_counts.sum(axis=1), axis=0)

# グラフのプロット
action_ratios.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
plt.title('Action Ratios per 100 Episodes')
plt.xlabel('Batch (100 Episodes Each)')
plt.ylabel('Action Ratio')
plt.legend(title="Action", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
