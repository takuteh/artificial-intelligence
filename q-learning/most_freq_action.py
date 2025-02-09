import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
filename = "episode_log.csv"  # ファイル名を適宜変更してください
data = pd.read_csv(filename)

# 各Stateごとに最も多く選ばれたActionを計算
most_common_action = (
    data.groupby('State')['Action']
    .apply(lambda x: x.value_counts().idxmax())  # 各Stateごとに最頻のActionを取得
    .reset_index(name='Most Frequent Action')   # 結果をデータフレームに変換
)

# グラフ作成
plt.figure(figsize=(12, 6))
plt.bar(most_common_action['State'], most_common_action['Most Frequent Action'], color='skyblue', edgecolor='black')

# グラフの装飾
plt.title('Most Common Action for Each State', fontsize=16)
plt.xlabel('State', fontsize=12)
plt.ylabel('Most Frequent Action', fontsize=12)
plt.xticks(most_common_action['State'], fontsize=10, rotation=45)  # 横軸を見やすく回転
plt.yticks([0,1,2,3], fontsize=10)  # Actionは1～4に限定
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# グラフ表示
plt.show()
