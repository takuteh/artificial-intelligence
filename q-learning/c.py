import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルの読み込み (State と Action のデータ)
# CSVファイルのフォーマットを想定
# State,Action
# 0,0
# 0,1
# 0,0
# 1,1
# ...
df = pd.read_csv('episode_log.csv')

# 各 State ごとに Action の頻度を集計
state_action_counts = df.groupby(['State', 'Action']).size().unstack(fill_value=0)

# 最も選択された Action とその割合を計算
most_selected_action = state_action_counts.idxmax(axis=1)  # 最も頻繁な Action (0 または 1)
most_selected_ratio = state_action_counts.max(axis=1) / state_action_counts.sum(axis=1)  # 割合

# データを結合して可視化用の DataFrame を作成
result_df = pd.DataFrame({
    'State': state_action_counts.index,
    'MostSelectedAction': most_selected_action,
    'MostSelectedRatio': most_selected_ratio
}).reset_index(drop=True)

# 棒グラフの作成
plt.figure(figsize=(12, 6))
plt.bar(result_df['State'], result_df['MostSelectedRatio'], color='skyblue', alpha=0.8, label='Most Selected Action Ratio')

# 各棒に Action (0 or 1) のラベルを表示
for idx, row in result_df.iterrows():
    plt.text(row['State'], row['MostSelectedRatio'] + 0.02, str(row['MostSelectedAction']), 
             ha='center', fontsize=10, color='black')

# グラフの設定
plt.title('Most Selected Action and Ratio per State', fontsize=16)
plt.xlabel('State', fontsize=12)
plt.ylabel('Ratio of Most Selected Action', fontsize=12)
plt.ylim(0, 1.1)  # 割合の範囲
plt.xticks(result_df['State'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()
