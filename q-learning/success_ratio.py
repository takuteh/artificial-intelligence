import pandas as pd
import matplotlib.pyplot as plt

data="episode_log.csv"
# データフレームとして読み込む
df = pd.read_csv(data)

# `running` を無視し、`success` または `fault` のみを抽出
filtered_df = df[df['Result'].isin(['success', 'fault'])]

# エピソードごとに最初の `Result` を取得
episode_results = (
    filtered_df.groupby('Episode')['Result']
    .first()  # エピソードごとの最初の結果を取得
    .reset_index()
)

# 100エピソードごとにグループ化し、成功率を計算
episode_results['EpisodeGroup'] = (episode_results['Episode'] - 1) #// 100
success_ratio = (
    episode_results.groupby('EpisodeGroup')['Result']
    .apply(lambda group: (group == 'success').sum() / len(group))  # 成功割合
    .reset_index(name='SuccessRatio')
)

# エピソード範囲ラベルを作成 (例: 0-99, 100-199,...)
success_ratio['EpisodeRange'] = success_ratio['EpisodeGroup'].apply(
    lambda x: f"{x*100}-{(x+1)*100-1}"
)

# グラフ作成
plt.figure(figsize=(10, 6))
plt.plot(success_ratio['EpisodeGroup'] * 100, success_ratio['SuccessRatio'], marker='o', color='blue', label='Success Ratio')
plt.title('Success Ratio per 100 Episodes')
plt.xlabel('Episode')
plt.ylabel('Success Ratio')
plt.ylim(0, 1)  # 成功率は0～1の範囲
plt.grid(alpha=0.6)
plt.xticks(ticks=success_ratio['EpisodeGroup'] * 100, labels=success_ratio['EpisodeRange'], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
