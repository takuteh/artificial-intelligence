import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
file_path = "Book4.csv"  # CSVファイルのパス
data = pd.read_csv(file_path)

# データの確認
print(data.head())  # データの先頭5行を表示

# グラフを作成
plt.figure(figsize=(8, 5))
plt.plot(data['episode'], data['sum'], marker='o', label="Sum")
plt.xlabel("Episode")  # 横軸のラベル
plt.ylabel("Sum")      # 縦軸のラベル
plt.title("Episode vs Sum")  # グラフのタイトル
plt.grid(True)  # グリッドを表示
plt.legend()    # 凡例を表示
plt.tight_layout()

# グラフを表示
plt.show()
