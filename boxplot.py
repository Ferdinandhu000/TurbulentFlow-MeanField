import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


csv_dir = Path('csv_outside')

data = []
labels = []

for file_path in csv_dir.glob('*.csv'):
    model_name = file_path.stem.replace('_outside', '')

    df = pd.read_csv(file_path)
    rmse = df['mae'].to_list()

    labels.append(model_name)
    data.append(rmse)

order = [4, 0, 5, 3, 1, 2]
data = [data[i] for i in order]
labels = [labels[i] for i in order]


plt.figure(figsize=(10, 5))




# 3. 绘制箱型图
bplot = plt.boxplot(data,
            labels=labels,
            showfliers=False,
            patch_artist=True)
face_colors = ["#bad0e7", "#b8e7c4", "#e9edb7", "#f0d7c3" , "#e7a7af", "#c4aade"]
edge_colors = ["#1c68b9", "#21bb48", "#bfb724", "#bb6118" , "#951625", "#732bbb"]

for i in range(len(data)):
    bplot['boxes'][i].set_facecolor(face_colors[i])
    bplot['boxes'][i].set_edgecolor(edge_colors[i])
    bplot['boxes'][i].set_linewidth(1.5)

    bplot['caps'][2*i].set_color(edge_colors[i])
    bplot['caps'][2*i+1].set_color(edge_colors[i])
    bplot['whiskers'][2*i].set_color(edge_colors[i])
    bplot['whiskers'][2*i+1].set_color(edge_colors[i])

    bplot['medians'][i].set_color(edge_colors[i])
    bplot['medians'][i].set_linewidth(2)

plt.title("Model Evaluation Loss")
plt.ylabel("MAE")
plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线
plt.show()