import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

file_path = "DL_Features.csv"
df = pd.read_csv(file_path)

names = df.iloc[:, 0].values                  # 第一列：姓名
labels = df.iloc[:, 1].values.astype(int)     # 第二列：类别标签（0或1）
X = df.iloc[:, 2:].values.astype(float)       # 第三列往后：512维特征

rs = 42  # 随机种子
tsne = TSNE(n_components=2, random_state=rs, perplexity=30, n_iter=500)
X_tsne = tsne.fit_transform(X)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(X_tsne[labels == 1, 0], X_tsne[labels == 1, 1],
           c='red', label='Mutation', alpha=0.7)
ax.scatter(X_tsne[labels == 0, 0], X_tsne[labels == 0, 1],
           c='blue', label='Wild-type', alpha=0.7)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.legend()
plt.title('t-SNE Plot', fontsize=12)

save_dir = "tsne_185"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"tsne_final_185_{rs}.svg")
plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.1)
plt.close()

