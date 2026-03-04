import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

fem_temp = pd.read_csv(r'C:\Users\pzou5\OneDrive\Documents\githubrepo\micetempact\FemTemp.csv')

# 时间轴转换成"天"
minutes = fem_temp['time (min)'].values
days = minutes / 1440  # 1天 = 1440分钟

# 画图：每只雌鼠一行，共14个子图
fig, axes = plt.subplots(14, 1, figsize=(14, 28), sharex=True)
fig.suptitle('Female CBT raw', fontsize=14, y=1.001)

mouse_cols = [c for c in fem_temp.columns if c != 'time (min)']

for i, (ax, col) in enumerate(zip(axes, mouse_cols)):
    ax.plot(days, fem_temp[col], linewidth=0.5, color='steelblue')
    ax.set_ylabel(col, fontsize=8, rotation=0, labelpad=30)
    ax.set_ylim(35, 40)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.tick_params(labelsize=7)
    # 每天画一条竖线方便看
    for d in range(1, 9):
        ax.axvline(x=d, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

axes[-1].set_xlabel('Time (days)', fontsize=10)
plt.tight_layout()
plt.savefig(r'C:\Users\pzou5\OneDrive\Documents\githubrepo\micetempact\female_temp.png', dpi=150, bbox_inches='tight')  # ← 改这里
print("图保存好了！")