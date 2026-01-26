import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 手动填写数据
# =========================

param_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7])
rank1_values = np.array([70.57, 71.56, 72.54, 72.85, 71.68, 72.29,71.54])
map_values   = np.array([68.14, 68.89, 70.06, 70.17, 69.52, 69.99,69.39])

# param_values = np.array([1.4,1.5,1.6,1.7,1.8,1.9,2.0])
# rank1_values = np.array([74.31,74.50,74.75,74.90,74.63,74.62,74.75])
# map_values   = np.array([71.39,71.47,71.81,71.84,70.79,71.12,71.57])

# param_values = np.array([0.20,0.21,0.22,0.23,0.24,0.25,0.26])
# rank1_values = np.array([74.85,75.00,74.43,74.18,75.33,74.94,74.46])
# map_values   = np.array([71.76,71.45,71.46,70.30,72.26,71.77,70.88])

# =========================
# 2. 颜色（写死）
# =========================
RANK1_COLOR = '#2FA8A0'
MAP_COLOR   = '#A6DED9'

# RANK1_COLOR = '#CC6677'
# MAP_COLOR   = '#DDCC77'

# RANK1_COLOR = '#4477AA'
# MAP_COLOR   = '#88CCEE' 

# =========================
# 3. 画柱状图
# =========================

x = np.arange(len(param_values))
bar_width = 0.35

plt.figure(figsize=(5.5, 4))

plt.bar(
    x - bar_width / 2,
    rank1_values,
    width=bar_width,
    color=RANK1_COLOR,
    label='Rank-1'
)

plt.bar(
    x + bar_width / 2,
    map_values,
    width=bar_width,
    color=MAP_COLOR,
    label='mAP'
)

# =========================
# 3.5 在柱顶标注数值（新增）
# =========================

for i, v in enumerate(rank1_values):
    plt.text(
        x[i] - bar_width / 2,
        v + 0.05,
        f'{v:.2f}',
        ha='center',
        va='bottom',
        fontsize=6,
        color='black'
    )

for i, v in enumerate(map_values):
    plt.text(
        x[i] + bar_width / 2,
        v + 0.05,
        f'{v:.2f}',
        ha='center',
        va='bottom',
        fontsize=6,
        color='black'
    )


# =========================
# 4. 坐标轴设置（修改重点）
# =========================

plt.xticks(x, [str(v) for v in param_values])
plt.xlabel(r'$\lambda_{1}$')
plt.ylabel('Rank-1 / mAP %')

# 🔹 自适应 Y 轴范围（放大差异）
y_min = min(rank1_values.min(), map_values.min()) - 2
y_max = max(rank1_values.max(), map_values.max()) + 2
plt.ylim(y_min, y_max)

plt.legend(frameon=False)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

# =========================
# 5. 保存图片
# =========================

# plt.savefig('lambda1_hyperparam.pdf', format='pdf', bbox_inches='tight')
plt.savefig('lambda1_hyperparam.png', dpi=300, bbox_inches='tight')

plt.show()
