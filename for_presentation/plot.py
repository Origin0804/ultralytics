import matplotlib.pyplot as plt
import pandas as pd

# 1. 读取数据并清洗列名（去除可能存在的空格）
df = pd.read_csv('for_presentation/results.csv')
df.columns = df.columns.str.strip()

# 2. 定义学术科技感配色
primary_color = '#1f77b4'  # 经典科技蓝（训练集/主要曲线）
val_color = '#ff7f0e'      # 活力橙（验证集曲线，形成鲜明对比）
accent_color = '#2ca02c'   # 翡翠绿（用于突出 mAP50 核心成果）

# ==========================================
# 图一：PPT 专用模型精度趋势图 (mAP Scores)
# ==========================================
plt.figure(figsize=(8, 5))  # 适合 PPT 单侧排版的黄金比例

# 绘制 mAP50 和 mAP50-95 曲线
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label=r'$mAP_{50}$', color=accent_color, linewidth=2.5)
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label=r'$mAP_{50-95}$', color=primary_color, linewidth=2)

# 自动定位或硬编码标注最佳性能点（经计算第 110 轮最佳）
best_epoch = 110
best_map50 = 0.9588

# 在图上绘制红点
plt.scatter(best_epoch, best_map50, color='red', s=60, zorder=5)
# 添加带有箭头的文本标注
plt.annotate(f'Best: {best_map50:.4f} (Epoch {best_epoch})', 
             xy=(best_epoch, best_map50), 
             xytext=(best_epoch - 42, best_map50 - 0.15), # 调整文本框位置防止挡住曲线
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
             fontsize=11, fontweight='bold')

# 图表精细化配置
plt.title('Model Accuracy Trend (mAP Scores)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xlim(0, df['epoch'].max() + 3)
plt.ylim(0, 1.05)
plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
# transparent=True 设置透明背景，dpi=200 保证答辩投影不模糊
plt.savefig('ppt_accuracy_metrics.png', dpi=200, transparent=True)
plt.close()


# ==========================================
# 图二：PPT 专用边界框定位损失收敛图 (Box Loss)
# ==========================================
plt.figure(figsize=(8, 5))

# 绘制训练集和验证集的 Box Loss
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color=primary_color, linewidth=2.5)
plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color=val_color, linestyle='--', linewidth=2.5)

# 图表精细化配置
plt.title('Bounding Box Regression Loss Convergence', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xlim(0, df['epoch'].max() + 3)
plt.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
# 同样保存为高清透明背景图片
plt.savefig('ppt_loss_convergence.png', dpi=200, transparent=True)
plt.close()

print("🎉 PPT专用高清图表已成功生成：'ppt_accuracy_metrics.png' 和 'ppt_loss_convergence.png'")