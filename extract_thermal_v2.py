"""使用 YOLO 加载提取 Thermal 权重"""
from ultralytics import YOLO

print("加载模型...")
model = YOLO('runs/detect/train28/weights/best.pt')

print("\n模型架构:")
print(model.model)

# 获取状态字典
state_dict = model.state_dict()
print(f"\n状态字典键数: {len(state_dict)}")

# 列出所有层
print("\n所有层 (前 30 个):")
for i, key in enumerate(sorted(state_dict.keys())[:30]):
    param = state_dict[key]
    param_count = param.numel() if hasattr(param, 'numel') else 0
    print(f"  {i:2d}. {key:<60} | {param_count:>10,}")

# 提取 Thermal 主干权重
print("\n" + "="*70)
print("提取 Thermal 主干权重")
print("="*70)

thermal_weights = {}
thermal_param_count = 0

for key, value in state_dict.items():
    # 检查 Thermal 主干层 (model.13-22)
    if any(f'model.{i}' in key for i in range(13, 23)):
        thermal_weights[key] = value
        param_count = value.numel() if hasattr(value, 'numel') else 0
        thermal_param_count += param_count
        print(f"✓ {key:<60} | {param_count:>10,}")

print(f"\n总计: {len(thermal_weights)} 层, {thermal_param_count:,} 参数")

if thermal_weights:
    import torch
    torch.save({
        'thermal_state_dict': thermal_weights,
        'total_params': thermal_param_count,
        'source': 'train28 (262 epochs)',
    }, 'thermal_backbone.pt')
    print(f"\n✓ 已保存到 thermal_backbone.pt")
else:
    print("\n⚠ 未找到 Thermal 层，检查模型架构...")
