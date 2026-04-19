"""快速提取 Thermal 主干权重"""
import torch
import json

# 加载完整训练权重
model_path = 'runs/detect/train28/weights/best.pt'
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

print(f"加载模型: {model_path}")
print(f"检查点键: {checkpoint.keys()}")

state_dict = checkpoint['model']
print(f"\n模型状态字典键数: {len(state_dict)}")

# 列出所有层
print("\n所有层名称:")
for i, key in enumerate(sorted(state_dict.keys())):
    param_count = state_dict[key].numel() if isinstance(state_dict[key], torch.Tensor) else 0
    print(f"  {i:3d}. {key:<50} | {param_count:>10,} params")

# 提取 Thermal 主干 (layers 13-22 在 dual-stream 架构中)
print("\n" + "="*70)
print("提取 Thermal 主干权重 (预期层 13-22)")
print("="*70)

thermal_weights = {}
thermal_param_count = 0

for key, value in state_dict.items():
    # 检查是否是 Thermal 主干的层 (索引 13-22)
    if 'model.13' in key or 'model.14' in key or 'model.15' in key or \
       'model.16' in key or 'model.17' in key or 'model.18' in key or \
       'model.19' in key or 'model.20' in key or 'model.21' in key or 'model.22' in key:
        thermal_weights[key] = value
        param_count = value.numel() if isinstance(value, torch.Tensor) else 0
        thermal_param_count += param_count
        print(f"✓ {key:<50} | {param_count:>10,} params")

print(f"\n总计: {len(thermal_weights)} 层, {thermal_param_count:,} 参数")

# 保存
save_path = 'thermal_backbone.pt'
torch.save({
    'thermal_state_dict': thermal_weights,
    'total_params': thermal_param_count,
    'source_model': model_path,
    'source_epochs': 262
}, save_path)

print(f"\n✓ 保存到: {save_path}")

# 验证
if torch.load(save_path, map_location='cpu')['thermal_state_dict']:
    print(f"✓ 验证成功: 包含 {len(torch.load(save_path)['thermal_state_dict'])} 层")
