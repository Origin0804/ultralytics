"""
提取当前训练模型中的 Thermal Backbone 权重，保存为独立的预训练文件
用于在新任务中初始化红外特征提取器

Thermal Backbone 对应的层：13-22（与 RGB Backbone 对称，都是处理 3 通道特征）
"""
import torch
from pathlib import Path

# 避免多进程导入问题，直接用 torch.load
if __name__ == "__main__":

SOURCE_MODEL = "runs/detect/train19/weights/best.pt"
OUTPUT_PATH = "thermal_backbone_pretrained.pt"

print("=" * 70)
print("提取 Thermal Backbone 预训练权重")
print("=" * 70)

if not Path(SOURCE_MODEL).exists():
    print(f"❌ 源模型不存在: {SOURCE_MODEL}")
    exit(1)

# 直接加载 .pt 文件（避免 YOLO 导入的多进程问题）
print(f"\n📂 加载源模型权重: {SOURCE_MODEL}")
try:
    checkpoint = torch.load(SOURCE_MODEL, map_location='cpu')
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit(1)

# 获取模型状态字典（.pt 文件中通常有 'model' 键）
if isinstance(checkpoint, dict) and 'model' in checkpoint:
    full_state = checkpoint['model']
elif isinstance(checkpoint, dict):
    full_state = checkpoint
else:
    print("❌ 无法识别模型格式")
    exit(1)

# 提取 Thermal Backbone 的权重（layer 13-22）
thermal_state = {}
thermal_layers = range(13, 23)  # 13-22 inclusive

print("\n🔍 提取 Thermal Backbone 层权重...")
for key in full_state.keys():
    try:
        layer_idx = int(key.split('.')[1])
        if layer_idx in thermal_layers:
            thermal_state[key] = full_state[key]
            print(f"  ✓ {key}")
    except (IndexError, ValueError):
        pass

print(f"\n✓ 成功提取 {len(thermal_state)} 个张量")

# 保存为独立的 .pt 文件
# 为了兼容 YOLO 加载，保存原始格式但仅包含 Thermal 权重
new_checkpoint = {
    "epoch": checkpoint.get("epoch", -1),
    "model": thermal_state,  # 仅保存 Thermal 部分的 state_dict
    "optimizer": None,
    "thermal_only": True,  # 标记这是 Thermal 专用权重
}

torch.save(new_checkpoint, OUTPUT_PATH)
print(f"\n💾 已保存到: {OUTPUT_PATH}")

# 统计信息
total_params = sum(p.numel() if isinstance(p, torch.Tensor) else 0 for p in thermal_state.values())
print(f"📊 Thermal Backbone 参数量: {total_params / 1e6:.2f}M")

print("\n" + "=" * 70)
print("📝 后续使用说明")
print("=" * 70)
print("""
1. 新任务中加载此权重的方式：
   
   # 方式1：直接加载状态字典
   import torch
   thermal_weights = torch.load('thermal_backbone_pretrained.pt')['model']
   model.load_state_dict(thermal_weights, strict=False)
   
   # 方式2：或在训练前初始化
   from ultralytics import YOLO
   model = YOLO('yolov8n_multimodal.yaml')
   model.load('thermal_backbone_pretrained.pt')

2. 注意事项：
   • 这只是 Thermal 部分，RGB Backbone 仍需其他初始化
   • 如果新任务的目标类别差异大，可能需要微调整个模型
   • 建议先冻结 Thermal 层，仅微调其他部分测试效果

3. 更优方案（推荐）：
   • 使用整个 runs/detect/train19/weights/best.pt 作为 multimodal 预训练
   • 而不是仅用 Thermal 部分
""")

print("=" * 70)
