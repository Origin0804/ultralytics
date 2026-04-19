"""
快速对比不同图像尺寸下的模型表现 (Recall/Precision/mAP)
用途：评估 256 vs 384 vs 512 对检测效果的影响，帮助决定是否升级到 640
"""
import os
from pathlib import Path
from ultralytics import YOLO

# 定义测试尺寸（递进式，避免立刻跳到 640 导致 OOM）
TEST_IMGSZ = [256, 384, 512]
MODEL_PATH = "yolov8n_multimodal_pretrained.pt"  # 使用当前最佳模型
DATA_CONFIG = "datasets/rgbt_subset.yaml"

if not Path(MODEL_PATH).exists():
    print(f"❌ 模型文件不存在: {MODEL_PATH}")
    print("   请确保训练已完成并检查路径")
    exit(1)

print("=" * 70)
print("快速尺寸对比评估")
print("=" * 70)

results_summary = []

for imgsz in TEST_IMGSZ:
    print(f"\n📏 测试尺寸: {imgsz}×{imgsz}")
    print("-" * 70)
    
    try:
        model = YOLO(MODEL_PATH)
        
        # 验证模型（仅用默认 conf=0.25）
        results = model.val(
            data=DATA_CONFIG,
            imgsz=imgsz,
            batch=4,  # 小 batch 防止 OOM
            plots=False,
            verbose=False
        )
        
        # 提取关键指标
        p = results.box.p  # Precision
        r = results.box.r  # Recall
        map50 = results.box.map50  # mAP@0.5
        map = results.box.map  # mAP@0.5:0.95
        
        record = {
            "imgsz": imgsz,
            "Precision": p,
            "Recall": r,
            "mAP50": map50,
            "mAP50-95": map
        }
        results_summary.append(record)
        
        print(f"  Precision:  {p:.4f}")
        print(f"  Recall:     {r:.4f}")
        print(f"  mAP50:      {map50:.4f}")
        print(f"  mAP50-95:   {map:.4f}")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  ⚠️  显存不足 (OOM)，跳过此尺寸")
            results_summary.append({"imgsz": imgsz, "status": "OOM"})
        else:
            print(f"  ❌ 错误: {e}")
            results_summary.append({"imgsz": imgsz, "status": "Error"})

# 汇总对比
print("\n" + "=" * 70)
print("📊 尺寸对比总结")
print("=" * 70)
print(f"\n{'尺寸':<10} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP50-95':<12}")
print("-" * 70)

for rec in results_summary:
    if "status" in rec:
        print(f"{rec['imgsz']:<10} {rec['status']:<12}")
    else:
        p_str = f"{rec['Precision']:.4f}" if isinstance(rec['Precision'], float) else "N/A"
        r_str = f"{rec['Recall']:.4f}" if isinstance(rec['Recall'], float) else "N/A"
        m50_str = f"{rec['mAP50']:.4f}" if isinstance(rec['mAP50'], float) else "N/A"
        m95_str = f"{rec['mAP50-95']:.4f}" if isinstance(rec['mAP50-95'], float) else "N/A"
        print(f"{rec['imgsz']:<10} {p_str:<12} {r_str:<12} {m50_str:<12} {m95_str:<12}")

# 分析与建议
print("\n" + "=" * 70)
print("💡 建议")
print("=" * 70)

if len(results_summary) >= 2 and "status" not in results_summary[0]:
    r_256 = results_summary[0]["Recall"]
    r_384 = results_summary[1]["Recall"]
    r_delta = r_384 - r_256
    
    print(f"\n✓ Recall 从 256 到 384 的变化: {r_delta:+.4f}")
    
    if r_delta > 0.05:
        print("  → Recall 明显改善！建议尝试更大尺寸（512 或 640）")
    elif r_delta > 0.01:
        print("  → 略有改善，继续增大尺寸可能有帮助")
    else:
        print("  → 改善不明显，问题可能不在分辨率，考虑调整:")
        print("     • 置信度阈值（conf）或 NMS 参数")
        print("     • 数据增强或 Anchor 尺寸")
        print("     • 检查是否真的是小物体漏检 vs 其他原因")
else:
    print("\n⚠️  测试未能完全进行，请检查模型和数据配置")

print("\n" + "=" * 70)
