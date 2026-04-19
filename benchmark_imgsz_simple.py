"""
简化版尺寸对比：单线程避免 Windows 多进程问题
"""
if __name__ == "__main__":
    import torch
    from pathlib import Path
    from ultralytics import YOLO
    
    TEST_IMGSZ = [256, 384, 512]
    MODEL_PATH = "runs/detect/train19/weights/best.pt"
    DATA_CONFIG = "datasets/rgbt_subset.yaml"
    
    if not Path(MODEL_PATH).exists():
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        exit(1)
    
    print("=" * 70)
    print("快速尺寸对比评估 (单线程)")
    print("=" * 70)
    
    results_summary = []
    
    for imgsz in TEST_IMGSZ:
        print(f"\n📏 测试尺寸: {imgsz}×{imgsz}")
        print("-" * 70)
        
        try:
            model = YOLO(MODEL_PATH)
            
            # 关键：workers=0 禁用多进程，避免 Windows 问题
            results = model.val(
                data=DATA_CONFIG,
                imgsz=imgsz,
                batch=4,
                workers=0,  # 单线程
                plots=False,
                verbose=False
            )
            
            p = results.box.p
            r = results.box.r
            map50 = results.box.map50
            map_val = results.box.map
            
            record = {
                "imgsz": imgsz,
                "Precision": p,
                "Recall": r,
                "mAP50": map50,
                "mAP50-95": map_val
            }
            results_summary.append(record)
            
            print(f"  Precision:  {p:.4f}")
            print(f"  Recall:     {r:.4f}")
            print(f"  mAP50:      {map50:.4f}")
            print(f"  mAP50-95:   {map_val:.4f}")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ⚠️  显存不足 (OOM)，跳过此尺寸")
                results_summary.append({"imgsz": imgsz, "status": "OOM"})
            else:
                print(f"  ❌ 错误: {e}")
                results_summary.append({"imgsz": imgsz, "status": "Error"})
    
    # 汇总
    print("\n" + "=" * 70)
    print("📊 尺寸对比总结")
    print("=" * 70)
    print(f"\n{'尺寸':<10} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP50-95':<12}")
    print("-" * 70)
    
    for rec in results_summary:
        if "status" in rec:
            print(f"{rec['imgsz']:<10} {rec['status']:<12}")
        else:
            p_str = f"{rec['Precision']:.4f}"
            r_str = f"{rec['Recall']:.4f}"
            m50_str = f"{rec['mAP50']:.4f}"
            m95_str = f"{rec['mAP50-95']:.4f}"
            print(f"{rec['imgsz']:<10} {p_str:<12} {r_str:<12} {m50_str:<12} {m95_str:<12}")
    
    # 分析建议
    print("\n" + "=" * 70)
    print("💡 建议")
    print("=" * 70)
    
    if len(results_summary) >= 2 and "status" not in results_summary[0]:
        r_256 = results_summary[0]["Recall"]
        r_384 = results_summary[1]["Recall"] if len(results_summary) > 1 and "Recall" in results_summary[1] else None
        
        if r_384:
            r_delta = r_384 - r_256
            print(f"\n✓ Recall 从 256 到 384 的变化: {r_delta:+.4f}")
            
            if r_delta > 0.05:
                print("  → Recall 明显改善！建议升级到 384 或更大")
            elif r_delta > 0.01:
                print("  → 略有改善，可继续尝试更大尺寸")
            else:
                print("  → 改善不明显，低 Recall 问题不在分辨率")
        else:
            print("\n⚠️  缺少对比数据")
    
    print("=" * 70)
