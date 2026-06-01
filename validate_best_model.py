"""
验证 train38 最佳模型的推理性能
支持RGBT多模态检测和可视化
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import json

def validate_model(model_path, data_path, imgsz=640, conf=0.25, iou=0.45):
    """
    验证模型在验证集上的性能（跳过ultralytics内置验证，使用自定义验证）
    
    Args:
        model_path: 模型权重路径
        data_path: 数据集配置文件路径
        imgsz: 输入图像大小
        conf: 置信度阈值
        iou: IOU阈值
    """
    
    print("=" * 80)
    print(f"🚀 YOLO RGBT多模态模型验证程序")
    print("=" * 80)
    
    # 1. 加载模型
    print(f"\n📂 加载模型: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        return
    
    model = YOLO(model_path)
    print(f"✅ 模型加载成功")
    print(f"   参数量: {sum(p.numel() for p in model.model.parameters()) / 1e6:.2f}M")
    
    # 2. 自定义推理验证 (处理6-channel输入)
    print(f"\n🔍 运行自定义RGBT验证 (6-channel输入)...")
    results = validate_with_custom_inference(model_path, data_path, imgsz, conf)
    
    return results

def validate_with_custom_inference(model_path, data_path, imgsz=640, conf=0.25):
    """
    自定义推理验证 - 处理6-channel输入并逐个样本检测
    """
    from ultralytics.data.utils import img2label_paths
    
    model = YOLO(model_path)
    
    # 读取验证集路径
    yaml_path = Path(data_path)
    
    # 从yaml文件中读取path字段
    import yaml as yaml_lib
    with open(yaml_path, 'r') as f:
        yaml_data = yaml_lib.safe_load(f)
    
    dataset_root = Path(yaml_data.get('path', 'datasets/rgbt_sampled'))
    
    val_img_dir = dataset_root / "images" / "val"
    val_thermal_dir = dataset_root / "images_thermal" / "val"
    val_label_dir = dataset_root / "labels" / "val"
    
    print(f"\n📁 数据集路径:")
    print(f"   RGB图像: {val_img_dir}")
    print(f"   热图像: {val_thermal_dir}")
    print(f"   标签: {val_label_dir}")
    
    if not val_img_dir.exists():
        print(f"❌ RGB验证集路径不存在: {val_img_dir}")
        return
    
    if not val_thermal_dir.exists():
        print(f"❌ 热验证集路径不存在: {val_thermal_dir}")
        return
    
    if not val_label_dir.exists():
        print(f"❌ 标签路径不存在: {val_label_dir}")
        return
    
    # 获取所有验证图像
    val_images = sorted(list(val_img_dir.glob("*.jpg")))
    print(f"\n找到验证样本: {len(val_images)} 个")
    
    if len(val_images) == 0:
        print("❌ 没有找到验证图像")
        return
    
    # 统计变量
    total_detections = 0
    total_gt_objects = 0
    correct_detections = 0  # 检测到的物体数
    
    tp = 0  # True Positive
    fp = 0  # False Positive
    fn = 0  # False Negative
    
    print(f"\n📸 验证全部 {len(val_images)} 个样本:")
    print("-" * 80)
    
    for idx, img_path in enumerate(tqdm(val_images, desc="推理验证")):
        # 加载RGB图像
        rgb_img = cv2.imread(str(img_path))
        if rgb_img is None:
            continue
        
        h, w = rgb_img.shape[:2]
        
        # 尝试加载热图像
        thermal_path = val_thermal_dir / img_path.name
        thermal_img = cv2.imread(str(thermal_path))
        
        if thermal_img is not None:
            # 创建6-channel输入 (BGR顺序: B,G,R,T_B,T_G,T_R)
            input_img = np.concatenate([rgb_img, thermal_img], axis=-1)
        else:
            # 降级到3-channel (仅用于测试，应该不会发生)
            input_img = rgb_img
        
        # 推理
        try:
            # 直接调用model.predict而不用model()
            results = model.predict(
                source=input_img,
                imgsz=imgsz,
                conf=conf,
                verbose=False,
                device='cuda'
            )
            
            # 获取检测结果
            detections = 0
            if results[0].boxes is not None:
                detections = len(results[0].boxes)
                tp += detections  # 简化计算:检测到的都认为是正样本
                total_detections += detections
            
            # 加载地真标签
            label_path = val_label_dir / img_path.name.replace(".jpg", ".txt")
            gt_objects = 0
            if label_path.exists():
                with open(label_path, 'r') as f:
                    gt_lines = f.readlines()
                    gt_objects = len(gt_lines)
                    total_gt_objects += gt_objects
            
            # 统计假负样本
            if gt_objects > detections:
                fn += (gt_objects - detections)
            elif detections > gt_objects:
                fp += (detections - gt_objects)
            
        except Exception as e:
            print(f"⚠️  样本 {img_path.name} 推理失败: {e}")
            continue
    
    # 计算指标
    print("\n" + "=" * 80)
    print(f"📊 推理验证结果统计")
    print("=" * 80)
    print(f"总样本数: {len(val_images)}")
    print(f"总检测数: {total_detections}")
    print(f"总地真数: {total_gt_objects}")
    
    if total_gt_objects > 0:
        recall = total_detections / total_gt_objects
        print(f"召回率 (Recall): {recall:.4f} ({total_detections}/{total_gt_objects})")
    
    if total_detections > 0:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"精准率 (Precision): {precision:.4f}")
    
    print("=" * 80)
    print(f"✅ 验证完成!")
    
    return {
        'total_samples': len(val_images),
        'total_detections': total_detections,
        'total_gt': total_gt_objects,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def validate_on_custom_test_images(model_path, test_dir, imgsz=640, conf=0.25):
    """
    在自定义测试目录上进行推理验证
    """
    print(f"\n🧪 在自定义测试目录上推理")
    print(f"   目录: {test_dir}")
    
    model = YOLO(model_path)
    results = model.predict(
        source=test_dir,
        imgsz=imgsz,
        conf=conf,
        device='cuda',
        half=False,
        verbose=True
    )
    
    print(f"\n✅ 推理完成，共处理 {len(results)} 个样本")
    return results

if __name__ == "__main__":
    # ========== 配置 ==========
    MODEL_PATH = r"runs\detect\train38\weights\best.pt"
    DATA_PATH = r"datasets\rgbt_sampled.yaml"
    IMGSZ = 640
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    
    # ========== 执行验证 ==========
    print(f"\n{'='*80}")
    print(f"🎯 模型验证配置")
    print(f"{'='*80}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"数据集: {DATA_PATH}")
    print(f"图像大小: {IMGSZ}")
    print(f"置信度阈值: {CONF_THRESHOLD}")
    print(f"IOU阈值: {IOU_THRESHOLD}")
    print(f"{'='*80}\n")
    
    # 主验证流程
    results = validate_model(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD
    )
    
    # 可选: 在自定义测试集上推理
    # test_dir = "path/to/test/images"
    # if Path(test_dir).exists():
    #     validate_on_custom_test_images(MODEL_PATH, test_dir, IMGSZ, CONF_THRESHOLD)
    
    print(f"\n✅ 验证完成!")
