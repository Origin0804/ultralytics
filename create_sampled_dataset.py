import os
import shutil
import cv2
import random
from pathlib import Path

src_dir = Path("E:/ultralytics/ultralytics/RGB-T234")
dst_dir = Path("E:/ultralytics/ultralytics/datasets/rgbt_sampled")

# 采样策略：SAMPLE_INTERVAL=5
# 这将把训练图片总量提升到约 2000~2500 张，在Train32(1700张)和过度扩展之间找到平衡
# 目的是在不引入太多噪声数据的前提下，提供充足的多样性供模型学习
SAMPLE_INTERVAL = 10 

def create_yolo_dataset():
    print(f"🚀 开始制作高复杂度抽样数据集...")
    print(f"📂 源目录: {src_dir}")
    print(f"📦 目标目录: {dst_dir}")
    print(f"⚙️ 采样策略: 每 {SAMPLE_INTERVAL} 张提取 1 张")
    print(f"📌 划分策略: 每个场景内部 80/20 帧级别划分（不是场景级别划分）")
    
    if dst_dir.exists():
        print("清理旧的采样数据集目录...")
        shutil.rmtree(dst_dir)
        
    for split in ['train', 'val']:
        (dst_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_dir / "images_thermal" / split).mkdir(parents=True, exist_ok=True)
        (dst_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    all_seqs = []
    for seq_path in src_dir.iterdir():
        if seq_path.is_dir():
            # 只有同时存在 visible、infrared 文件夹及标签文件才视为有效场景
            if (seq_path / "visible").exists() and (seq_path / "infrared").exists() and (seq_path / "visible.txt").exists():
                all_seqs.append(seq_path.name)
                
    # 随机打乱场景顺序（固定一个随机种子保证结果可复现）
    random.seed(42)
    random.shuffle(all_seqs)
    
    print(f"🔍 找到有效场景: {len(all_seqs)} 个")
    print(f"   => 所有场景都会参与 train/val 划分（帧级别）")
    
    total_train = 0
    total_val = 0
    
    for i_s, seq in enumerate(all_seqs):
        seq_path = src_dir / seq
        
        vis_imgs = sorted(list((seq_path / "visible").glob("*.jpg")))
        inf_imgs = sorted(list((seq_path / "infrared").glob("*.jpg")))
        
        try:
            with open(seq_path / "visible.txt", "r") as f:
                lines = f.readlines()
        except Exception as e:
            continue
            
        if not vis_imgs: continue
        
        img = cv2.imread(str(vis_imgs[0]))
        if img is None: continue
        h_img, w_img = img.shape[:2]
        
        # 先收集该场景的所有满足条件的帧索引
        valid_frame_indices = []
        for i, (v_im, i_im) in enumerate(zip(vis_imgs, inf_imgs)):
            # 只考虑满足采样间隔的帧
            if i % SAMPLE_INTERVAL != 0:
                continue
                
            if i >= len(lines):
                break
                
            box_str = lines[i].strip()
            if not box_str: continue
            parts = box_str.split(',')
            if len(parts) < 4: continue
            try:
                x_min, y_min, w, h = map(float, parts[:4])
            except: continue
            if w <= 0 or h <= 0: continue
            
            valid_frame_indices.append(i)
        
        # 对有效帧进行 80/20 划分
        random.seed(42 + i_s)  # 不同场景不同随机种子，但可复现
        random.shuffle(valid_frame_indices)
        split_idx = int(len(valid_frame_indices) * 0.8)
        train_indices = set(valid_frame_indices[:split_idx])
        val_indices = set(valid_frame_indices[split_idx:])
        
        # 处理该场景的所有帧
        extracted_train = 0
        extracted_val = 0
        for i, (v_im, i_im) in enumerate(zip(vis_imgs, inf_imgs)):
            if i not in train_indices and i not in val_indices:
                continue
                
            if i >= len(lines):
                break
                
            box_str = lines[i].strip()
            if not box_str: continue
            parts = box_str.split(',')
            if len(parts) < 4: continue
            try:
                x_min, y_min, w, h = map(float, parts[:4])
            except: continue
            if w <= 0 or h <= 0: continue
            
            x_center = (x_min + w / 2) / w_img
            y_center = (y_min + h / 2) / h_img
            norm_w = w / w_img
            norm_h = h / h_img
            
            split = 'train' if i in train_indices else 'val'
            base_name = f"{seq}_{i:05d}.jpg"
            label_name = f"{seq}_{i:05d}.txt"
            
            shutil.copy(str(v_im), str(dst_dir / "images" / split / base_name))
            shutil.copy(str(i_im), str(dst_dir / "images_thermal" / split / base_name))
            with open(dst_dir / "labels" / split / label_name, "w") as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            
            if split == 'train':
                extracted_train += 1
                total_train += 1
            else:
                extracted_val += 1
                total_val += 1
        
        if (i_s + 1) % 20 == 0 or (i_s + 1) == len(all_seqs):
            print(f"[{i_s+1}/{len(all_seqs)}] 已处理完毕。当前采集数量 - Train: {total_train} | Val: {total_val}")
            
    print("\n✅ 数据集采样构建完成!")
    print(f"   📊 抽取到训练集: {total_train} 对图像")
    print(f"   📊 抽取到验证集: {total_val} 对图像")
    print(f"   📈 总计得到: {total_train + total_val} 对高质量图像")

    # 自动创建 yaml 配置文件
    yaml_content = f"""path: {dst_dir.absolute().as_posix()}
train: images/train
val: images/val

names:
  0: object
"""
    yaml_path = Path("E:/ultralytics/ultralytics/datasets/rgbt_sampled.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"📃 Dataset配置文件已生成: {yaml_path}")

if __name__ == "__main__":
    create_yolo_dataset()
