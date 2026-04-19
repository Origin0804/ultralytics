import os
import shutil
import cv2
from pathlib import Path

src_dir = Path("./RGB-T234")
dst_dir = Path("datasets/rgbt_subset")

subset_seqs = [
    'afterrain', 'aftertree', 'baby', 'baginhand', 'balancebike', 
    'baketballwaliking', 'basketball2', 'bicyclecity', 'bike', 'bikeman', 
    'bikemove1', 'biketwo', 'blackwoman', 'bluebike', 'blueCar', 
    'boundaryandfast', 'bus6', 'call', 'car', 'car10', 'car20', 'car3'
]
# Split into train: val
train_seqs = subset_seqs[:18]
val_seqs = subset_seqs[18:]

def create_yolo_dataset():
    for split in ['train', 'val']:
        (dst_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_dir / "images_thermal" / split).mkdir(parents=True, exist_ok=True)
        (dst_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    for seq in subset_seqs:
        seq_path = src_dir / seq
        split = 'train' if seq in train_seqs else 'val'
        
        vis_dir = seq_path / "visible"
        inf_dir = seq_path / "infrared"
        
        if not vis_dir.exists() or not inf_dir.exists():
            continue
            
        vis_imgs = sorted(list(vis_dir.glob("*.jpg")))
        inf_imgs = sorted(list(inf_dir.glob("*.jpg")))
        
        # Load vis annotations
        vis_txt = seq_path / "visible.txt"
        if not vis_txt.exists():
            continue
            
        with open(vis_txt, "r") as f:
            lines = f.readlines()
            
        # Get image bounds from first image
        img = cv2.imread(str(vis_imgs[0]))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        
        for i, (v_im, i_im) in enumerate(zip(vis_imgs, inf_imgs)):
            if i >= len(lines):
                break
                
            box_str = lines[i].strip()
            if not box_str:
                continue
            
            parts = box_str.split(',')
            if len(parts) < 4:
                continue
                
            x_min, y_min, w, h = map(float, parts[:4])
            # Filter zero/invalid boxes
            if w <= 0 or h <= 0:
                continue
            
            # Convert to YOLO (x_center, y_center, width, height) normalized
            x_center = (x_min + w / 2) / w_img
            y_center = (y_min + h / 2) / h_img
            norm_w = w / w_img
            norm_h = h / h_img
            
            # Destination names
            base_name = f"{seq}_{i:05d}.jpg"
            label_name = f"{seq}_{i:05d}.txt"
            
            dst_vis = dst_dir / "images" / split / base_name
            dst_inf = dst_dir / "images_thermal" / split / base_name
            dst_lbl = dst_dir / "labels" / split / label_name
            
            # Copy files
            shutil.copy(v_im, dst_vis)
            shutil.copy(i_im, dst_inf)
            
            with open(dst_lbl, "w") as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

if __name__ == "__main__":
    create_yolo_dataset()
    print("Subset creation completed.")
