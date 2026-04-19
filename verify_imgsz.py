from ultralytics import YOLO
import torch

def verify():
    # Load the trained RGBT model
    weights_path = 'runs/detect/train28/weights/last.pt'
    print(f"Loading weights from {weights_path}")
    model = YOLO(weights_path)
    
    print("\n" + "="*50)
    print("Running validation with imgsz = 256")
    print("="*50)
    res_256 = model.val(data='datasets/rgbt_subset.yaml', imgsz=256, batch=4, plots=False, split='val')
    
    print("\n" + "="*50)
    print("Running validation with imgsz = 640")
    print("="*50)
    res_640 = model.val(data='datasets/rgbt_subset.yaml', imgsz=640, batch=4, plots=False, split='val')
    
    # Extract metrics
    metrics_256 = {
        'Precision': res_256.box.mp,
        'Recall': res_256.box.mr,
        'mAP@50': res_256.box.map50,
        'mAP@50-95': res_256.box.map
    }
    
    metrics_640 = {
        'Precision': res_640.box.mp,
        'Recall': res_640.box.mr,
        'mAP@50': res_640.box.map50,
        'mAP@50-95': res_640.box.map
    }
    
    print("\n" + "="*50)
    print("COMPARISON RESULTS:")
    print(f"{'Metric':<15} | {'imgsz=256':<15} | {'imgsz=640':<15} | {'Improvement':<15}")
    print("-" * 65)
    for k in metrics_256.keys():
        diff = metrics_640[k] - metrics_256[k]
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        print(f"{k:<15} | {metrics_256[k]:<15.4f} | {metrics_640[k]:<15.4f} | {diff_str:<15}")
    print("="*50)

if __name__ == '__main__':
    verify()