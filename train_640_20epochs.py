from ultralytics import YOLO

def train_640_from_scratch():
    print("="*50)
    print("Starting 20-epoch RGBT training test at imgsz=640")
    print("="*50)
    
    # 这里我们使用之前为你特制的含有COCO权重映射的双流初始底座
    # 这是最纯净的“从零开始”（对这批目标没有记忆，但具备基础的边缘提取能力）
    model = YOLO('yolov8n_multimodal_pretrained.pt')
    
    # 开始训练
    results = model.train(
        data='datasets/rgbt_subset.yaml',
        epochs=20,
        imgsz=640,         # 提升到正规的 640 分辨率
        batch=16,           # 640尺寸下双流极其吃显存，必须把batch调低到4
        name='train_640_20epochs', # 新建一个专门的文件夹名以防混淆
        device=0,          # 使用第一张显卡
        workers=2,         # 减少数据加载线程防止CPU卡顿
        save=True          # 确保保存权重
    )
    
    print("\nTraining completed! Check runs/detect/train_640_20epochs for results.")

if __name__ == '__main__':
    train_640_from_scratch()
