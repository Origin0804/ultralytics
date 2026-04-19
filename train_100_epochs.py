from ultralytics import YOLO
import torch
from multiprocessing import freeze_support

# # 检查 Mac M 系列芯片的 MPS (Metal Performance Shaders) 加速是否可用
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# print(f"🚀 Training will use device: {device}")

def main():
	# load the custom model with cloned pretrained weights
        model = YOLO('yolov8n_multimodal_pretrained.pt')

        # train the model with localized parameters for RTX 4060 Laptop (8GB VRAM)
        model.train(
            data='datasets/rgbt_sampled.yaml', # 切成了我们刚刚抽样出来的全场景高复杂度数据集
            epochs=300, 
            imgsz=640, 
            batch=16,          # 根据实际测试，8是安全的，可以避免OOM风险
            workers=4,        # 根据实际测试，4是稳定的，避免内存泄漏
            device='cuda', 
            patience=300      # 禁用Early Stop：允许完整的300轮训练以观察完整的收敛曲线
        )
if __name__ == '__main__':
	freeze_support()
	main()
