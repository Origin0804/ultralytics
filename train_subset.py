from ultralytics import YOLO

# load the custom model
model = YOLO('ultralytics/cfg/models/v8/yolov8_multimodal.yaml')

# train the model with local ultralytics code
model.train(data='datasets/rgbt_subset.yaml', epochs=1, imgsz=256, batch=2, workers=0)
