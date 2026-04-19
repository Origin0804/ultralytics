from ultralytics.models.yolo.model import YOLO
model = YOLO('ultralytics/cfg/models/v8/yolov8_multimodal.yaml')
print(model.info())
