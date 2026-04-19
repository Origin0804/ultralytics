import torch
from ultralytics import YOLO

# 1. Load a pre-trained YOLOv8n weights
pretrained_model = YOLO("yolov8n.pt")
pretrained_dict = pretrained_model.model.state_dict()

# 2. Load our custom randomly initialized dual-stream model
multimodal_model = YOLO('ultralytics/cfg/models/v8/yolov8_multimodal.yaml')
multimodal_dict = multimodal_model.model.state_dict()

# Custom YAML Backbone mapping:
# RGB uses layer index 2-11 corresponding to single 0-9
# Thermal uses layer index 13-22 corresponding to single 0-9

updated_state_dict = {}
for k, v in multimodal_dict.items():
    # If the layer exists in pretrained with the exact same name, copy it directly (like the head)
    if k in pretrained_dict and pretrained_dict[k].shape == v.shape:
        updated_state_dict[k] = pretrained_dict[k]
    else:
        # Check if it belongs to RGB Backbone (layer 2 to 11)
        part = k.split('.')
        try:
            layer_idx = int(part[1])
            if 2 <= layer_idx <= 11: # RGB Backbone
                mapped_idx = layer_idx - 2
                mapped_k = k.replace(f"model.{layer_idx}.", f"model.{mapped_idx}.")
                if mapped_k in pretrained_dict and pretrained_dict[mapped_k].shape == v.shape:
                    updated_state_dict[k] = pretrained_dict[mapped_k]
                else:
                    updated_state_dict[k] = v
            elif 13 <= layer_idx <= 22: # Thermal Backbone
                mapped_idx = layer_idx - 13
                mapped_k = k.replace(f"model.{layer_idx}.", f"model.{mapped_idx}.")
                if mapped_k in pretrained_dict and pretrained_dict[mapped_k].shape == v.shape:
                    updated_state_dict[k] = pretrained_dict[mapped_k]
                else:
                    updated_state_dict[k] = v
            else:
                updated_state_dict[k] = v # Concat, newly shaped Convs, etc.
        except (IndexError, ValueError):
            updated_state_dict[k] = v

multimodal_model.model.load_state_dict(updated_state_dict, strict=False)
multimodal_model.save("yolov8n_multimodal_pretrained.pt")
print("✅ Weights cloned into Dual-Backbones successfully!")
