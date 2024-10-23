import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Export the model to ONNX format
success = model.export(format='onnx', opset=12)

if success:
    print("Model successfully converted to ONNX format.")
