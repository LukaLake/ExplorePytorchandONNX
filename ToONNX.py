from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolo11m-seg.pt')

# Move the model to CPU
model.to('cpu')

# Export the model to ONNX format with static input shapes and without simplification
model.export(
    format='onnx',
    opset=12,
    simplify=False,
    dynamic=False,
)

print("Model successfully converted to ONNX format.")
