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

## Transform the SAM model to ONNX format
from segment_anything import sam_model_registry
import torch
# 选择模型，例如 "vit_b" 版本
sam = sam_model_registry["vit_b"](checkpoint="path_to_sam_checkpoint.pth")
sam.eval()

# 输入示例，用于导出
dummy_input = torch.randn(1, 3, 1024, 1024)  # 适应 SAM 的输入分辨率

# 导出 ONNX 模型
torch.onnx.export(
    sam, 
    dummy_input, 
    "sam_model.onnx",
    input_names=["input_image"],
    output_names=["output_mask"],
    opset_version=11  # 选择与 ONNX 兼容的版本
)