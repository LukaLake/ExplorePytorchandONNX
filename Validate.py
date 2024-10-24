import onnx
import onnxruntime as ort

# Load and check the model
onnx_model = onnx.load('yolov8s.onnx')
onnx.checker.check_model(onnx_model)

# Create an ONNX Runtime session
ort_session = ort.InferenceSession('yolov8s.onnx')

# Prepare a dummy input matching the model's expected input shape
import numpy as np
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

# Run inference
outputs = ort_session.run(None, {'images': dummy_input})

print("ONNX Runtime inference successful.")
