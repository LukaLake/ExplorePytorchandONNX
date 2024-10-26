import onnxruntime as ort

# 创建 ONNX Runtime 会话
ort_session = ort.InferenceSession('yolo11n-seg.onnx')

# 获取模型输入信息
input_meta = ort_session.get_inputs()[0]
print("输入名称:", input_meta.name)
print("输入形状:", input_meta.shape)
print("输入数据类型:", input_meta.type)
