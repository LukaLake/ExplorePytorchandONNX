# ExplorePytorchandONNX
 Explore application using pytorch and onnx frame

ONNX框架在UE5中更加泛用，特别是使用NNE插件时支持直接导入ONNX文件、
因此先使用pytorch训练后，导出为onnx

利用ultralytics读入/下载yolov8s，转换后如下
```
ONNX: starting export with onnx 1.16.0 opset 12...
ONNX: slimming with onnxslim 0.1.35...
ONNX: export success ✅ 23.0s, saved as 'yolov8s.onnx' (42.8 MB)

Export complete (23.4s)
Results saved to C:\Users\Administrator\Documents\GitHub\ExplorePytorchandONNX
Predict:         yolo predict task=detect model=yolov8s.onnx imgsz=640
Validate:        yolo val task=detect model=yolov8s.onnx imgsz=640 data=coco.yaml
Visualize:       https://netron.app
Model successfully converted to ONNX format.
```