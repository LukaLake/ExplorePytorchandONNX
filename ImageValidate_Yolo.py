# Ultralytics YOLO 🚀, AGPL-3.0 license
""" 
YOLO11 分割模型 ONNXRuntime 
    功能1: 支持不用尺寸图像的输入
    功能2: 支持可视化分割结果
"""

import time
import argparse
import sys
import cv2
import numpy as np
import onnxruntime as ort
from screeninfo import get_monitors

# 新路径列表
new_paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp"
]

# 添加每个新路径到 PATH（避免重复添加）
for path in new_paths:
    if path not in sys.path:
        sys.path.insert(0, path)  # 插入到 sys.path 开头，优先搜索
 
# 定义 COCO 数据集的 80 个类别名称
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'potted plant',
    'bed', 'dining table', 'toilet', 'tvmonitor', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# 定义每个类别对应的颜色（R, G, B）
# 这里使用随机颜色，您可以根据需要自定义
np.random.seed(42)  # 为可重复性设置种子
CLASS_COLORS = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(len(CLASS_NAMES))}
 
class YOLO11Seg:
    def __init__(self, onnx_model):
        # 创建 Ort 推理会话，选择 CPU 或 GPU 提供者
        self.session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"],
        )
        print("使用提供者:", ort.get_device() )
        # 根据 ONNX 模型类型选择 Numpy 数据类型（支持 FP32 和 FP16）
        self.ndtype = np.half if self.session.get_inputs()[0].type == "tensor(float16)" else np.single

        # 获取模型的输入宽度和高度（YOLO11-seg 只有一个输入）
        self.model_height, self.model_width = [x.shape for x in self.session.get_inputs()][0][-2:]

        # 打印模型的输入尺寸
        print("YOLO11 🚀 实例分割 ONNXRuntime")
        print("模型名称：", onnx_model)
        print(f"模型输入尺寸：宽度 = {self.model_width}, 高度 = {self.model_height}")

        # 加载类别名称
        self.classes = CLASS_NAMES

        # 加载类别对应的颜色
        self.class_colors = CLASS_COLORS
 
    def get_color_for_class(self, class_id):
        return self.class_colors.get(class_id, (255, 255, 255))  # 如果没有找到类别颜色，返回白色
 
    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45, nm=32):
        """
        完整的推理流程：预处理 -> 推理 -> 后处理
        Args:
            im0 (Numpy.ndarray): 原始输入图像
            conf_threshold (float): 置信度阈值
            iou_threshold (float): NMS 中的 IoU 阈值
            nm (int): 掩膜数量
        Returns:
            boxes (List): 边界框列表
            segments (List): 分割区域列表
            masks (np.ndarray): [N, H, W] 输出掩膜
        """
        # 图像预处理
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)
 
        # ONNX 推理
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})
 
        # 后处理
        boxes, segments, masks = self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=nm,
        )
        return boxes, segments, masks
 
    def preprocess(self, img):
        """
        图像预处理
        Args:
            img (Numpy.ndarray): 输入图像
        Returns:
            img_process (Numpy.ndarray): 处理后的图像
            ratio (tuple): 宽高比例
            pad_w (float): 宽度的填充
            pad_h (float): 高度的填充
        """
        # 调整输入图像大小并使用 letterbox 填充
        shape = img.shape[:2]  # 原始图像大小
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # 填充宽高
        if shape[::-1] != new_unpad:  # 调整图像大小
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
 
        # 转换：HWC -> CHW -> BGR 转 RGB -> 除以 255 -> contiguous -> 添加维度
        img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)
 
    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        """
        推理后的结果后处理
        Args:
            preds (Numpy.ndarray): 来自 ONNX 的推理结果
            im0 (Numpy.ndarray): [h, w, c] 原始输入图像
            ratio (tuple): 宽高比例
            pad_w (float): 宽度的填充
            pad_h (float): 高度的填充
            conf_threshold (float): 置信度阈值
            iou_threshold (float): IoU 阈值
            nm (int): 掩膜数量
        Returns:
            boxes (List): 边界框列表
            segments (List): 分割区域列表
            masks (np.ndarray): 掩膜数组
        """
        x, protos = preds[0], preds[1]  # 获取模型的两个输出：预测和原型

        # 输出x的前116个数据
        print("输出x的前15个数据：", x[0][0][:15])
        
        # 转换维度
        x = np.einsum("bcn->bnc", x)
        print("输出x的前15个数据：", x[0][0][:15])

        # 输出x形状
        print("输出x形状：", x.shape)
 
        # 置信度过滤
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        print("置信度过滤后的x形状：", x.shape)
        
 
        # 合并边界框、置信度、类别和掩膜
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]
 
        # NMS 过滤
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]
 
        # 解析并返回结果
        if len(x) > 0:
            # 边界框格式转换：从 cx cy w h -> x y x y
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]
 
            # 缩放边界框，使其与原始图像尺寸匹配
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)
 
            # 限制边界框在图像边界内
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])
 
            # 处理掩膜
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)
 
            # 将掩膜转换为分割区域
            segments = self.masks2segments(masks)
            return x[..., :6], segments, masks  # 返回边界框、分割区域和掩膜
        else:
            return [], [], []
 
    @staticmethod
    def masks2segments(masks):
        """
        将掩膜转换为分割区域
        Args:
            masks (numpy.ndarray): 模型输出的掩膜，形状为 (n, h, w)
        Returns:
            segments (List): 分割区域的列表
        """
        segments = []
        for x in masks.astype("uint8"):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # 找到轮廓
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # 如果没有找到分割区域，返回空数组
            segments.append(c.astype("float32"))
        return segments
 
    @staticmethod
    def crop_mask(masks, boxes):
        """
        裁剪掩膜，使其与边界框对齐
        Args:
            masks (Numpy.ndarray): [n, h, w] 掩膜数组
            boxes (Numpy.ndarray): [n, 4] 边界框
        Returns:
            (Numpy.ndarray): 裁剪后的掩膜
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
 
    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        处理模型输出的掩膜
        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w] 掩膜原型
            masks_in (numpy.ndarray): [n, mask_dim] 掩膜数量
            bboxes (numpy.ndarray): 缩放到原始图像尺寸的边界框
            im0_shape (tuple): 原始输入图像的尺寸 (h,w,c)
        Returns:
            (numpy.ndarray): 处理后的掩膜
        """
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # 将掩膜从 P3 尺寸缩放到原始输入图像大小
        masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)  # 裁剪掩膜
        return np.greater(masks, 0.5)  # 返回二值化后的掩膜
 
    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        将掩膜缩放至原始图像大小
        Args:
            masks (np.ndarray): 缩放和填充后的掩膜
            im0_shape (tuple): 原始图像大小
            ratio_pad (tuple): 填充与原始图像的比例
        Returns:
            masks (np.ndarray): 缩放后的掩膜
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # 计算比例
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # 比例
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # 填充
        else:
            pad = ratio_pad[1]
 
        # 计算掩膜的边界
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" 应该是 2 或 3，但得到 {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # 使用 INTER_LINEAR 插值调整大小
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks
 
    def draw_and_visualize(self, im, bboxes, segments, vis=False, save=True):
        """
        绘制和可视化结果
        Args:
            im (np.ndarray): 原始图像，形状为 [h, w, c]
            bboxes (numpy.ndarray): [n, 6]，n 是边界框数量
            segments (List): 分割区域的列表
            vis (bool): 是否使用 OpenCV 显示图像
            save (bool): 是否保存带注释的图像
        Returns:
            None
        """
        # 创建图像副本
        im_canvas = im.copy()

        for (*box, conf, cls_id), segment in zip(bboxes, segments):
            cls_id = int(cls_id)
            if cls_id >= len(self.classes):
                class_name = 'unknown'
                color = (255, 255, 255)  # 白色
            else:
                class_name = self.classes[cls_id]
                color = self.class_colors.get(cls_id, (255, 255, 255))  # 默认白色

            label = f"{class_name}: {conf:.3f}"

            # 绘制轮廓和填充掩膜
            if len(segment) > 0:
                cv2.polylines(im, [np.int32(segment)], True, color, 2)  # 绘制边框
                cv2.fillPoly(im_canvas, [np.int32(segment)], color)  # 使用类别对应的颜色填充多边形

            # 绘制边界框
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(im, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # 在图像上绘制类别名称和置信度
            cv2.putText(im, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # 将图像和绘制的多边形混合
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

        # 获取屏幕尺寸
        monitor = get_monitors()[0]  # 获取第一个监视器的信息
        screen_width, screen_height = monitor.width, monitor.height

        # 设置窗口尺寸为屏幕尺寸的90%
        window_width, window_height = int(screen_width * 0.9), int(screen_height * 0.9)

        # 计算图像缩放比例
        img_height, img_width = im.shape[:2]
        scale = min(window_width / img_width, window_height / img_height, 1.0)  # 不放大图像

        # 调整图像大小以适应窗口
        if scale < 1.0:
            new_size = (int(img_width * scale), int(img_height * scale))
            im = cv2.resize(im, new_size, interpolation=cv2.INTER_AREA)

        # 创建可调整大小的窗口并设置尺寸
        if vis:
            cv2.namedWindow("Segmentation Result", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Segmentation Result", im.shape[1], im.shape[0])  # 设置窗口大小为图像大小
            cv2.imshow("Segmentation Result", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # 保存图像
        if save:
            cv2.imwrite("seg_result_picture.jpg", im)
            print("结果已保存为 'seg_result_picture.jpg'。")

 
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=r"Models/yolo11n-seg.onnx" , help="ONNX 模型路径")
    parser.add_argument("--source", type=str, default=r"Images/Test11.png", help="输入图像路径")
    parser.add_argument("--conf", type=float, default=0.7, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS 的 IoU 阈值")
    args = parser.parse_args()
 
    # 加载模型
    model = YOLO11Seg(args.model)
 
    # 使用 OpenCV 读取图像
    img = cv2.imread(args.source)

    # 计算推理时间

    start_time = time.time()
    # 模型推理
    boxes, segments, _ = model(img, conf_threshold=args.conf, iou_threshold=args.iou)
 
    # 如果检测到目标，绘制边界框和分割区域
    if len(boxes) > 0:
        model.draw_and_visualize(img, boxes, segments, vis=False, save=True)
    else:
        print("未检测到目标。")

    # 计算推理时间
    end_time = time.time()
    print("推理时间：", end_time - start_time, "秒。")