import cv2
import random
import numpy as np
import mindspore.dataset.vision as vision

class Compose:
    """组合多种数据增强变换"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, labels=None):
        for t in self.transforms:
            if labels is None:
                img = t(img)
            else:
                img, labels = t(img, labels)
        return (img, labels) if labels is not None else img

class RandomHSV:
    """随机调整图像的色调(Hue)、饱和度(Saturation)和明度(Value)"""
    def __init__(self, hgain=0.015, sgain=0.7, vgain=0.4):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, img, labels=None):
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        dtype = img.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return (img, labels) if labels is not None else img

class RandomFlip:
    """
    随机水平翻转图像及其对应的空间标签
    支持基础目标检测边界框以及姿态估计关键点的同步翻转
    """
    def __init__(self, p=0.5, flip_idx=None):
        self.p = p
        # flip_idx 记录翻转后关键点对应的新索引位置 (如 COCO 17 个关键点的对称关系)
        self.flip_idx = flip_idx

    def __call__(self, img, labels=None):
        if random.random() < self.p:
            img = np.ascontiguousarray(img[:, ::-1])
            if labels is not None:
                # 翻转边界框中心点 X 坐标 (格式为 [cls, cx, cy, w, h, ...])
                if len(labels.shape) > 1 and labels.shape[1] >= 5:
                    labels[:, 1] = 1.0 - labels[:, 1]

                # 翻转姿态关键点
                if len(labels.shape) > 1 and labels.shape[1] > 5 and self.flip_idx is not None:
                    kpts = labels[:, 5:].reshape(labels.shape[0], -1, 3)

                    # 仅对可见的关键点执行翻转
                    mask = kpts[..., 2] > 0
                    kpts[..., 0][mask] = 1.0 - kpts[..., 0][mask]

                    # 交换左右对称的关键点索引
                    kpts = kpts[:, self.flip_idx, :]
                    labels[:, 5:] = kpts.reshape(labels.shape[0], -1)

        return (img, labels) if labels is not None else img

def get_classify_transforms(imgsz=224, is_training=True):
    """构建分类任务专用变换流水线 (基于 MindSpore Vision 算子加速)"""
    trans = []
    if is_training:
        trans += [
            vision.RandomResizedCrop(imgsz),
            vision.RandomHorizontalFlip(prob=0.5)
        ]
    else:
        trans += [
            vision.Resize(int(imgsz * 1.14)),
            vision.CenterCrop(imgsz)
        ]

    trans += [
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        vision.HWC2CHW()
    ]
    return trans

def v8_transforms(imgsz=640, hyp=None, flip_idx=None):
    """构建 YOLO 通用检测/姿态/分割任务变换组合"""
    hyp = hyp or {}
    return Compose([
        # 修复项：使用字典的 .get() 方法安全获取 YAML 参数
        RandomHSV(hgain=hyp.get('hsv_h', 0.015),
                  sgain=hyp.get('hsv_s', 0.7),
                  vgain=hyp.get('hsv_v', 0.4)),
        RandomFlip(p=hyp.get('fliplr', 0.5), flip_idx=flip_idx)
    ])