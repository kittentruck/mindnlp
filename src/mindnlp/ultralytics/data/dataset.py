import os
import cv2
import numpy as np
from pathlib import Path

class YOLOBaseDataset:
    """数据集基类，提供通用的图像读取与空间几何变换方法"""
    def __init__(self, imgsz=640):
        self.imgsz = imgsz

    def load_image(self, img_path):
        """读取图像并转换为 RGB 格式。"""
        img = cv2.imread(str(img_path))
        if img is None:
            return np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    def letterbox(self, img, bboxes=None, keypoints=None, segments=None, color=(114, 114, 114)):
        """
        几何变换核心逻辑：保持长宽比缩放并填充边缘
        同步支持边界框 (Bbox)、关键点 (Keypoints) 及多边形 (Segments) 的坐标映射与归一化
        """
        shape = img.shape[:2]
        h0, w0 = shape
        r = min(self.imgsz / h0, self.imgsz / w0)
        new_unpad = int(round(w0 * r)), int(round(h0 * r))
        dw = (self.imgsz - new_unpad[0]) / 2  
        dh = (self.imgsz - new_unpad[1]) / 2  

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        # 边界框坐标映射与截断保护 (防止因 Padding 导致坐标溢出 1.0)
        if bboxes is not None and len(bboxes) > 0:
            bboxes[:, [0, 2]] *= new_unpad[0]
            bboxes[:, [1, 3]] *= new_unpad[1]
            bboxes[:, 0] += dw
            bboxes[:, 1] += dh
            bboxes /= self.imgsz
            bboxes = np.clip(bboxes, 0.0, 1.0) 

        # 关键点坐标映射
        if keypoints is not None and len(keypoints) > 0:
            mask = keypoints[..., 2] > 0 
            keypoints[..., 0][mask] = keypoints[..., 0][mask] * new_unpad[0] + dw
            keypoints[..., 1][mask] = keypoints[..., 1][mask] * new_unpad[1] + dh
            keypoints[..., 0][mask] /= self.imgsz
            keypoints[..., 1][mask] /= self.imgsz
            keypoints[..., :2] = np.clip(keypoints[..., :2], 0.0, 1.0)

        # 实例分割多边形坐标映射
        if segments is not None and len(segments) > 0:
            for i in range(len(segments)):
                segments[i][:, 0] *= new_unpad[0]
                segments[i][:, 1] *= new_unpad[1]
                segments[i][:, 0] += dw
                segments[i][:, 1] += dh
                segments[i] /= self.imgsz

        result = [img]
        if bboxes is not None: result.append(bboxes)
        if keypoints is not None: result.append(keypoints)
        if segments is not None: result.append(segments)
        
        return tuple(result) if len(result) > 1 else img


class YOLOClassifyDataset(YOLOBaseDataset):
    """图像分类数据集接口"""
    def __init__(self, root, imgsz=224):
        super().__init__(imgsz)
        self.root = Path(root)
        
        # 基于 train 目录建立类别索引映射，确保 train 与 val 映射关系一致
        data_root = self.root.parent
        train_dir = data_root / 'train'
        reference_dir = train_dir if train_dir.exists() else self.root
        
        self.classes = sorted([d.name for d in reference_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue 
            
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), self.class_to_idx[cls]))
        
        if not self.samples:
            raise RuntimeError(f"[ERROR] 目录 {self.root} 中未检测到有效图像样本。")

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = self.load_image(path)
        return img, np.array(label, dtype=np.int32)

    def __len__(self):
        return len(self.samples)


class YOLODetectDataset(YOLOBaseDataset):
    """目标检测数据集接口"""
    def __init__(self, img_path, imgsz=640, transforms=None):
        super().__init__(imgsz)
        self.img_path = Path(img_path)
        self.im_files = sorted([str(f) for f in self.img_path.rglob("*") 
                               if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
        
        # 基于目录结构约定，由图像路径推导标签路径
        self.label_files = [p.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
                             .replace(".jpg", ".txt").replace(".png", ".txt") 
                             for p in self.im_files]
        self.transforms = transforms

    def __getitem__(self, index):
        img = self.load_image(self.im_files[index])
        label_path = self.label_files[index]
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                content = f.read().strip()
                l = np.array([x.split() for x in content.splitlines()], dtype=np.float32) if content else np.zeros((0, 5), dtype=np.float32)
        else:
            l = np.zeros((0, 5), dtype=np.float32)

        if self.transforms is not None and len(l) > 0:
            img, l = self.transforms(img, l)    

        cls = l[:, 0:1] if len(l) else np.zeros((0, 1), dtype=np.float32)
        bboxes = l[:, 1:5] if len(l) else np.zeros((0, 4), dtype=np.float32)
        
        img, bboxes = self.letterbox(img, bboxes=bboxes)
        return img, cls, bboxes

    def __len__(self):
        return len(self.im_files)


class YOLOSegmentDataset(YOLOBaseDataset):
    """实例分割数据集接口"""
    def __init__(self, img_path, imgsz=640, transforms=None):
        super().__init__(imgsz)
        self.img_path = Path(img_path)
        self.im_files = sorted([str(f) for f in self.img_path.rglob("*") 
                               if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
        self.label_files = [p.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
                             .replace(".jpg", ".txt").replace(".png", ".txt") 
                             for p in self.im_files]
        self.transforms = transforms 

    def __getitem__(self, index):
        img = self.load_image(self.im_files[index])
        label_path = self.label_files[index]
        
        cls_list, segments, raw_labels = [], [], []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.read().strip().splitlines():
                    parts = list(map(float, line.split()))
                    if len(parts) >= 5: 
                        raw_labels.append(parts)

        # 当前针对分割任务仅执行像素级的数据增强（如 HSV 变换），不执行空间几何变换
        if self.transforms is not None and len(raw_labels) > 0:
             img = self.transforms(img) 
             
        for parts in raw_labels:
            cls_list.append([parts[0]])
            poly = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            segments.append(poly)

        if len(segments) > 0:
            img, segments = self.letterbox(img, segments=segments)
        else:
            img = self.letterbox(img) 

        bboxes, masks = [], []
        # 由经过归一化缩放后的多边形坐标生成掩码拓扑和边界框
        for poly in segments:
            x1, y1 = poly[:, 0].min(), poly[:, 1].min()
            x2, y2 = poly[:, 0].max(), poly[:, 1].max()
            bboxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
            
            mask = np.zeros((self.imgsz, self.imgsz), dtype=np.uint8)
            poly_pixel = (poly * [self.imgsz, self.imgsz]).astype(np.int32)
            cv2.fillPoly(mask, [poly_pixel], 1)
            masks.append(mask)

        if not cls_list:
            return img, np.zeros((0, 1), dtype=np.float32), np.zeros((0, 4), dtype=np.float32), np.zeros((0, self.imgsz, self.imgsz), dtype=np.float32)

        return (img, 
                np.array(cls_list, dtype=np.float32), 
                np.array(bboxes, dtype=np.float32), 
                np.array(masks, dtype=np.float32))
    
    def __len__(self):
        return len(self.im_files)


class YOLOPoseDataset(YOLOBaseDataset):
    """
    姿态估计数据集接口
    预期标签格式: [class_id, x, y, w, h, kpt1_x, kpt1_y, kpt1_v, ...]
    """
    def __init__(self, img_path, imgsz=640, nkpt=17, ndim=3, transforms=None):
        super().__init__(imgsz)
        self.img_path = Path(img_path)
        self.im_files = sorted([str(f) for f in self.img_path.rglob("*") 
                               if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
        self.label_files = [p.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
                             .replace(".jpg", ".txt").replace(".png", ".txt") 
                             for p in self.im_files]
        self.nkpt = nkpt
        self.ndim = ndim 
        self.transforms = transforms

    def __getitem__(self, index):
        img = self.load_image(self.im_files[index])
        label_path = self.label_files[index]
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                content = f.read().strip()
                labels = np.array([x.split() for x in content.splitlines()], dtype=np.float32) if content else np.zeros((0, 5 + self.nkpt * self.ndim), dtype=np.float32)
        else:
            labels = np.zeros((0, 5 + self.nkpt * self.ndim), dtype=np.float32)

        if self.transforms is not None and len(labels) > 0:
            img, labels = self.transforms(img, labels)    

        if len(labels):
            cls = labels[:, 0:1]
            bboxes = labels[:, 1:5]
            keypoints = labels[:, 5:].reshape(-1, self.nkpt, self.ndim) 
        else:
            cls = np.zeros((0, 1), dtype=np.float32)
            bboxes = np.zeros((0, 4), dtype=np.float32)
            keypoints = np.zeros((0, self.nkpt, self.ndim), dtype=np.float32)

        img, bboxes, keypoints = self.letterbox(img, bboxes=bboxes, keypoints=keypoints)

        return img, cls, bboxes, keypoints

    def __len__(self):
        return len(self.im_files)