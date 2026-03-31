import os
import cv2
import glob
import numpy as np
import mindspore.dataset as ds
from pathlib import Path
from .dataset import (YOLOClassifyDataset, YOLODetectDataset, 
                      YOLOSegmentDataset, YOLOPoseDataset)
from .augment import get_classify_transforms, v8_transforms

# Collate 聚合函数：针对各类任务生成特定的 batch_idx
def yolo_collate_fn(imgs, clss, bboxes, batch_info=None):
    """检测任务批量重组映射函数"""
    batch_imgs = np.stack(imgs, axis=0)
    batch_imgs = batch_imgs.transpose(0, 3, 1, 2)
    batch_imgs = batch_imgs.astype(np.float32) 
    batch_cls, batch_bboxes, batch_idx = [], [], []
    
    for i in range(len(clss)):
        n = clss[i].shape[0]
        if n > 0:
            batch_cls.append(clss[i])
            batch_bboxes.append(bboxes[i])
            batch_idx.append(np.full((n, 1), i, dtype=np.float32))
            
    if len(batch_cls) > 0:
        batch_cls = np.concatenate(batch_cls, axis=0)
        batch_bboxes = np.concatenate(batch_bboxes, axis=0)
        batch_idx = np.concatenate(batch_idx, axis=0)
    else:
        batch_cls = np.zeros((0, 1), dtype=np.float32)
        batch_bboxes = np.zeros((0, 4), dtype=np.float32)
        batch_idx = np.zeros((0, 1), dtype=np.float32)
        
    return batch_imgs, batch_cls, batch_bboxes, batch_idx


def pose_collate_fn(imgs, clss, bboxes, kpts, batch_info=None):
    """姿态估计任务批量重组映射函数"""
    batch_imgs = np.stack(imgs, axis=0)
    batch_imgs = batch_imgs.transpose(0, 3, 1, 2)
    batch_imgs = batch_imgs.astype(np.float32)
    batch_cls, batch_bboxes, batch_kpts, batch_idx = [], [], [], []
    
    for i in range(len(clss)):
        n = clss[i].shape[0]
        if n > 0:
            batch_cls.append(clss[i])
            batch_bboxes.append(bboxes[i])
            batch_kpts.append(kpts[i])
            batch_idx.append(np.full((n, 1), i, dtype=np.float32))
            
    if len(batch_cls) > 0:
        batch_cls = np.concatenate(batch_cls, axis=0)
        batch_bboxes = np.concatenate(batch_bboxes, axis=0)
        batch_kpts = np.concatenate(batch_kpts, axis=0)
        batch_idx = np.concatenate(batch_idx, axis=0)
    else:
        batch_cls = np.zeros((0, 1), dtype=np.float32)
        batch_bboxes = np.zeros((0, 4), dtype=np.float32)
        batch_kpts = np.zeros((0, 17, 3), dtype=np.float32)
        batch_idx = np.zeros((0, 1), dtype=np.float32)
        
    return batch_imgs, batch_cls, batch_bboxes, batch_kpts, batch_idx


def segment_collate_fn(imgs, clss, bboxes, masks, batch_info=None):
    """实例分割任务批量重组映射函数"""
    batch_imgs = np.stack(imgs, axis=0)
    if batch_imgs.shape[-1] == 3:  
        batch_imgs = batch_imgs.transpose(0, 3, 1, 2)
    batch_imgs = batch_imgs.astype(np.float32) 
    batch_cls, batch_bboxes, batch_masks, batch_idx = [], [], [], []
    
    for i in range(len(clss)):
        n = clss[i].shape[0]
        if n > 0:
            batch_cls.append(clss[i])
            batch_bboxes.append(bboxes[i])
            batch_masks.append(masks[i])
            batch_idx.append(np.full((n, 1), i, dtype=np.int32))
            
    if len(batch_cls) > 0:
        batch_cls = np.concatenate(batch_cls, axis=0)
        batch_bboxes = np.concatenate(batch_bboxes, axis=0)
        batch_masks = np.concatenate(batch_masks, axis=0)
        batch_idx = np.concatenate(batch_idx, axis=0)
    else:
        batch_cls = np.zeros((0, 1), dtype=np.float32)
        batch_bboxes = np.zeros((0, 4), dtype=np.float32)
        batch_masks = np.zeros((0, 160, 160), dtype=np.float32)
        batch_idx = np.zeros((0, 1), dtype=np.float32)
        
    return batch_imgs.astype(np.float32), batch_cls, batch_bboxes, batch_masks, batch_idx


# 数据加载器组装器
def create_dataloader(path, imgsz=640, batch_size=16, task='classify', is_training=True, num_workers=8, hyp=None):
    """构建多任务兼容的 MindSpore 数据预处理流水线"""
    
    # 针对姿态任务所需的 COCO 对称点映射字典
    coco_flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15] 
    
    # 动态注入数据增强策略 (验证阶段则保持原图特征提取)
    common_transforms = v8_transforms(imgsz=imgsz, hyp=hyp, flip_idx=coco_flip_idx) if is_training else None

    if task == 'classify':
        dataset_generator = YOLOClassifyDataset(path, imgsz=imgsz)
        column_names = ["image", "label"]
    elif task == 'detect':
        dataset_generator = YOLODetectDataset(path, imgsz=imgsz, transforms=common_transforms)
        source_column_names = ["image", "cls", "bboxes"]
    elif task == 'segment':
        dataset_generator = YOLOSegmentDataset(path, imgsz=imgsz, transforms=common_transforms)
        source_column_names = ["image", "cls", "bboxes", "masks"]
    elif task == 'pose':
        dataset_generator = YOLOPoseDataset(path, imgsz=imgsz, transforms=common_transforms)
        source_column_names = ["image", "cls", "bboxes", "keypoints"]
    else:
        raise ValueError(f"[ERROR] 尚不支持的任务类型: {task}")

    # 封装为 MindSpore GeneratorDataset 并激活多进程加载引擎
    dataset = ds.GeneratorDataset(
        source=dataset_generator, 
        column_names=source_column_names if task != 'classify' else column_names, 
        shuffle=is_training,
        num_parallel_workers=num_workers if num_workers > 0 else 1,
        python_multiprocessing=True 
    )

    if task == 'classify':
        trans = get_classify_transforms(imgsz=imgsz, is_training=is_training)
        dataset = dataset.map(
            operations=trans, 
            input_columns="image", 
            num_parallel_workers=num_workers
        )
        dataset = dataset.project(column_names)
        dataset = dataset.batch(batch_size, drop_remainder=is_training, num_parallel_workers=num_workers)
    
    elif task == 'detect':
        dataset = dataset.project(source_column_names)
        dataset = dataset.batch(
            batch_size, 
            per_batch_map=yolo_collate_fn, 
            output_columns=["image", "cls", "bboxes", "batch_idx"],
            drop_remainder=is_training
        )
        
    elif task == 'segment':
        dataset = dataset.project(source_column_names)
        dataset = dataset.batch(
            batch_size, 
            per_batch_map=segment_collate_fn, 
            input_columns=source_column_names, 
            output_columns=["image", "cls", "bboxes", "masks", "batch_idx"],
            drop_remainder=is_training,
            num_parallel_workers=num_workers
        )
        
    elif task == 'pose':
        dataset = dataset.project(source_column_names)
        dataset = dataset.batch(
            batch_size, 
            per_batch_map=pose_collate_fn, 
            output_columns=["image", "cls", "bboxes", "keypoints", "batch_idx"],
            drop_remainder=is_training
        )
        
    return dataset


# 推理阶段专用精简加载器
def letterbox_classify(im, new_shape=224):
    """分类图像的预处理：等比例缩放短边后，进行中心裁剪"""
    h, w = im.shape[:2]
    r = new_shape / min(h, w) 
    h_new, w_new = int(h * r), int(w * r)
    im = cv2.resize(im, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    
    top = (h_new - new_shape) // 2
    left = (w_new - new_shape) // 2
    return im[top : top + new_shape, left : left + new_shape]

def letterbox_pad(img, new_shape=(640, 640), color=(114, 114, 114)):
    """检测/分割/姿态图像的预处理：等比例缩放后，进行边缘填充 (Padding)"""
    shape = img.shape[:2] 
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  
    dw, dh = dw / 2, dh / 2  

    if shape[::-1] != new_unpad:  
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

class LoadImages:
    """
    适用于本地部署的前向推理轻量化加载器
    基于迭代器模式设计，内存占用小，支持多任务预处理路由
    """
    def __init__(self, path, imgsz=640, task='detect'):
        p = str(Path(path).absolute())
        if os.path.isdir(p):
            self.files = sorted(glob.glob(os.path.join(p, '*.*')))
        elif os.path.isfile(p):
            self.files = [p]
        else:
            raise FileNotFoundError(f"[ERROR] 未能索引至有效的文件路径: {p}")
            
        self.imgsz = imgsz
        self.task = task  # 记录当前任务类型
        self.count = 0
        self.nf = len(self.files)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        
        path = self.files[self.count]
        self.count += 1
        
        img = cv2.imread(path)
        if img is None:
            print(f"[WARNING] 无法正常解析图像源文件: {path}，予以跳过。")
            return self.__next__()
            
        # 保留一份原始图像，供后续画框、保存使用
        img0 = img.copy() 
        
        # BGR 转换为 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 核心逻辑路由：根据任务类型执行不同的几何预处理
        if self.task == 'classify':
            img = letterbox_classify(img, new_shape=self.imgsz)
        else:
            img = letterbox_pad(img, new_shape=(self.imgsz, self.imgsz))
        
        return path, img, img0

    def __len__(self):
        return self.nf