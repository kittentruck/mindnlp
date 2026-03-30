import os
import sys
import argparse
import logging
import gc
import mindspore as ms

# 配置自定义日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

LOGGER = logging.getLogger("YOLO11-Pipeline")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ultralytics import YOLO

if __name__ == "__main__":
    
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

    LOGGER.info(" 开始初始化 YOLO11 模型...")

    #微调
    model = YOLO("yolo11n.pt")
    #从头开始训练
    #model = YOLO("yolo11.yaml")

    LOGGER.info(" 启动训练流程...")

    #  训练模型
    results = model.train(
        data="coco128.yaml", 
        epochs=100, 
        imgsz=640, 
        batch=16, 
        amp=False, 
        val_interval=10,
    )

    # 打印最佳权重精度 
    LOGGER.info(f" 训练完成！最高综合评价指标 (Fitness): {results.best_fitness:.4f}")

    # 内存清理
    gc.collect()

    #  执行推理预测
    source_img = "./datasets/coco128/images/train2017"
    LOGGER.info(f" 开始推理预测: {source_img}")

    predict_results = model(
        source=source_img,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        save=True,
    )

    LOGGER.info(f" 推理完成！结果保存在: {predict_results[0].save_dir}")