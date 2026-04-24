import os
import re
from types import SimpleNamespace 
import contextlib
import numpy as np
import mindspore as ms
from mindspore.common.initializer import Initializer, Zero, initializer


from ultralytics.tools.convert import universal_convert

# 导入各大任务的 Trainer (训练器)
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.models.yolo.pose.train import PoseTrainer

# 导入各大任务的 Validator (验证器)
from ultralytics.models.yolo.classify.val import ClassificationValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.models.yolo.pose.val import PoseValidator

# 导入各大任务的 Predictor (推理器)
from ultralytics.models.yolo.classify.predict import ClassificationPredictor
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.models.yolo.pose.predict import PosePredictor

class YOLO:
    def __init__(self, model='yolo11n.pt', task=None):
        self.model_name = str(model)
        self.task = task
        self.scale = 'n'
        
        self.model = None
        self._parse_model_info()


        if self.model_name.endswith('.yaml'):
            print(f"[MindNLP YOLO] 检测到传入 YAML 架构文件: {self.model_name}")
            print(f"[MindNLP YOLO] 模式: 从头开始随机初始化训练 (跳过权重转换)。")
            self.yaml_path = self.model_name
            self.ckpt_path = ""  # 空路径表示不加载任何预训练权重
        else:
            self.yaml_path = None
            pt_path = self.model_name
            
            if self.model_name.endswith('.pt'):
                self.ckpt_path = self.model_name.replace('.pt', '.ckpt')
                pt_path = self.model_name
            elif self.model_name.endswith('.ckpt') and not os.path.exists(self.model_name):
                pt_path = self.model_name.replace('.ckpt', '.pt')
                self.ckpt_path = self.model_name
            else:
                self.ckpt_path = self.model_name

            if self.ckpt_path and not os.path.exists(self.ckpt_path):
                print(f"[MindNLP YOLO] 未找到本地权重 {self.ckpt_path}，准备获取源文件 {pt_path} 并转换...")
                universal_convert(pt_path=pt_path, 
                                  ckpt_path=self.ckpt_path, 
                                  task=self.task, 
                                  scale=self.scale)
            elif self.ckpt_path:
                print(f"[MindNLP YOLO] 发现已存在权重: {self.ckpt_path}，直接加载。")

    def _parse_model_info(self):
        """内部方法：从权重文件名中正则解析出任务类型和模型规模"""
        # 解析任务 Task
        if self.task is None:
            if '-cls' in self.model_name:
                self.task = 'classify'
            elif '-seg' in self.model_name:
                self.task = 'segment'
            elif '-pose' in self.model_name:
                self.task = 'pose'
            else:
                self.task = 'detect'  # 默认无后缀为目标检测

        # 解析规模 Scale (正则匹配 yolo11 后面的 n, s, m, l, x)
        match = re.search(r'yolo11([nsmlx])', self.model_name.lower())
        if match:
            self.scale = match.group(1)
        else:
            self.scale = 'n'  # 若未匹配到，默认使用 nano 规模

    def train(self, **kwargs):
        """统一训练路由方法"""
        current_dir = os.path.dirname(os.path.abspath(__file__)) 
        parent_dir = os.path.dirname(current_dir) 
        
        if getattr(self, 'yaml_path', None):
            kwargs['model'] = self.yaml_path
            kwargs['weights'] = ""  # 无预训练权重
        else:
            kwargs['model'] = self.ckpt_path
            kwargs['weights'] = self.ckpt_path
            
        kwargs['scale'] = getattr(self, 'scale', 'n')
        
        # 3. 智能匹配任务架构文件 
        if 'model_cfg' not in kwargs:
            cfg_folder = os.path.join(parent_dir, 'cfg', 'models', '11') 
            task_cfg_map = {
                'detect': 'yolo11.yaml',
                'segment': 'yolo11-seg.yaml',
                'pose': 'yolo11-pose.yaml',
                'classify': 'yolo11-cls.yaml'
            }
            cfg_file = task_cfg_map.get(self.task, 'yolo11.yaml')
            kwargs['model_cfg'] = os.path.join(cfg_folder, cfg_file)
            
            if not os.path.exists(kwargs['model_cfg']):
                print(f"[MindNLP YOLO 错误] 找不到架构文件: {kwargs['model_cfg']}")

        # 4. 定位超参数文件
        if 'hyp' not in kwargs:
            kwargs['hyp'] = os.path.join(parent_dir, 'cfg', 'hyp.yaml')

        # 5. 定位数据集配置
        if 'data' in kwargs and not os.path.isabs(kwargs['data']):
            data_cfg_path = os.path.join(parent_dir, 'cfg', 'datasets', kwargs['data'])
            if os.path.exists(data_cfg_path):
                kwargs['data'] = data_cfg_path
                print(f"[MindNLP YOLO] 自动定位数据集: {kwargs['data']}")

        # 6. 包装参数并路由
        if 'save_dir' not in kwargs:
            kwargs['save_dir'] = os.path.join("runs", self.task, "train")

        args_obj = SimpleNamespace(**kwargs) 
        
        print(f"[MindNLP YOLO] 准备启动 {self.task} 任务的训练...")
        
        if self.task == 'classify':
            trainer = ClassificationTrainer(args=args_obj)
        elif self.task == 'detect':
            trainer = DetectionTrainer(args=args_obj)
        elif self.task == 'segment':
            trainer = SegmentationTrainer(args=args_obj)
        elif self.task == 'pose':
            trainer = PoseTrainer(args=args_obj)
        else:
            raise ValueError(f"[MindNLP YOLO] 暂不支持的任务类型: {self.task}")
            
        trainer.train()
        if hasattr(trainer, 'model'):
            self.model = trainer.model
        elif hasattr(trainer, 'ema') and hasattr(trainer.ema, 'ema'):
            self.model = trainer.ema.ema # 如果用了指数移动平均
            
        return trainer

    def val(self, **kwargs):
        """
        统一验证路由方法。根据实例化的任务类型，拉起对应的验证器。
        """
        if self.model is None:
            # 如果内存里没有模型实体，就让验证器去读本地权重文件
            kwargs['model'] = self.ckpt_path
            pass_model = self.ckpt_path 
            print("[MindNLP YOLO] 当前内存无模型实体，将传递权重路径给 Validator 进行初始化。")
        else:
            # 如果刚刚 train 完，内存里有模型，直接传给验证器
            kwargs['model'] = self.ckpt_path # 依然保留路径作为元数据
            pass_model = self.model
        print(f"[MindNLP YOLO] 准备启动 {self.task} 任务的验证...")
        
        if self.task == 'classify':
            validator = ClassificationValidator(args=kwargs)
        elif self.task == 'detect':
            validator = DetectionValidator(args=kwargs)
        elif self.task == 'segment':
            validator = SegmentationValidator(args=kwargs)
        elif self.task == 'pose':
            validator = PoseValidator(args=kwargs)
        else:
            raise ValueError(f"[MindNLP YOLO] 暂不支持的任务类型: {self.task}")
            
        return validator(model=pass_model)

    def __call__(self, source=None, **kwargs):
        """
        统一推理路由方法
        """
        kwargs['model'] = self.ckpt_path
        kwargs['source'] = source
        print(f"[MindNLP YOLO] 准备启动 {self.task} 任务的推理...")
        
        if self.task == 'classify':
            predictor = ClassificationPredictor(cfg=kwargs)
        elif self.task == 'detect':
            predictor = DetectionPredictor(cfg=kwargs)
        elif self.task == 'segment':
            predictor = SegmentationPredictor(cfg=kwargs)
        elif self.task == 'pose':
            predictor = PosePredictor(cfg=kwargs)
        else:
            raise ValueError(f"[MindNLP YOLO] 暂不支持的任务类型: {self.task}")

        return predictor(source=source, model=self.model)
