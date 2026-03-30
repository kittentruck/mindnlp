import os
import sys
import argparse
import numpy as np
import mindspore as ms
from mindspore import Tensor
import subprocess

project_root = os.path.abspath(__file__)
while os.path.basename(project_root) != 'ultralytics' and project_root != '/':
    project_root = os.path.dirname(project_root)
project_root = os.path.dirname(project_root) 

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ultralytics import modeling_yolo
from ultralytics.configuration_yolo import YOLOConfig

def translate_ms_to_pt(ms_name, task):
    """
    参数映射路由表：将 MindSpore 架构下的参数名称转换为 PyTorch 架构对应的参数名称
    """
    pt_name = ms_name.replace(".moving_mean", ".running_mean") \
                     .replace(".moving_variance", ".running_var") \
                     .replace(".gamma", ".weight") \
                     .replace(".beta", ".bias")
                     
    pt_name = pt_name.replace(".conv1.", ".cv1.")
    pt_name = pt_name.replace(".conv2.", ".cv2.")
    pt_name = pt_name.replace(".conv3.", ".cv3.")
    
    return pt_name

def universal_convert(pt_path, ckpt_path, task="detect", scale="n"):
    """
    通用权重转换流水线。
    支持将官方 PyTorch (.pt) 转换为符合 MindSpore (.ckpt) 的权重。
    """
    model_map = {
        "classify": modeling_yolo.YOLO11ForClassification,
        "detect": modeling_yolo.YOLO11ForObjectDetection,
        "segment": modeling_yolo.YOLO11ForSegmentation,
        "pose": modeling_yolo.YOLO11ForPose
    }
    yaml_map = {
        "classify": "cfg/models/11/yolo11-cls.yaml", 
        "detect": "cfg/models/11/yolo11.yaml", 
        "segment": "cfg/models/11/yolo11-seg.yaml",
        "pose": "cfg/models/11/yolo11-pose.yaml"
    }
    nc_map = {"classify": 1000, "detect": 80, "segment": 80, "pose": 1} 

    print(f"[INFO] 启动 YOLO11-{task.upper()} 权重转换流程...")
    
    current_yaml = yaml_map[task]
    
    # 1. 动态构建网络配置与 MindSpore 模型实例
    if task == "pose":
        cfg = YOLOConfig(yaml_path=current_yaml, scale=scale, nc=nc_map[task], kpt_shape=[17, 3])
    else:
        cfg = YOLOConfig(yaml_path=current_yaml, scale=scale, nc=nc_map[task])
        
    ms_model = model_map[task](cfg)
    ms_model.set_train(False)

    # 通过子进程调用系统里的正版 PyTorch，提取纯 NumPy 数据
    npz_file = pt_path + ".npz"
    if not os.path.exists(npz_file):
        print(f"[INFO] 正在启动独立子进程，跨越物理隔离提取纯净权重...")
        extract_script = f"""
import torch
from ultralytics import YOLO
import numpy as np

# 此处是在干净的子进程中，载入的是官方原版环境
print("  -> [子进程] 正在加载官方 PyTorch 模型: {pt_path}")
model = YOLO('{pt_path}')
state_dict = model.model.state_dict()

# 剔除 PyTorch 依赖，将其彻底降维成通用 NumPy 格式
print("  -> [子进程] 正在降维并持久化至 NumPy 数组...")
np_dict = {{k: v.cpu().numpy() for k, v in state_dict.items()}}
np.savez('{npz_file}', **np_dict)
"""
        # 将子进程代码写入临时文件并执行
        with open("temp_extract.py", "w") as f:
            f.write(extract_script.strip())
        
        # 运行子进程提取数据
        subprocess.run(["python", "temp_extract.py"], check=True)
        os.remove("temp_extract.py")

    # 2. 回到主进程：读取绝对纯净的 NumPy 字典，彻底断绝与 PyTorch 的瓜葛
    print(f"[INFO] 正在解析纯净权重容器: {npz_file}")
    pt_dict = dict(np.load(npz_file))

    new_ms_ckpt = []
    matched_count = 0
    ms_params = list(ms_model.parameters_and_names())
    
    print("[INFO] 开始执行参数映射与维度校验...")
    
    # 3. 执行转换主循环
    for ms_name, ms_param in ms_params:
        ms_shape = tuple(ms_param.shape)
        ms_size = ms_param.size
        
        expected_pt_name = translate_ms_to_pt(ms_name, task)

        if expected_pt_name in pt_dict:
            pt_v = pt_dict[expected_pt_name]
            
            # 注意：pt_v 现在已经是 NumPy 数组了，所以用 .size 而不是 numel()
            if pt_v.size == ms_size:
                val_np = pt_v.reshape(ms_shape)
                
                if "moving_variance" in ms_name:
                    val_np = np.maximum(val_np, 1e-5)
                    
                ms_param.set_data(Tensor(val_np, ms.float32))
                new_ms_ckpt.append({'name': ms_name, 'data': ms_param.data})
                
                print(f"[对齐成功] {ms_name:<55} <- {expected_pt_name}")
                matched_count += 1
                del pt_dict[expected_pt_name] 
            else:
                print(f"[形状冲突] 参数 {ms_name} 期望尺寸 {ms_shape}，实际载入尺寸 {tuple(pt_v.shape)}")
        else:
            if "stride" in ms_name or "dfl.conv.weight" in ms_name:
                new_ms_ckpt.append({'name': ms_name, 'data': ms_param.data})
                print(f"[保留原值] {ms_name:<55} (框架内置固定张量)")
                matched_count += 1

    # 4. 输出审计并保存
    print("-" * 80)
    print("[INFO] 权重转换审计清单")
    matched_names = [x['name'] for x in new_ms_ckpt]
    failed_names = [n for n, _ in ms_params if n not in matched_names]
    
    if not failed_names: 
        print("  校验通过：所有模型参数均已成功映射。")
    else: 
        print(f"  校验警告：存在 {len(failed_names)} 个未匹配参数，请检查拓扑结构：")
        for fn in failed_names: 
            print(f"    - {fn}")

    ms.save_checkpoint(new_ms_ckpt, ckpt_path)
    print(f"[INFO] 转换结束。参数对齐率: {matched_count}/{len(ms_params)}。已序列化至: {ckpt_path}")
    
    # 清理打工完毕的临时 NumPy 文件
    if os.path.exists(npz_file):
        os.remove(npz_file)
        
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11 参数转换工具 (PyTorch to MindSpore Checkpoint)")
    
    parser.add_argument("--pt_path", type=str, default=None, help="输入的 .pt 文件路径")
    parser.add_argument("--ckpt_path", type=str, default=None, help="输出的 .ckpt 文件路径")
    
    parser.add_argument("--task", "-t", type=str, default="detect", 
                        choices=["classify", "detect", "segment", "pose"],
                        help="目标模型的基础任务类型")
    parser.add_argument("--scale", "-s", type=str, default="n", 
                        choices=["n", "s", "m", "l", "x"],
                        help="指定模型的规模缩放因子") 
    args = parser.parse_args()

    if args.pt_path is None:
        suffix_map = {"classify": "-cls", "segment": "-seg", "pose": "-pose", "detect": ""}
        args.pt_path = f"yolo11{args.scale}{suffix_map[args.task]}.pt"
        
    if args.ckpt_path is None:
        args.ckpt_path = args.pt_path.replace(".pt", ".ckpt")

    universal_convert(args.pt_path, args.ckpt_path, args.task, args.scale)