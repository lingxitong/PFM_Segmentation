#!/usr/bin/env python3
"""
Batch Bootstrap Inference Script

This script automatically discovers trained model folders and runs
bootstrap inference (infer_bootstrap.py) for each one.

Folder naming convention: {dataset_name}__{model_name}__seed{seed_number}
Example: BCSS__uni_v1__seed2025

Author: @chenwm
"""

import os
import sys
import json
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import threading
from collections import defaultdict

# ==========================================
# 配置区域
# ==========================================

# 默认路径配置
PROJECT_ROOT = "/Path/to/yours/PFM_Segmentation"
DATASET_JSON_DIR = f"{PROJECT_ROOT}/dataset_json"
SIZE_INFO_PATH = f"{PROJECT_ROOT}/dataset_json/dataset_size_info.json"
OUTPUT_BASE_DIR = "/Path/to/yours/PFM_Segmentation_Output/inference_bootstrap"
TASK_LOG_DIR = "/Path/to/yours/PFM_Segmentation_Output/task_logs_bootstrap"

# 虚拟环境配置
CONDA_ENV_NAME = "pfm_seg"

# 需要使用 resize_14 的模型列表
RESIZE_14_MODELS = {
    'virchow_v1', 'virchow_v2', 'uni_v2', 'midnight12k', 
    'kaiko-vitl14', 'hibou_l', 'hoptimus_0', 'hoptimus_1'
}

# musk 模型固定尺寸
MUSK_SIZE = 384

# GPU调度配置（默认值）
DEFAULT_MAX_PER_GPU = 3
DEFAULT_MAX_TOTAL = 9
DEFAULT_AVAILABLE_GPUS = [0, 1, 2]
DEFAULT_MIN_FREE_MEMORY = 20480  # MB
DEFAULT_WAIT_TIME_FULL = 5  # 秒
DEFAULT_WAIT_TIME_AFTER_START = 30  # 秒


@dataclass
class InferenceTask:
    """推理任务数据类"""
    folder_name: str
    folder_path: str
    parent_folder_name: str
    dataset_name: str
    model_name: str
    seed: int
    input_json: str
    checkpoint_dir: str
    config_path: str
    output_dir: str
    input_size: int
    extra: Optional[str] = None  # 额外标记: 'full', 'resize256' 等


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Batch Bootstrap Inference Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 推理单个目录下的所有模型
  python batch_infer_bootstrap.py --input_dir /path/to/models
  
  # 指定GPU和并发数
  python batch_infer_bootstrap.py --input_dir /path/to/models --gpus 0 1 2 --max_per_gpu 2
  
  # 只生成任务列表，不执行
  python batch_infer_bootstrap.py --input_dir /path/to/models --dry_run
        """
    )
    
    parser.add_argument('--input_dir', type=str, default="/Path/to/yours/PFM_Segmentation_Output/logs_frozen_01_11",
                        help='包含训练好模型文件夹的目录路径')
    parser.add_argument('--output_base_dir', type=str, default=OUTPUT_BASE_DIR,
                        help=f'推理结果输出的基础目录 (默认: {OUTPUT_BASE_DIR})')
    parser.add_argument('--dataset_json_dir', type=str, default=DATASET_JSON_DIR,
                        help=f'数据集JSON文件目录 (默认: {DATASET_JSON_DIR})')
    parser.add_argument('--size_info_path', type=str, default=SIZE_INFO_PATH,
                        help=f'数据集尺寸信息JSON路径 (默认: {SIZE_INFO_PATH})')
    
    # GPU调度参数
    parser.add_argument('--gpus', type=int, nargs='+', default=DEFAULT_AVAILABLE_GPUS,
                        help=f'可用GPU列表 (默认: {DEFAULT_AVAILABLE_GPUS})')
    parser.add_argument('--max_per_gpu', type=int, default=DEFAULT_MAX_PER_GPU,
                        help=f'每个GPU最多同时运行的进程数 (默认: {DEFAULT_MAX_PER_GPU})')
    parser.add_argument('--max_total', type=int, default=DEFAULT_MAX_TOTAL,
                        help=f'总共最多同时运行的进程数 (默认: {DEFAULT_MAX_TOTAL})')
    parser.add_argument('--min_free_memory', type=int, default=DEFAULT_MIN_FREE_MEMORY,
                        help=f'GPU最小剩余显存(MB) (默认: {DEFAULT_MIN_FREE_MEMORY})')
    parser.add_argument('--wait_time_full', type=int, default=DEFAULT_WAIT_TIME_FULL,
                        help=f'GPU满载时的等待时间(秒) (默认: {DEFAULT_WAIT_TIME_FULL})')
    parser.add_argument('--wait_time_after_start', type=int, default=DEFAULT_WAIT_TIME_AFTER_START,
                        help=f'进程启动后的等待时间(秒) (默认: {DEFAULT_WAIT_TIME_AFTER_START})')
    
    # 推理参数
    parser.add_argument('--batch_size', type=int, default=2,
                        help='推理batch size (默认: 8)')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Bootstrap迭代次数 (默认: 1000)')
    parser.add_argument('--resize_or_windowslide', type=str, 
                        choices=['resize', 'windowslide'], default='resize',
                        help='推理模式 (默认: resize)')
    parser.add_argument('--save_vis', default=True,
                        help='保存推理可视化结果 (masks和overlays)')
    parser.add_argument('--max_save_per_task', type=int, default=20,
                        help='每个任务最多保存的可视化样本数 (默认: 20)')
    
    # 其他参数
    parser.add_argument('--dry_run', action='store_true',
                        help='只打印任务列表，不执行')
    parser.add_argument('--skip_existing', default=True,
                        help='跳过已存在输出目录的任务')
    parser.add_argument('--filter_dataset', type=str, nargs='+', default=None,
                        help='只处理指定的数据集')
    parser.add_argument('--filter_model', type=str, nargs='+', default=None,
                        help='只处理指定的模型')
    parser.add_argument('--skip_datasets', type=str, nargs='+', default=None,
                        help='跳过指定的数据集')
    parser.add_argument('--skip_models', type=str, nargs='+', default=None,
                        help='跳过指定的模型')
    
    return parser.parse_args()


def load_size_info(size_info_path: str) -> Dict:
    """加载数据集尺寸信息"""
    with open(size_info_path, 'r') as f:
        return json.load(f)


def parse_folder_name(folder_name: str) -> Optional[Tuple[str, str, int, Optional[str]]]:
    """
    解析文件夹名称，提取数据集名、模型名、seed和额外标记
    
    支持的格式:
    - 3部分: {dataset_name}__{model_name}__seed{seed_number}
    - 4部分: {dataset_name}__{model_name}__{extra}__seed{seed_number}
      - extra 可以是 'full' (全量微调) 或 'resize{size}' (resize测试)
    
    Returns:
        (dataset_name, model_name, seed, extra) 或 None（如果解析失败）
        extra 为 None 表示普通frozen训练
    """
    parts = folder_name.split('__')
    
    if len(parts) == 3:
        # 格式: {dataset}__{model}__seed{seed}
        dataset_name = parts[0]
        model_name = parts[1]
        seed_part = parts[2]
        extra = None
    elif len(parts) == 4:
        # 格式: {dataset}__{model}__{extra}__seed{seed}
        dataset_name = parts[0]
        model_name = parts[1]
        extra = parts[2]  # 'full' 或 'resize256' 等
        seed_part = parts[3]
    else:
        return None
    
    # 解析seed
    if not seed_part.startswith('seed'):
        return None
    try:
        seed = int(seed_part[4:])  # 去掉 'seed' 前缀
    except ValueError:
        return None
    
    return dataset_name, model_name, seed, extra


def get_input_size(dataset_name: str, model_name: str, size_info: Dict, extra: Optional[str] = None) -> Optional[int]:
    """
    根据数据集和模型获取输入尺寸
    
    Args:
        dataset_name: 数据集名称
        model_name: 模型名称
        size_info: 数据集尺寸信息字典
        extra: 额外标记，如 'full', 'resize256' 等
    
    Returns:
        输入尺寸或None
    """
    # 如果 extra 包含 resize 信息，直接从中提取尺寸
    if extra and extra.startswith('resize'):
        try:
            return int(extra[6:])  # 去掉 'resize' 前缀，提取数字
        except ValueError:
            pass  # 解析失败，继续使用默认逻辑
    
    if dataset_name not in size_info:
        return None
    
    # musk模型使用固定尺寸
    if model_name == 'musk':
        return MUSK_SIZE
    
    # 根据模型选择resize_14或resize_16
    if model_name in RESIZE_14_MODELS:
        return size_info[dataset_name].get('resize_14')
    else:
        return size_info[dataset_name].get('resize_16')


def find_checkpoint_dir(folder_path: str) -> Optional[str]:
    """查找checkpoint目录"""
    checkpoint_dir = os.path.join(folder_path, 'checkpoints')
    if os.path.isdir(checkpoint_dir):
        return checkpoint_dir
    return None


def find_config_file(folder_path: str) -> Optional[str]:
    """查找config.yaml文件"""
    config_path = os.path.join(folder_path, 'config.yaml')
    if os.path.isfile(config_path):
        return config_path
    return None


def check_training_completed(folder_path: str) -> bool:
    """
    检查训练是否完成
    
    通过检查文件夹下是否存在 training_history.png 来判断
    
    Returns:
        True 表示训练已完成，False 表示训练未完成
    """
    training_history_path = os.path.join(folder_path, 'training_history.png')
    return os.path.isfile(training_history_path)


def discover_tasks(
    input_dir: str,
    dataset_json_dir: str,
    size_info: Dict,
    output_base_dir: str,
    filter_dataset: Optional[List[str]] = None,
    filter_model: Optional[List[str]] = None,
    skip_datasets: Optional[List[str]] = None,
    skip_models: Optional[List[str]] = None,
    skip_existing: bool = False
) -> List[InferenceTask]:
    """
    发现并解析所有推理任务
    
    Args:
        input_dir: 输入目录路径
        dataset_json_dir: 数据集JSON目录
        size_info: 数据集尺寸信息
        output_base_dir: 输出基础目录
        filter_dataset: 只处理指定数据集
        filter_model: 只处理指定模型
        skip_datasets: 跳过指定数据集
        skip_models: 跳过指定模型
        skip_existing: 跳过已存在输出的任务
    
    Returns:
        推理任务列表
    """
    tasks = []
    input_path = Path(input_dir)
    parent_folder_name = input_path.name  # 父目录名称
    
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return tasks
    
    # 遍历所有子文件夹
    for folder in sorted(input_path.iterdir()):
        if not folder.is_dir():
            continue
        
        folder_name = folder.name
        
        # 解析文件夹名称
        parsed = parse_folder_name(folder_name)
        if parsed is None:
            print(f"  跳过: {folder_name} (无法解析文件夹名称格式)")
            continue
        
        dataset_name, model_name, seed, extra = parsed
        
        # 应用过滤器
        if filter_dataset and dataset_name not in filter_dataset:
            print(f"  跳过: {folder_name} (不在指定数据集列表中)")
            continue
        
        if filter_model and model_name not in filter_model:
            print(f"  跳过: {folder_name} (不在指定模型列表中)")
            continue
        
        if skip_datasets and dataset_name in skip_datasets:
            print(f"  跳过: {folder_name} (在排除数据集列表中)")
            continue
        
        if skip_models and model_name in skip_models:
            print(f"  跳过: {folder_name} (在排除模型列表中)")
            continue
        
        # 查找input_json
        input_json = os.path.join(dataset_json_dir, f"{dataset_name}.json")
        if not os.path.isfile(input_json):
            print(f"  跳过: {folder_name} (未找到数据集JSON: {input_json})")
            continue
        
        # 检查训练是否完成（通过 training_history.png 判断）
        if not check_training_completed(str(folder)):
            print(f"  跳过: {folder_name} (训练未完成，未找到 training_history.png)")
            continue
        
        # 查找checkpoint目录
        checkpoint_dir = find_checkpoint_dir(str(folder))
        if checkpoint_dir is None:
            print(f"  跳过: {folder_name} (未找到checkpoints目录)")
            continue
        
        # 查找config.yaml
        config_path = find_config_file(str(folder))
        if config_path is None:
            print(f"  跳过: {folder_name} (未找到config.yaml)")
            continue
        
        # 获取input_size
        input_size = get_input_size(dataset_name, model_name, size_info, extra)
        if input_size is None:
            print(f"  跳过: {folder_name} (未找到数据集尺寸信息)")
            continue
        
        # 构建output_dir: output_base_dir / parent_folder_name / folder_name
        output_dir = os.path.join(output_base_dir, parent_folder_name, folder_name)
        
        # 检查是否跳过已存在的
        if skip_existing and os.path.exists(output_dir):
            metrics_file = os.path.join(output_dir, 'bootstrap_metrics.json')
            if os.path.isfile(metrics_file):
                print(f"  跳过: {folder_name} (输出已存在)")
                continue
        
        # 创建任务
        task = InferenceTask(
            folder_name=folder_name,
            folder_path=str(folder),
            parent_folder_name=parent_folder_name,
            dataset_name=dataset_name,
            model_name=model_name,
            seed=seed,
            input_json=input_json,
            checkpoint_dir=checkpoint_dir,
            config_path=config_path,
            output_dir=output_dir,
            input_size=input_size,
            extra=extra
        )
        tasks.append(task)
        extra_info = f" [{extra}]" if extra else ""
        print(f"  发现: {folder_name}{extra_info}")
    
    return tasks


def build_command(task: InferenceTask, args: argparse.Namespace, gpu_id: int) -> str:
    """构建推理命令"""
    # 构建 conda 激活命令
    conda_activate = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {CONDA_ENV_NAME} &&"
    
    cmd_parts = [
        conda_activate,
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "python",
        os.path.join(PROJECT_ROOT, "scripts", "infer_bootstrap.py"),
        f"--config '{task.config_path}'",
        f"--checkpoint '{task.checkpoint_dir}'",
        f"--input_json '{task.input_json}'",
        f"--output_dir '{task.output_dir}'",
        f"--device cuda:0",  # 因为设置了CUDA_VISIBLE_DEVICES，所以用cuda:0
        f"--input_size {task.input_size}",
        f"--seed {task.seed}",
        f"--batch_size {args.batch_size}",
        f"--n_bootstrap {args.n_bootstrap}",
        f"--resize_or_windowslide {args.resize_or_windowslide}"
    ]
    
    # 添加可视化保存参数
    if args.save_vis:
        cmd_parts.append("--save_vis")
        cmd_parts.append(f"--max_save_per_task {args.max_save_per_task}")
    
    return " ".join(cmd_parts)


def get_gpu_free_memory(gpu_id: int) -> int:
    """获取指定GPU的剩余显存（MB）"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', str(gpu_id)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return 0


class GPUScheduler:
    """GPU调度器"""
    
    def __init__(self, gpus: List[int], max_per_gpu: int, max_total: int, 
                 min_free_memory: int):
        self.gpus = gpus
        self.max_per_gpu = max_per_gpu
        self.max_total = max_total
        self.min_free_memory = min_free_memory
        self.gpu_processes: Dict[int, List[subprocess.Popen]] = {gpu: [] for gpu in gpus}
        self.lock = threading.Lock()
    
    def _cleanup_finished(self):
        """清理已完成的进程"""
        for gpu in self.gpus:
            self.gpu_processes[gpu] = [
                p for p in self.gpu_processes[gpu] 
                if p.poll() is None
            ]
    
    def get_total_running(self) -> int:
        """获取总运行进程数"""
        with self.lock:
            self._cleanup_finished()
            return sum(len(procs) for procs in self.gpu_processes.values())
    
    def get_gpu_load(self, gpu: int) -> int:
        """获取指定GPU的负载（运行进程数）"""
        with self.lock:
            self._cleanup_finished()
            return len(self.gpu_processes[gpu])
    
    def select_gpu(self) -> Optional[int]:
        """选择最合适的GPU"""
        with self.lock:
            self._cleanup_finished()
            
            # 检查总进程数
            total = sum(len(procs) for procs in self.gpu_processes.values())
            if total >= self.max_total:
                return None
            
            # 寻找负载最小且满足条件的GPU
            best_gpu = None
            min_load = float('inf')
            
            for gpu in self.gpus:
                load = len(self.gpu_processes[gpu])
                if load >= self.max_per_gpu:
                    continue
                
                # 检查显存
                free_mem = get_gpu_free_memory(gpu)
                if free_mem < self.min_free_memory:
                    continue
                
                if load < min_load:
                    min_load = load
                    best_gpu = gpu
            
            return best_gpu
    
    def register_process(self, gpu: int, process: subprocess.Popen):
        """注册新进程"""
        with self.lock:
            self.gpu_processes[gpu].append(process)
    
    def wait_all(self):
        """等待所有进程完成"""
        all_procs = []
        for procs in self.gpu_processes.values():
            all_procs.extend(procs)
        for p in all_procs:
            p.wait()


def run_tasks(tasks: List[InferenceTask], args: argparse.Namespace) -> Dict[str, List[str]]:
    """
    运行所有推理任务
    
    Returns:
        包含成功和失败任务的字典
    """
    # 创建日志目录
    os.makedirs(TASK_LOG_DIR, exist_ok=True)
    
    # 初始化调度器
    scheduler = GPUScheduler(
        gpus=args.gpus,
        max_per_gpu=args.max_per_gpu,
        max_total=args.max_total,
        min_free_memory=args.min_free_memory
    )
    
    results = {
        'success': [],
        'failed': [],
        'log_files': []
    }
    
    print("\n" + "=" * 60)
    print(f"开始调度任务，共 {len(tasks)} 个任务...")
    print(f"可用GPU: {args.gpus}")
    print(f"限制: 每个GPU最多 {args.max_per_gpu} 个进程，总共最多 {args.max_total} 个进程")
    print(f"显存要求: 每个GPU剩余显存必须大于 {args.min_free_memory // 1024}G ({args.min_free_memory} MB)")
    print("=" * 60 + "\n")
    
    task_processes = []  # (task, process, log_file)
    
    for i, task in enumerate(tasks):
        while True:
            gpu = scheduler.select_gpu()
            if gpu is not None:
                # 构建命令
                cmd = build_command(task, args, gpu)
                free_mem = get_gpu_free_memory(gpu)
                load = scheduler.get_gpu_load(gpu)
                
                print(f"[任务 {i+1}/{len(tasks)}] 分配到 GPU {gpu} "
                      f"(负载: {load}/{args.max_per_gpu}, 剩余显存: {free_mem // 1024}G)")
                print(f"  任务: {task.folder_name}")
                print(f"  命令: {cmd[:100]}...")
                
                # 创建日志文件
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(
                    TASK_LOG_DIR, 
                    f"bootstrap_{i}_gpu{gpu}_{task.folder_name}_{timestamp}.log"
                )
                
                # 创建输出目录
                os.makedirs(task.output_dir, exist_ok=True)
                
                # 启动进程 (使用 bash 以支持 conda activate)
                with open(log_file, 'w') as f:
                    process = subprocess.Popen(
                        cmd,
                        shell=True,
                        executable='/bin/bash',
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL
                    )
                
                scheduler.register_process(gpu, process)
                task_processes.append((task, process, log_file))
                results['log_files'].append(log_file)
                
                print(f"  -> 日志文件: {log_file} (PID: {process.pid})")
                
                # 等待一段时间让进程占用显存
                time.sleep(args.wait_time_after_start)
                break
            else:
                # 所有GPU都满了，等待
                time.sleep(args.wait_time_full)
    
    # 等待所有进程完成
    print("\n所有任务已启动，等待后台进程完成...")
    print(f"日志文件保存在: {TASK_LOG_DIR}")
    
    scheduler.wait_all()
    
    # 检查结果
    print("\n" + "=" * 60)
    print("检查任务结果...")
    print("=" * 60)
    
    for task, process, log_file in task_processes:
        if process.returncode == 0:
            # 还需要检查输出文件是否存在
            metrics_file = os.path.join(task.output_dir, 'bootstrap_metrics.json')
            if os.path.isfile(metrics_file):
                results['success'].append(task.folder_name)
                print(f"✓ 成功: {task.folder_name}")
            else:
                results['failed'].append(task.folder_name)
                print(f"✗ 失败: {task.folder_name} (未生成输出文件)")
        else:
            results['failed'].append(task.folder_name)
            print(f"✗ 失败: {task.folder_name} (返回码: {process.returncode})")
    
    return results


def print_summary(tasks: List[InferenceTask], results: Optional[Dict] = None):
    """打印任务摘要"""
    print("\n" + "=" * 60)
    print("任务摘要")
    print("=" * 60)
    
    # 按数据集、模型和实验类型统计
    by_dataset = defaultdict(list)
    by_model = defaultdict(list)
    by_extra = defaultdict(list)
    
    for task in tasks:
        by_dataset[task.dataset_name].append(task)
        by_model[task.model_name].append(task)
        extra_type = task.extra if task.extra else "frozen"
        by_extra[extra_type].append(task)
    
    print(f"\n总任务数: {len(tasks)}")
    
    print(f"\n按数据集统计 ({len(by_dataset)} 个数据集):")
    for ds, ds_tasks in sorted(by_dataset.items()):
        print(f"  {ds}: {len(ds_tasks)} 个任务")
    
    print(f"\n按模型统计 ({len(by_model)} 个模型):")
    for model, model_tasks in sorted(by_model.items()):
        print(f"  {model}: {len(model_tasks)} 个任务")
    
    print(f"\n按实验类型统计 ({len(by_extra)} 种类型):")
    for extra_type, extra_tasks in sorted(by_extra.items()):
        print(f"  {extra_type}: {len(extra_tasks)} 个任务")
    
    if results:
        print(f"\n执行结果:")
        print(f"  成功: {len(results['success'])} 个")
        print(f"  失败: {len(results['failed'])} 个")
        
        if results['failed']:
            print(f"\n失败的任务:")
            for name in results['failed']:
                print(f"  - {name}")


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("Batch Bootstrap Inference Script")
    print("=" * 60)
    print(f"\n输入目录: {args.input_dir}")
    print(f"输出基础目录: {args.output_base_dir}")
    print(f"数据集JSON目录: {args.dataset_json_dir}")
    print(f"尺寸信息文件: {args.size_info_path}")
    
    # 加载尺寸信息
    print("\n加载数据集尺寸信息...")
    size_info = load_size_info(args.size_info_path)
    print(f"  已加载 {len(size_info)} 个数据集的尺寸信息")
    
    # 发现任务
    print(f"\n扫描目录: {args.input_dir}")
    tasks = discover_tasks(
        input_dir=args.input_dir,
        dataset_json_dir=args.dataset_json_dir,
        size_info=size_info,
        output_base_dir=args.output_base_dir,
        filter_dataset=args.filter_dataset,
        filter_model=args.filter_model,
        skip_datasets=args.skip_datasets,
        skip_models=args.skip_models,
        skip_existing=args.skip_existing
    )
    
    if not tasks:
        print("\n未发现任何有效任务!")
        return
    
    # 打印任务摘要
    print_summary(tasks)
    
    # Dry run模式
    if args.dry_run:
        print("\n[DRY RUN] 以下是将要执行的命令:")
        print("-" * 60)
        for i, task in enumerate(tasks):
            cmd = build_command(task, args, gpu_id=0)
            print(f"\n[{i+1}] {task.folder_name}")
            print(f"    {cmd}")
        print("\n[DRY RUN] 实际执行时会根据GPU负载自动分配")
        return
    
    # 执行任务
    results = run_tasks(tasks, args)
    
    # 打印最终摘要
    print_summary(tasks, results)
    
    print("\n" + "=" * 60)
    print("所有任务执行完毕！")
    print("=" * 60)


if __name__ == '__main__':
    main()
