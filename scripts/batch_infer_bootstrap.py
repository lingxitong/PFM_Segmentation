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
# Configuration
# ==========================================

# Default path configuration
PROJECT_ROOT = "/Path/to/yours/PFM_Segmentation"
DATASET_JSON_DIR = f"{PROJECT_ROOT}/dataset_json"
SIZE_INFO_PATH = f"{PROJECT_ROOT}/dataset_json/dataset_size_info.json"
OUTPUT_BASE_DIR = "/Path/to/yours/PFM_Segmentation_Output/inference_bootstrap"
TASK_LOG_DIR = "/Path/to/yours/PFM_Segmentation_Output/task_logs_bootstrap"

# Virtual environment configuration
CONDA_ENV_NAME = "pfm_seg"

# Models that require resize_14
RESIZE_14_MODELS = {
    'virchow_v1', 'virchow_v2', 'uni_v2', 'midnight12k',
    'kaiko-vitl14', 'hibou_l', 'hoptimus_0', 'hoptimus_1', 'h0_mini',
}

# Fixed size for musk model
MUSK_SIZE = 384

# GPU scheduling configuration (defaults)
DEFAULT_MAX_PER_GPU = 3
DEFAULT_MAX_TOTAL = 9
DEFAULT_AVAILABLE_GPUS = [0, 1, 2]
DEFAULT_MIN_FREE_MEMORY = 20480  # MB
DEFAULT_WAIT_TIME_FULL = 5  # seconds
DEFAULT_WAIT_TIME_AFTER_START = 30  # seconds


@dataclass
class InferenceTask:
    """Data class for an inference task"""
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
    extra: Optional[str] = None  # Extra tag: 'full', 'resize256', etc.


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Batch Bootstrap Inference Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Run inference for all models under a single directory
  python batch_infer_bootstrap.py --input_dir /path/to/models
  
  # Specify GPUs and concurrency
  python batch_infer_bootstrap.py --input_dir /path/to/models --gpus 0 1 2 --max_per_gpu 2
  
  # Only generate the task list without executing
  python batch_infer_bootstrap.py --input_dir /path/to/models --dry_run
        """
    )
    
    parser.add_argument('--input_dir', type=str, default="/Path/to/yours/PFM_Segmentation_Output/logs_frozen_01_11",
                        help='Directory containing trained model folders')
    parser.add_argument('--output_base_dir', type=str, default=OUTPUT_BASE_DIR,
                        help=f'Base directory for inference outputs (default: {OUTPUT_BASE_DIR})')
    parser.add_argument('--dataset_json_dir', type=str, default=DATASET_JSON_DIR,
                        help=f'Directory of dataset JSON files (default: {DATASET_JSON_DIR})')
    parser.add_argument('--size_info_path', type=str, default=SIZE_INFO_PATH,
                        help=f'Path to dataset size info JSON (default: {SIZE_INFO_PATH})')
    
    # GPU scheduling arguments
    parser.add_argument('--gpus', type=int, nargs='+', default=DEFAULT_AVAILABLE_GPUS,
                        help=f'List of available GPUs (default: {DEFAULT_AVAILABLE_GPUS})')
    parser.add_argument('--max_per_gpu', type=int, default=DEFAULT_MAX_PER_GPU,
                        help=f'Max concurrent processes per GPU (default: {DEFAULT_MAX_PER_GPU})')
    parser.add_argument('--max_total', type=int, default=DEFAULT_MAX_TOTAL,
                        help=f'Max concurrent processes in total (default: {DEFAULT_MAX_TOTAL})')
    parser.add_argument('--min_free_memory', type=int, default=DEFAULT_MIN_FREE_MEMORY,
                        help=f'Minimum free GPU memory in MB (default: {DEFAULT_MIN_FREE_MEMORY})')
    parser.add_argument('--wait_time_full', type=int, default=DEFAULT_WAIT_TIME_FULL,
                        help=f'Wait time in seconds when GPUs are full (default: {DEFAULT_WAIT_TIME_FULL})')
    parser.add_argument('--wait_time_after_start', type=int, default=DEFAULT_WAIT_TIME_AFTER_START,
                        help=f'Wait time in seconds after starting a process (default: {DEFAULT_WAIT_TIME_AFTER_START})')
    
    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Inference batch size (default: 8)')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap iterations (default: 1000)')
    parser.add_argument('--resize_or_windowslide', type=str, 
                        choices=['resize', 'windowslide'], default='resize',
                        help='Inference mode (default: resize)')
    parser.add_argument('--save_vis', default=True,
                        help='Save inference visualization results (masks and overlays)')
    parser.add_argument('--max_save_per_task', type=int, default=20,
                        help='Max number of visualization samples to save per task (default: 20)')
    
    # Other arguments
    parser.add_argument('--dry_run', action='store_true',
                        help='Only print the task list without executing')
    parser.add_argument('--skip_existing', default=True,
                        help='Skip tasks whose output directories already exist')
    parser.add_argument('--filter_dataset', type=str, nargs='+', default=None,
                        help='Only process the specified datasets')
    parser.add_argument('--filter_model', type=str, nargs='+', default=None,
                        help='Only process the specified models')
    parser.add_argument('--skip_datasets', type=str, nargs='+', default=None,
                        help='Skip the specified datasets')
    parser.add_argument('--skip_models', type=str, nargs='+', default=None,
                        help='Skip the specified models')
    
    return parser.parse_args()


def load_size_info(size_info_path: str) -> Dict:
    """Load dataset size information"""
    with open(size_info_path, 'r') as f:
        return json.load(f)


def parse_folder_name(folder_name: str) -> Optional[Tuple[str, str, int, Optional[str]]]:
    """
    Parse a folder name to extract dataset name, model name, seed, and optional extra tag
    
    Supported formats:
    - 3 parts: {dataset_name}__{model_name}__seed{seed_number}
    - 4 parts: {dataset_name}__{model_name}__{extra}__seed{seed_number}
      - extra can be 'full' (full fine-tuning) or 'resize{size}' (resize test)
    
    Returns:
        (dataset_name, model_name, seed, extra) or None if parsing fails
        extra is None for normal frozen training
    """
    parts = folder_name.split('__')
    
    if len(parts) == 3:
        # Format: {dataset}__{model}__seed{seed}
        dataset_name = parts[0]
        model_name = parts[1]
        seed_part = parts[2]
        extra = None
    elif len(parts) == 4:
        # Format: {dataset}__{model}__{extra}__seed{seed}
        dataset_name = parts[0]
        model_name = parts[1]
        extra = parts[2]  # 'full' or 'resize256', etc.
        seed_part = parts[3]
    else:
        return None
    
    # Parse seed
    if not seed_part.startswith('seed'):
        return None
    try:
        seed = int(seed_part[4:])  # Strip the 'seed' prefix
    except ValueError:
        return None
    
    return dataset_name, model_name, seed, extra


def get_input_size(dataset_name: str, model_name: str, size_info: Dict, extra: Optional[str] = None) -> Optional[int]:
    """
    Get the input size based on dataset and model
    
    Args:
        dataset_name: Dataset name
        model_name: Model name
        size_info: Dataset size info dictionary
        extra: Extra tag such as 'full', 'resize256', etc.
    
    Returns:
        Input size or None
    """
    # If extra contains resize info, extract the size from it directly
    if extra and extra.startswith('resize'):
        try:
            return int(extra[6:])  # Strip the 'resize' prefix and extract the number
        except ValueError:
            pass  # Parse failed; fall back to default logic
    
    if dataset_name not in size_info:
        return None
    
    # musk model uses a fixed size
    if model_name == 'musk':
        return MUSK_SIZE
    
    # Choose resize_14 or resize_16 based on the model
    if model_name in RESIZE_14_MODELS:
        return size_info[dataset_name].get('resize_14')
    else:
        return size_info[dataset_name].get('resize_16')


def find_checkpoint_dir(folder_path: str) -> Optional[str]:
    """Find the checkpoint directory"""
    checkpoint_dir = os.path.join(folder_path, 'checkpoints')
    if os.path.isdir(checkpoint_dir):
        return checkpoint_dir
    return None


def find_config_file(folder_path: str) -> Optional[str]:
    """Find the config.yaml file"""
    config_path = os.path.join(folder_path, 'config.yaml')
    if os.path.isfile(config_path):
        return config_path
    return None


def check_training_completed(folder_path: str) -> bool:
    """
    Check whether training has completed
    
    Determined by whether training_history.png exists under the folder
    
    Returns:
        True if training is completed, False otherwise
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
    Discover and parse all inference tasks
    
    Args:
        input_dir: Input directory path
        dataset_json_dir: Dataset JSON directory
        size_info: Dataset size information
        output_base_dir: Base output directory
        filter_dataset: Only process the specified datasets
        filter_model: Only process the specified models
        skip_datasets: Skip the specified datasets
        skip_models: Skip the specified models
        skip_existing: Skip tasks whose outputs already exist
    
    Returns:
        List of inference tasks
    """
    tasks = []
    input_path = Path(input_dir)
    parent_folder_name = input_path.name  # Parent directory name
    
    if not input_path.exists():
        print(f"Error: input directory does not exist: {input_dir}")
        return tasks
    
    # Iterate over all subfolders
    for folder in sorted(input_path.iterdir()):
        if not folder.is_dir():
            continue
        
        folder_name = folder.name
        
        # Parse folder name
        parsed = parse_folder_name(folder_name)
        if parsed is None:
            print(f"  Skip: {folder_name} (unable to parse folder name format)")
            continue
        
        dataset_name, model_name, seed, extra = parsed
        
        # Apply filters
        if filter_dataset and dataset_name not in filter_dataset:
            print(f"  Skip: {folder_name} (not in the specified dataset list)")
            continue
        
        if filter_model and model_name not in filter_model:
            print(f"  Skip: {folder_name} (not in the specified model list)")
            continue
        
        if skip_datasets and dataset_name in skip_datasets:
            print(f"  Skip: {folder_name} (in the excluded dataset list)")
            continue
        
        if skip_models and model_name in skip_models:
            print(f"  Skip: {folder_name} (in the excluded model list)")
            continue
        
        # Find input_json
        input_json = os.path.join(dataset_json_dir, f"{dataset_name}.json")
        if not os.path.isfile(input_json):
            print(f"  Skip: {folder_name} (dataset JSON not found: {input_json})")
            continue
        
        # Check whether training has completed (via training_history.png)
        if not check_training_completed(str(folder)):
            print(f"  Skip: {folder_name} (training not completed, training_history.png not found)")
            continue
        
        # Find checkpoint directory
        checkpoint_dir = find_checkpoint_dir(str(folder))
        if checkpoint_dir is None:
            print(f"  Skip: {folder_name} (checkpoints directory not found)")
            continue
        
        # Find config.yaml
        config_path = find_config_file(str(folder))
        if config_path is None:
            print(f"  Skip: {folder_name} (config.yaml not found)")
            continue
        
        # Get input_size
        input_size = get_input_size(dataset_name, model_name, size_info, extra)
        if input_size is None:
            print(f"  Skip: {folder_name} (dataset size info not found)")
            continue
        
        # Build output_dir: output_base_dir / parent_folder_name / folder_name
        output_dir = os.path.join(output_base_dir, parent_folder_name, folder_name)
        
        # Check whether to skip existing outputs
        if skip_existing and os.path.exists(output_dir):
            metrics_file = os.path.join(output_dir, 'bootstrap_metrics.json')
            if os.path.isfile(metrics_file):
                print(f"  Skip: {folder_name} (output already exists)")
                continue
        
        # Create task
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
        print(f"  Found: {folder_name}{extra_info}")
    
    return tasks


def build_command(task: InferenceTask, args: argparse.Namespace, gpu_id: int) -> str:
    """Build the inference command"""
    # Build conda activate command
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
        f"--device cuda:0",  # Use cuda:0 because CUDA_VISIBLE_DEVICES is set
        f"--input_size {task.input_size}",
        f"--seed {task.seed}",
        f"--batch_size {args.batch_size}",
        f"--n_bootstrap {args.n_bootstrap}",
        f"--resize_or_windowslide {args.resize_or_windowslide}"
    ]
    
    # Add visualization save arguments
    if args.save_vis:
        cmd_parts.append("--save_vis")
        cmd_parts.append(f"--max_save_per_task {args.max_save_per_task}")
    
    return " ".join(cmd_parts)


def get_gpu_free_memory(gpu_id: int) -> int:
    """Get free memory of the specified GPU (MB)"""
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
    """GPU scheduler"""
    
    def __init__(self, gpus: List[int], max_per_gpu: int, max_total: int, 
                 min_free_memory: int):
        self.gpus = gpus
        self.max_per_gpu = max_per_gpu
        self.max_total = max_total
        self.min_free_memory = min_free_memory
        self.gpu_processes: Dict[int, List[subprocess.Popen]] = {gpu: [] for gpu in gpus}
        self.lock = threading.Lock()
    
    def _cleanup_finished(self):
        """Clean up finished processes"""
        for gpu in self.gpus:
            self.gpu_processes[gpu] = [
                p for p in self.gpu_processes[gpu] 
                if p.poll() is None
            ]
    
    def get_total_running(self) -> int:
        """Get the total number of running processes"""
        with self.lock:
            self._cleanup_finished()
            return sum(len(procs) for procs in self.gpu_processes.values())
    
    def get_gpu_load(self, gpu: int) -> int:
        """Get the load of the specified GPU (number of running processes)"""
        with self.lock:
            self._cleanup_finished()
            return len(self.gpu_processes[gpu])
    
    def select_gpu(self) -> Optional[int]:
        """Select the most suitable GPU"""
        with self.lock:
            self._cleanup_finished()
            
            # Check total process count
            total = sum(len(procs) for procs in self.gpu_processes.values())
            if total >= self.max_total:
                return None
            
            # Find the GPU with the lowest load that satisfies constraints
            best_gpu = None
            min_load = float('inf')
            
            for gpu in self.gpus:
                load = len(self.gpu_processes[gpu])
                if load >= self.max_per_gpu:
                    continue
                
                # Check free memory
                free_mem = get_gpu_free_memory(gpu)
                if free_mem < self.min_free_memory:
                    continue
                
                if load < min_load:
                    min_load = load
                    best_gpu = gpu
            
            return best_gpu
    
    def register_process(self, gpu: int, process: subprocess.Popen):
        """Register a new process"""
        with self.lock:
            self.gpu_processes[gpu].append(process)
    
    def wait_all(self):
        """Wait for all processes to finish"""
        all_procs = []
        for procs in self.gpu_processes.values():
            all_procs.extend(procs)
        for p in all_procs:
            p.wait()


def run_tasks(tasks: List[InferenceTask], args: argparse.Namespace) -> Dict[str, List[str]]:
    """
    Run all inference tasks
    
    Returns:
        Dictionary containing successful and failed tasks
    """
    # Create log directory
    os.makedirs(TASK_LOG_DIR, exist_ok=True)
    
    # Initialize scheduler
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
    print(f"Starting task scheduling, {len(tasks)} tasks in total...")
    print(f"Available GPUs: {args.gpus}")
    print(f"Limits: at most {args.max_per_gpu} processes per GPU, at most {args.max_total} processes in total")
    print(f"Memory requirement: each GPU must have more than {args.min_free_memory // 1024}G free ({args.min_free_memory} MB)")
    print("=" * 60 + "\n")
    
    task_processes = []  # (task, process, log_file)
    
    for i, task in enumerate(tasks):
        while True:
            gpu = scheduler.select_gpu()
            if gpu is not None:
                # Build command
                cmd = build_command(task, args, gpu)
                free_mem = get_gpu_free_memory(gpu)
                load = scheduler.get_gpu_load(gpu)
                
                print(f"[Task {i+1}/{len(tasks)}] Assigned to GPU {gpu} "
                      f"(load: {load}/{args.max_per_gpu}, free memory: {free_mem // 1024}G)")
                print(f"  Task: {task.folder_name}")
                print(f"  Command: {cmd[:100]}...")
                
                # Create log file
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(
                    TASK_LOG_DIR, 
                    f"bootstrap_{i}_gpu{gpu}_{task.folder_name}_{timestamp}.log"
                )
                
                # Create output directory
                os.makedirs(task.output_dir, exist_ok=True)
                
                # Start process (use bash to support conda activate)
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
                
                print(f"  -> Log file: {log_file} (PID: {process.pid})")
                
                # Wait a while for the process to occupy GPU memory
                time.sleep(args.wait_time_after_start)
                break
            else:
                # All GPUs are full; wait
                time.sleep(args.wait_time_full)
    
    # Wait for all processes to finish
    print("\nAll tasks have been started; waiting for background processes to finish...")
    print(f"Log files are saved in: {TASK_LOG_DIR}")
    
    scheduler.wait_all()
    
    # Check results
    print("\n" + "=" * 60)
    print("Checking task results...")
    print("=" * 60)
    
    for task, process, log_file in task_processes:
        if process.returncode == 0:
            # Also check whether the output file exists
            metrics_file = os.path.join(task.output_dir, 'bootstrap_metrics.json')
            if os.path.isfile(metrics_file):
                results['success'].append(task.folder_name)
                print(f"✓ Success: {task.folder_name}")
            else:
                results['failed'].append(task.folder_name)
                print(f"✗ Failed: {task.folder_name} (output file not generated)")
        else:
            results['failed'].append(task.folder_name)
            print(f"✗ Failed: {task.folder_name} (return code: {process.returncode})")
    
    return results


def print_summary(tasks: List[InferenceTask], results: Optional[Dict] = None):
    """Print task summary"""
    print("\n" + "=" * 60)
    print("Task Summary")
    print("=" * 60)
    
    # Statistics by dataset, model, and experiment type
    by_dataset = defaultdict(list)
    by_model = defaultdict(list)
    by_extra = defaultdict(list)
    
    for task in tasks:
        by_dataset[task.dataset_name].append(task)
        by_model[task.model_name].append(task)
        extra_type = task.extra if task.extra else "frozen"
        by_extra[extra_type].append(task)
    
    print(f"\nTotal tasks: {len(tasks)}")
    
    print(f"\nBy dataset ({len(by_dataset)} datasets):")
    for ds, ds_tasks in sorted(by_dataset.items()):
        print(f"  {ds}: {len(ds_tasks)} tasks")
    
    print(f"\nBy model ({len(by_model)} models):")
    for model, model_tasks in sorted(by_model.items()):
        print(f"  {model}: {len(model_tasks)} tasks")
    
    print(f"\nBy experiment type ({len(by_extra)} types):")
    for extra_type, extra_tasks in sorted(by_extra.items()):
        print(f"  {extra_type}: {len(extra_tasks)} tasks")
    
    if results:
        print(f"\nExecution results:")
        print(f"  Success: {len(results['success'])}")
        print(f"  Failed: {len(results['failed'])}")
        
        if results['failed']:
            print(f"\nFailed tasks:")
            for name in results['failed']:
                print(f"  - {name}")


def main():
    """Main function"""
    args = parse_args()
    
    print("=" * 60)
    print("Batch Bootstrap Inference Script")
    print("=" * 60)
    print(f"\nInput directory: {args.input_dir}")
    print(f"Output base directory: {args.output_base_dir}")
    print(f"Dataset JSON directory: {args.dataset_json_dir}")
    print(f"Size info file: {args.size_info_path}")
    
    # Load size info
    print("\nLoading dataset size information...")
    size_info = load_size_info(args.size_info_path)
    print(f"  Loaded size info for {len(size_info)} datasets")
    
    # Discover tasks
    print(f"\nScanning directory: {args.input_dir}")
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
        print("\nNo valid tasks found!")
        return
    
    # Print task summary
    print_summary(tasks)
    
    # Dry run mode
    if args.dry_run:
        print("\n[DRY RUN] Commands that would be executed:")
        print("-" * 60)
        for i, task in enumerate(tasks):
            cmd = build_command(task, args, gpu_id=0)
            print(f"\n[{i+1}] {task.folder_name}")
            print(f"    {cmd}")
        print("\n[DRY RUN] In a real run, tasks will be assigned automatically based on GPU load")
        return
    
    # Execute tasks
    results = run_tasks(tasks, args)
    
    # Print final summary
    print_summary(tasks, results)
    
    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
