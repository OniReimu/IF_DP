"""Device helpers for single- and multi-GPU execution."""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from config import get_logger

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

logger = get_logger("device")

# LUMI-G 拓扑映射
LUMI_GPU_CPU_MAP = {
    0: [49, 50, 51, 52, 53, 54, 55],
    1: [57, 58, 59, 60, 61, 62, 63],
    2: [17, 18, 19, 20, 21, 22, 23],
    3: [25, 26, 27, 28, 29, 30, 31],
    4: [1, 2, 3, 4, 5, 6, 7],
    5: [9, 10, 11, 12, 13, 14, 15],
    6: [33, 34, 35, 36, 37, 38, 39],
    7: [41, 42, 43, 44, 45, 46, 47],
}

def _set_cpu_affinity(local_rank: int, global_rank: int) -> None:
    if psutil is None: return
    cpu_list = LUMI_GPU_CPU_MAP.get(local_rank)
    if not cpu_list: return
    try:
        psutil.Process().cpu_affinity(cpu_list)
        logger.info("Rank %s (local %s) binding to CPUs %s", global_rank, local_rank, cpu_list)
    except Exception as exc:
        logger.warn("Failed to set CPU affinity: %s", exc)

def _init_distributed(args) -> bool:
    """根据 multi_gpu 参数初始化分布式环境"""
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is None:
        logger.warn("--distributed requested but LOCAL_RANK not found. Running in single-GPU mode.")
        setattr(args, "distributed", False)
        return False

    # 初始化进程组
    if not dist.is_initialized():
        backend = os.environ.get("IFDP_DIST_BACKEND", "nccl")
        dist.init_process_group(backend=backend)

    local_rank = int(local_rank_env)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", local_rank))

    torch.cuda.set_device(local_rank)
    _set_cpu_affinity(local_rank, global_rank)

    # 将分布式状态挂载到 args
    setattr(args, "distributed", True)
    setattr(args, "local_rank", local_rank)
    setattr(args, "global_rank", global_rank)
    setattr(args, "world_size", world_size)
    
    logger.info("Distributed mode enabled (rank %s/%s, local GPU %s)", global_rank, world_size, local_rank)
    return True

def resolve_device(args) -> torch.device:
    """确定主设备"""
    # 尝试初始化分布式
    if _init_distributed(args):
        return torch.device(f"cuda:{getattr(args, 'local_rank')}")

    # CPU / MPS 逻辑 (非分布式)
    if getattr(args, "cpu", False):
        return torch.device("cpu")

    if getattr(args, "mps", False) and torch.backends.mps.is_available():
        return torch.device("mps")

    # 单卡 CUDA 逻辑
    if torch.cuda.is_available():
        # 如果用户手动指定了某个 ID，否则默认为 0
        cuda_id = getattr(args, "cuda_devices", 0)
        return torch.device(f"cuda:{cuda_id}")

    return torch.device("cpu")

def maybe_wrap_model_for_multi_gpu(model: torch.nn.Module, args) -> torch.nn.Module:
    """如果开启了 multi_gpu，则使用 DDP 包装模型"""
    if getattr(args, "distributed", False):
        local_rank = getattr(args, "local_rank", 0)
        device = torch.device(f"cuda:{local_rank}")
        
        model = model.to(device)
        if not isinstance(model, DistributedDataParallel):
            logger.info("Wrapping model with DistributedDataParallel (device cuda:%s)", local_rank)
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    else:
        # 非分布式模式，直接移动到 resolve_device 确定的设备
        # 假设在 main 中 model.to(device) 已经处理过
        pass
    return model
    
def freeze_batchnorm_stats(model: torch.nn.Module) -> int:
    """Freeze BatchNorm running stats to avoid private-data leakage via buffers."""
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count