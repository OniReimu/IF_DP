"""Device helpers for single- and multi-GPU execution."""

from __future__ import annotations

from typing import List, Optional

import torch

from config import get_logger

logger = get_logger("device")

def _parse_cuda_devices(spec: Optional[str]) -> Optional[List[int]]:
    if not spec:
        return None
    devices: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            devices.append(int(chunk))
        except ValueError as exc:  # pragma: no cover - defensive parsing
            raise ValueError(f"Invalid CUDA device id '{chunk}'") from exc
    return devices or None


def resolve_device(args) -> torch.device:
    """
    Determine the primary torch.device and persist requested CUDA ids on args.
    """
    requested_devices = _parse_cuda_devices(getattr(args, "cuda_devices", None))
    setattr(args, "_cuda_devices", requested_devices)

    if getattr(args, "cpu", False):
        logger.info("Using CPU")
        return torch.device("cpu")

    if getattr(args, "mps", False):
        if torch.backends.mps.is_available():
            logger.info("Using MPS")
            return torch.device("mps")
        logger.warn("MPS requested but unavailable; falling back to CPU/CUDA.")

    if torch.cuda.is_available():
        device_ids = requested_devices
        if device_ids is None and getattr(args, "multi_gpu", False):
            visible = torch.cuda.device_count()
            if visible > 1:
                device_ids = list(range(visible))
                setattr(args, "_cuda_devices", device_ids)
        if device_ids:
            primary = device_ids[0]
            msg = f"Using CUDA devices {device_ids} (primary cuda:{primary})" if len(device_ids) > 1 else f"Using CUDA:{primary}"
            logger.info(msg)
        else:
            primary = getattr(args, "cuda_id", None)
            primary = primary if primary is not None else 0
            logger.info("Using CUDA:%s", primary)
        return torch.device(f"cuda:{primary}")

    logger.info("Using CPU")
    return torch.device("cpu")


def maybe_wrap_model_for_multi_gpu(model: torch.nn.Module, args):
    """
    Wrap the model with DataParallel when multiple CUDA devices are requested.
    """
    if not torch.cuda.is_available():
        return model

    device_ids = getattr(args, "_cuda_devices", None)
    if device_ids is None:
        if getattr(args, "multi_gpu", False):
            visible = torch.cuda.device_count()
            if visible > 1:
                device_ids = list(range(visible))
                setattr(args, "_cuda_devices", device_ids)
        if device_ids is None:
            return model

    if len(device_ids) <= 1:
        return model

    if isinstance(model, torch.nn.DataParallel):
        return model

    logger.info("Enabling DataParallel on CUDA devices %s", device_ids)
    return torch.nn.DataParallel(model, device_ids=device_ids)


def freeze_batchnorm_stats(model: torch.nn.Module) -> int:
    """Freeze BatchNorm running stats to avoid private-data leakage via buffers."""
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count
