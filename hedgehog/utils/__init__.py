"""
Utility functions for hedgehog.
"""

import os
import torch
import logging
from typing import Optional, List
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("hedgehog")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get logger instance."""
    if name:
        return logging.getLogger(f"hedgehog.{name}")
    return logging.getLogger("hedgehog")


def get_device(device: Optional[str] = None) -> torch.device:
    """Get torch device with automatic detection.

    Priority: CUDA > MPS > CPU

    Args:
        device: Device string (cpu, cuda, mps, auto) or None for auto-detection
    """
    if device and device != "auto":
        return torch.device(device)

    # Auto-detect
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_count() -> int:
    """Get number of available devices."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        return 1  # MPS doesn't support multi-device
    else:
        return 1


def set_device(local_rank: Optional[int] = None) -> str:
    """Set device for distributed training."""
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_size(model: torch.nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def safe_save_checkpoint(
    checkpoint: dict,
    checkpoint_path: str,
) -> None:
    """Safely save checkpoint."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def safe_load_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> dict:
    """Safely load checkpoint."""
    return torch.load(checkpoint_path, map_location=device)


def find_free_port() -> int:
    """Find a free port for distributed training."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def setup_distributed() -> dict:
    """Setup distributed training environment variables."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    node_rank = int(os.environ.get("NODE_RANK", 0))

    return {
        "local_rank": local_rank,
        "world_size": world_size,
        "node_rank": node_rank,
    }


def collate_fn(batch: List[dict]) -> dict:
    """Collate function for dataloader."""
    result = {}
    for key in batch[0].keys():
        result[key] = torch.stack([item[key] for item in batch])
    return result


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
