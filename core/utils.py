# core/utils.py
"""
Miscellaneous utilities for the core training loop.
"""

import os
import time
from typing import Any, Dict

import torch


class Timer:
    """
    Simple context-based timer.

    Example:
        with Timer() as t:
            do_something()
        print(t.elapsed)
    """

    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time


def move_to_device(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """
    Move model to the specified device.
    """
    return model.to(device)


def get_device(config: Dict[str, Any]) -> torch.device:
    """
    Get torch.device from config["device"].
    """
    dev_str = config.get("device", "cuda")
    if dev_str == "cuda" and not torch.cuda.is_available():
        dev_str = "cpu"
    return torch.device(dev_str)


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
) -> str:
    """
    Save a checkpoint (state dict) to disk.

    Args:
        state: dict containing model/optimizer/other states.
        checkpoint_dir: directory to save file.
        filename: filename of checkpoint.

    Returns:
        Full path of saved checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    return path
