# utils/seed_utils.py
"""
Utilities for setting random seeds and controlling deterministic behavior.

Call `set_global_seed(seed)` early in main.py to make experiments more reproducible.
"""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed to set.
        deterministic: If True, enable PyTorch deterministic mode
                       (may slow down training).
    """
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # Environment variable for some CUDA backends
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch CPU / CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # A common choice: allow CuDNN to pick fastest algorithm
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
