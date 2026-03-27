# logging_utils/summary_writer.py
"""
Wrapper around torch.utils.tensorboard.SummaryWriter.

If tensorboard is not available, this module falls back to a dummy writer.
"""

import os
from typing import Optional


class _DummyWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def close(self):
        pass


def create_summary_writer(log_dir: str, exp_name: str, use_tensorboard: bool = True):
    """
    Create a SummaryWriter or a dummy writer if tensorboard is not available.

    Args:
        log_dir: root directory for logs.
        exp_name: experiment name; events will be saved to log_dir/exp_name.
        use_tensorboard: whether to try using tensorboard.

    Returns:
        writer: a SummaryWriter-like object with add_scalar(...) API.
    """
    if not use_tensorboard:
        return _DummyWriter()

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return _DummyWriter()

    exp_dir = os.path.join(log_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=exp_dir)
    return writer
