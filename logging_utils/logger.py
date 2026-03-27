# logging_utils/logger.py
"""
Simple logging wrapper around the standard logging module.
"""

import logging
import os
from typing import Optional


def create_logger(
    log_dir: str,
    exp_name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a logger that logs to both console and a file.

    Args:
        log_dir: directory to store log files.
        exp_name: experiment name used as log file prefix.
        level: logging level.

    Returns:
        logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(exp_name)
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs if root has handlers

    if logger.handlers:
        # already created
        return logger

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_path = os.path.join(log_dir, f"{exp_name}.log")
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
