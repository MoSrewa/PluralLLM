import torch
import time
import logging
from collections import OrderedDict
from matplotlib import pyplot as plt
"""
Module: logging_utils

This module provides utility classes and functions for logging, monitoring, and summarizing training or evaluation metrics.

It includes:
- A unified logger for consistent console and file logging.
- A RunningAverage class to track and compute averages over time.
- A helper function to convert RunningAverage values into a dictionary format for logging or visualization.
"""

def get_logger(name=None, filename="app.log", level=logging.INFO, mode="a"):
    """
    Returns a unified logger instance with the same format for both console and file.
    - `name`: The name of the logger (module-level logging).
    - `filename`: Log file name.
    - `level`: Logging level (default: INFO).
    - `mode`: File mode ('a' for append, 'w' for overwrite).
    """
    logger = logging.getLogger(name if name else "unified_logger")

    # Prevent duplicate handlers
    if not logger.hasHandlers():    
        logger.setLevel(level)

        # Define a unified log format
        log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Create a file handler
        file_handler = logging.FileHandler(filename, mode=mode)
        file_handler.setFormatter(log_format)

        # Create a console handler (same format as file)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


class RunningAverage(object):
    def __init__(self, *keys):
        self.sum = OrderedDict()
        self.cnt = OrderedDict()
        self.clock = time.time()
        for key in keys:
            self.sum[key] = 0
            self.cnt[key] = 0

    def update(self, key, val):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if self.sum.get(key, None) is None:
            self.sum[key] = val
            self.cnt[key] = 1
        else:
            self.sum[key] = self.sum[key] + val
            self.cnt[key] += 1

    def reset(self):
        for key in self.sum.keys():
            self.sum[key] = 0
            self.cnt[key] = 0
        self.clock = time.time()

    def clear(self):
        self.sum = OrderedDict()
        self.cnt = OrderedDict()
        self.clock = time.time()

    def keys(self):
        return self.sum.keys()

    def get(self, key):
        assert(self.sum.get(key, None) is not None)
        return self.sum[key] / self.cnt[key]

    def info(self, show_et=True):
        line = ''
        for key in self.sum.keys():
            val = self.sum[key] / self.cnt[key]
            if type(val) == float:
                line += f'{key} {val:.4f} '
            else:
                line += f'{key} {val} '.format(key, val)
        if show_et:
            line += f'({time.time()-self.clock:.3f} secs)'
        return line

def running_average_to_dict(ravg,source="centralized"):
    return {f"{source} - {k}": ravg.get(k) for k in ravg.keys()}
