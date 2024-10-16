"""
@Author: Conghao Wong
@Date: 2022-06-20 20:10:58
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-16 20:14:33
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import logging
import os
import plistlib
import time
from typing import TypeVar, overload

import numpy as np
import torch

T = TypeVar('T')

"""
Configs
"""
# Basic parameters
TIME = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

# Paths settings
ROOT_DIR = '.'
ROOT_TEMP_DIR = os.path.join(ROOT_DIR, 'temp_files')
DATASET_CONFIG_DIR = os.path.join(ROOT_DIR, 'dataset_configs')
DEFAULT_LOG_DIR = os.path.join(ROOT_DIR, 'logs')

# Args
ARGS_FILE_NAME = 'args.json'

# Dataset configs
INIT_POSITION = 100000000

# Preprocess configs
ROTATE_BIAS = 0.01
SCALE_THRESHOLD = 0.05

# Log paths and configs
LOG_FILE = os.path.join(ROOT_DIR, 'test.log')
LOG_MAX_LIST_LEN = 10
LOG_STREAM_HANDLER = logging.StreamHandler()

# Weights configs
WEIGHTS_FORMAT = '.pt'
CHECKPOINT_FILENAME = 'best_ade_epoch.txt'

# Type of agents
MAX_TYPE_NAME_LEN = 30


def dir_check(target_dir: str) -> str:
    """
    Used for checking if the `target_dir` exists.
    If it does not exist, it will make it.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    return target_dir


def get_relative_path(reference_file_path: str, relative_path: str):
    return os.path.join(os.path.dirname(reference_file_path), relative_path)


@overload
def move_to_device(item: torch.Tensor, d: torch.device) -> \
    torch.Tensor: ...


@overload
def move_to_device(item: list[torch.Tensor],
                   d: torch.device) -> list[torch.Tensor]: ...


def move_to_device(item, d):
    if isinstance(item, list):
        return [move_to_device(i, d) for i in item]
    elif isinstance(item, torch.Tensor):
        return item.to(d)
    else:
        return item


def load_from_plist(path: str) -> dict:
    """
    Load plist files into python `dict` object.

    :param path: The path of the plist file.
    :return dat: A `dict` object loaded from the file.
    """
    with open(path, 'rb') as f:
        dat = plistlib.load(f)

    return dat


def get_mask(input: torch.Tensor, dtype=torch.float32):
    return (input < 0.05 * INIT_POSITION).to(dtype=dtype)


@overload
def get_loss_mask(obs: torch.Tensor | np.ndarray,
                  label: torch.Tensor | np.ndarray,) -> torch.Tensor: ...


@overload
def get_loss_mask(obs: torch.Tensor | np.ndarray,
                  label: torch.Tensor | np.ndarray,
                  return_numpy: bool) -> np.ndarray: ...


def get_loss_mask(obs: torch.Tensor | np.ndarray,
                  label: torch.Tensor | np.ndarray,
                  return_numpy=False):
    """
    Get mask from both model predictions and labels.
    Return type: `torch.float32`.

    :param obs: Observed trajectories, shape = `(..., steps, dim)`
    :param label: Label trajectories, shape = `(..., steps, dim)`
    """
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)

    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)

    pred_mask = get_mask(torch.sum(obs, dim=[-1, -2]))
    label_mask = get_mask(torch.sum(label, dim=[-1, -2]))
    mask = pred_mask * label_mask
    if return_numpy:
        mask = mask.numpy()
    return mask


def batch_matmul(a: torch.Tensor, b: torch.Tensor, transpose_b=False):
    """
    Run matmul operations on a batch of inputs.
    Other args will be wrapped to `torch.matmul`.

    :param a: Input, shape is `(..., a, b)`.
    :param b: Another input, shape is `(..., b, c)`.
    """
    if transpose_b:
        b = torch.transpose(b, -1, -2)
    if a.ndim <= 4:
        return torch.matmul(a, b)

    batch = a.shape[:-3]
    _a = torch.reshape(a, [-1]+list(a.shape[2:]))
    _b = torch.reshape(b, [-1]+list(b.shape[2:]))
    res = torch.matmul(_a, _b)

    return torch.reshape(res, list(batch) + list(res.shape[1:]))


def encode_string(name: str, depth=MAX_TYPE_NAME_LEN) -> np.ndarray:
    _array = np.array([ord(s) for s in name])

    if (l := len(_array)) < depth:
        _array = np.concatenate([_array, np.zeros(depth - l)])
    else:
        _array = _array[:depth]
    return _array


def decode_string(array: np.ndarray) -> str:
    return ''.join([chr(_a) for _a in array.astype(int) if _a])
