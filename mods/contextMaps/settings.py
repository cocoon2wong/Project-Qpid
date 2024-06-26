"""
@Author: Conghao Wong
@Date: 2023-10-17 17:49:36
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-30 12:10:55
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from qpid.constant import DATASET_CONFIGS

# Context map configs
SEG_IMG = DATASET_CONFIGS.SEG_IMG
RGB_IMG = DATASET_CONFIGS.RGB_IMG

# WINDOW_EXPAND_PIXEL = 0.3
# WINDOW_SIZE_PIXEL = 200.0
WINDOW_EXPAND_PIXEL = 10.0
WINDOW_SIZE_PIXEL = 10.0

WINDOW_EXPAND_METER = 10.0
WINDOW_SIZE_METER = 10.0

MAP_HALF_SIZE = 50  # Local map's half size
AVOID_SIZE = 15     # Avoid size in grid cells when modeling social interaction
INTEREST_SIZE = 20  # Interest size in grid cells when modeling social interaction

POOLING_BEFORE_SAVING = True


class INPUT_TYPES():
    MAP = 'MAP'
    MAP_PARAS = 'MAP_PARAS'
