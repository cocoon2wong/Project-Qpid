"""
@Author: Conghao Wong
@Date: 2023-11-07 16:34:17
@LastEditors: Conghao Wong
@LastEditTime: 2024-03-07 15:23:22
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import cv2
import numpy as np

from qpid.base import BaseManager
from qpid.constant import DATASET_CONFIGS, INPUT_TYPES
from qpid.dataset import SplitManager
from qpid.dataset.__base import BaseExtInputManager
from qpid.utils import dir_check

from .settings import NORMALIZED_SIZE


class SegMapParasManager(BaseExtInputManager):

    TEMP_FILE = 'configs.npy'
    INPUT_TYPE = INPUT_TYPES.SEG_MAP_PARAS

    is_dataset_wise_input = True

    def __init__(self, manager: BaseManager,
                 name='Segmentation Map Parameters Manager'):

        super().__init__(manager, name)

        self.W: np.ndarray = np.array(None)
        self.b: np.ndarray = np.array(None)

    def save(self, *args, **kwargs):
        """
        Weights are `real2pixel` weights:
        - `x_pixel = (wx * scale * x_real + bx) // POOL_SIZE`
        - `y_pixel = (wy * scale * y_real + by) // POOL_SIZE`
        """
        weights = self.working_clip.matrix
        scale = self.working_clip.get_manager(SplitManager).scale

        # Load the original map to get scene size
        _m = cv2.imread(self.working_clip.other_files[DATASET_CONFIGS.SEG_IMG])
        lenx, leny = _m.shape[:2]

        # Coefficient after resizing
        coex, coey = [lenx/NORMALIZED_SIZE, leny/NORMALIZED_SIZE]

        w = [weights[0], weights[2]]
        b = [weights[1], weights[3]]

        ix, iy = [0, 1]
        wx = w[ix] * scale / coex
        bx = b[ix] / coex

        wy = w[iy] * scale / coey
        by = b[iy] / coey

        self.W = np.array([wx, wy])
        self.b = np.array([bx, by])

        if not self.temp_file:
            raise ValueError

        dir_check(self.temp_dir)
        np.save(self.temp_file, np.array([self.W, self.b]))

    def load(self, *args, **kwargs):
        if not self.temp_file:
            raise ValueError

        self.W, self.b = np.load(self.temp_file, allow_pickle=True)[:2]

        return np.concatenate([self.W, self.b])
