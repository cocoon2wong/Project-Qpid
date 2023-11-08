"""
@Author: Conghao Wong
@Date: 2023-11-07 16:34:17
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-08 11:00:40
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import numpy as np

from qpid.base import BaseManager

from ...constant import INPUT_TYPES
from ...dataset import SplitManager
from ...dataset.__base import BaseExtInputManager


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
        order = self.working_clip.order
        scale = self.working_clip.get_manager(SplitManager).scale

        w = [weights[0], weights[2]]
        b = [weights[1], weights[3]]
        
        # X-Y order loaded from opencv is different from others
        iy, ix = order[:2]
        wx = w[ix] * scale
        bx = b[ix]

        wy = w[iy] * scale
        by = b[iy]

        self.W = np.array([wx, wy])
        self.b = np.array([bx, by])

        if not self.temp_file:
            raise ValueError
        
        np.save(self.temp_file, np.array([self.W, self.b]))

    def load(self, *args, **kwargs):
        if not self.temp_file:
            raise ValueError
        
        self.W, self.b = np.load(self.temp_file, allow_pickle=True)[:2]
        
        return np.concatenate([self.W, self.b])
