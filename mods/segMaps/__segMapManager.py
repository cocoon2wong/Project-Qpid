"""
@Author: Conghao Wong
@Date: 2023-11-08 10:04:32
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-08 11:14:45
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import cv2
import numpy as np
import torch

from qpid.base import BaseManager
from qpid.constant import DATASET_CONFIGS, INPUT_TYPES
from qpid.dataset.__base import BaseExtInputManager
from qpid.utils import dir_check

from .__args import SegMapArgs


class SegMapManager(BaseExtInputManager):

    TEMP_FILE = 'pooled_seg_map.png'
    INPUT_TYPE = INPUT_TYPES.SEG_MAP

    is_dataset_wise_input = True

    def __init__(self, manager: BaseManager,
                 name='Segmentation Map Manager'):

        super().__init__(manager, name)

        self.pool_layer: torch.nn.AvgPool2d | None = None
        self.m_args = self.args.register_subargs(SegMapArgs, name=__package__)

        if self.m_args.seg_map_pool_size < 0:
            match self.args.dataset:
                case 'ETH-UCY':
                    v = 10
                case 'SDD':
                    v = 20
                case _:
                    v = 20
            self.m_args._set('seg_map_pool_size', v)

    def save(self, *args, **kwargs):
        seg_map_path = self.working_clip.other_files[DATASET_CONFIGS.SEG_IMG]
        seg_map: np.ndarray = np.array(cv2.imread(seg_map_path))[..., 0]

        if (p := self.m_args.seg_map_pool_size) > 1:
            if self.pool_layer is None:
                self.pool_layer = torch.nn.AvgPool2d((p, p))

            _seg_map = torch.from_numpy(
                seg_map[None, None, ...]).to(torch.float32)
            _seg_map = self.pool_layer(_seg_map)
            seg_map = (_seg_map.numpy()[0, 0, ...]).astype(np.uint8)

        if not self.temp_file:
            raise ValueError

        dir_check(self.temp_dir)
        cv2.imwrite(self.temp_file, seg_map)

    def load(self, *args, **kwargs):
        if not self.temp_file:
            raise ValueError

        return np.array(cv2.imread(self.temp_file), dtype=np.float32)[..., 0]/255
