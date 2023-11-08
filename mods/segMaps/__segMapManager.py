"""
@Author: Conghao Wong
@Date: 2023-11-08 10:04:32
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-08 20:28:21
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import cv2
import numpy as np

from qpid.base import BaseManager
from qpid.constant import DATASET_CONFIGS, INPUT_TYPES
from qpid.dataset.__base import BaseExtInputManager
from qpid.utils import dir_check

from .settings import NORMALIZED_SIZE


class SegMapManager(BaseExtInputManager):

    TEMP_FILE = 'pooled_seg_map.png'
    INPUT_TYPE = INPUT_TYPES.SEG_MAP

    is_dataset_wise_input = True

    def __init__(self, manager: BaseManager,
                 name='Segmentation Map Manager'):

        super().__init__(manager, name)

    def save(self, *args, **kwargs):
        # Load the original segmentation map
        seg_map_path = self.working_clip.other_files[DATASET_CONFIGS.SEG_IMG]
        seg_map: np.ndarray = np.array(cv2.imread(seg_map_path))[..., 0]

        # Resize to the given size
        seg_map = cv2.resize(seg_map, [NORMALIZED_SIZE, NORMALIZED_SIZE])

        if not self.temp_file:
            raise ValueError

        dir_check(self.temp_dir)
        cv2.imwrite(self.temp_file, seg_map)

    def load(self, *args, **kwargs):
        if not self.temp_file:
            raise ValueError

        return np.array(cv2.imread(self.temp_file), dtype=np.float32)[..., 0]/255
