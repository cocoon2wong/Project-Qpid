"""
@Author: Conghao Wong
@Date: 2023-10-17 16:16:40
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-17 17:48:01
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from ...args import EmptyArgs
from ...constant import ARG_TYPES

DYNAMIC = ARG_TYPES.DYNAMIC
STATIC = ARG_TYPES.STATIC
TEMPORARY = ARG_TYPES.TEMPORARY


class ContextMapsArgs(EmptyArgs):
    @property
    def use_seg_maps(self) -> int:
        """
        Controls whether to use segmentation maps instead of the
        calculated trajectory maps.
        """
        return self._arg('use_seg_maps', 0, argtype=DYNAMIC)
