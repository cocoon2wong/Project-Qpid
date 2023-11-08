"""
@Author: Conghao Wong
@Date: 2023-11-08 09:56:59
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-08 10:47:11
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from qpid.args import DYNAMIC, STATIC, TEMPORARY, EmptyArgs


class SegMapArgs(EmptyArgs):

    @property
    def seg_map_pool_size(self) -> int:
        """
        Pooling size for the full segmentation map.
        """
        return self._arg('seg_map_pool_size', -1, argtype=STATIC)
