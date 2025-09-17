"""
@Author: Conghao Wong
@Date: 2025-09-16 19:27:53
@LastEditors: Conghao Wong
@LastEditTime: 2025-09-16 21:24:25
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

from typing import Any

from qpid import sys_mgr
from qpid.constant import ANN_TYPES

from .__normal2D import Normal2DCanvas
from .__plt2D import PLT2DCanvas
from .__plt3DSkeleton17 import PLT3DSkeleton17Canvas


def get(anntype: str,
        video_status: bool | Any,
        image_status: bool | Any,
        force_using_plt: bool | Any = False):
    """
    Get the type of used canvas manager, under the given conditions.
    """

    if force_using_plt or ((video_status is None) and (image_status is None)):
        if anntype == ANN_TYPES.CO_2D:
            h = PLT2DCanvas
        elif anntype == ANN_TYPES.SKE_3D_17:
            h = PLT3DSkeleton17Canvas
        else:
            __report_error(anntype)

    else:
        if anntype in [ANN_TYPES.CO_2D, ANN_TYPES.BB_2D]:
            h = Normal2DCanvas
        else:
            __report_error(anntype)

    return h


def __report_error(anntype: str):
    s = f'Annotation type `{anntype}` is not currently supported to visualize.'
    sys_mgr.log(s, level='error', raiseError=NotImplementedError)
