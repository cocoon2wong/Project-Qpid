"""
@Author: Conghao Wong
@Date: 2022-11-28 21:29:49
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-30 09:52:02
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from qpid.training import Structure

from ..constant import INTERPOLATION_TYPES
from .__linear import (LinearAccHandlerModel, LinearHandlerModel,
                       LinearSpeedHandlerModel)
from .__newton import NewtonHandlerModel

INTERPOLATION_HANDLER_DICT: dict = {
    INTERPOLATION_TYPES.LINEAR: [Structure, LinearHandlerModel],
    INTERPOLATION_TYPES.LINEAR_SPEED: [Structure, LinearSpeedHandlerModel],
    INTERPOLATION_TYPES.LINEAR_ACC: [Structure, LinearAccHandlerModel],
    INTERPOLATION_TYPES.NEWTON: [Structure, NewtonHandlerModel],
}
