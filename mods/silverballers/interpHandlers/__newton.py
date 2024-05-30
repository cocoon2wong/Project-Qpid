"""
@Author: Conghao Wong
@Date: 2022-11-29 09:49:26
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-30 09:52:31
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model.layers import interpolation

from .__baseInterpHandler import _BaseInterpHandlerModel


class NewtonHandlerModel(_BaseInterpHandlerModel):
    INTERP_LAYER_TYPE = interpolation.NewtonInterpolation

    def interp(self, index: torch.Tensor,
               value: torch.Tensor,
               obs_traj: torch.Tensor) -> torch.Tensor:
        return self.interp_layer(index, value, ord=len(index)-1)
