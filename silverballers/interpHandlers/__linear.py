"""
@Author: Conghao Wong
@Date: 2022-11-29 09:39:09
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-11 12:33:27
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from ...model.layers import interpolation
from .__baseInterpHandler import _BaseInterpHandlerModel


class LinearHandlerModel(_BaseInterpHandlerModel):
    INTERP_LAYER_TYPE = interpolation.LinearPositionInterpolation

    def interp(self, index: torch.Tensor,
               value: torch.Tensor,
               obs_traj: torch.Tensor) -> torch.Tensor:

        # Calculate linear interpolation -> (batch, pred, 2)
        return self.interp_layer(index, value)


class LinearSpeedHandlerModel(_BaseInterpHandlerModel):
    INTERP_LAYER_TYPE = interpolation.LinearSpeedInterpolation

    def interp(self, index: torch.Tensor,
               value: torch.Tensor,
               obs_traj: torch.Tensor) -> torch.Tensor:

        # Calculate linear interpolation -> (batch, pred, 2)
        v0 = obs_traj[..., -1:, :] - obs_traj[..., -2:-1, :]
        return self.interp_layer(index, value, init_speed=v0)


class LinearAccHandlerModel(_BaseInterpHandlerModel):
    INTERP_LAYER_TYPE = interpolation.LinearAccInterpolation

    def interp(self, index: torch.Tensor,
               value: torch.Tensor,
               obs_traj: torch.Tensor) -> torch.Tensor:

        # Calculate linear interpolation -> (batch, pred, 2)
        v_last = obs_traj[..., -1:, :] - obs_traj[..., -2:-1, :]
        v_second_to_last = obs_traj[..., -2:-1, :] - obs_traj[..., -3:-2, :]
        return self.interp_layer(index, value,
                                 init_speed=v_last,
                                 init_acc=v_last - v_second_to_last)
