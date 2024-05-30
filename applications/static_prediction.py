"""
@Author: Conghao Wong
@Date: 2024-03-18 15:59:40
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-30 09:17:01
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.constant import INPUT_TYPES
from qpid.model import Model
from qpid.training import Structure


class StaticModel(Model):
    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        obs = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        return torch.repeat_interleave(input=obs[..., -1:, :],
                                       repeats=self.args.pred_frames,
                                       dim=-2)


class Static(Structure):
    """
    It is a special prediction model that only used to debug.
    It could not actually predict trajectories but returns all agents'
    observed positions at the current observation step.
    """
    is_trainable = False
    MODEL_TYPE = StaticModel
