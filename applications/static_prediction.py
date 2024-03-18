"""
@Author: Conghao Wong
@Date: 2024-03-18 15:59:40
@LastEditors: Conghao Wong
@LastEditTime: 2024-03-18 16:10:22
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model import Model
from qpid.training import Structure


class StaticModel(Model):
    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        return torch.repeat_interleave(input=inputs[0][..., -1:, :],
                                       repeats=self.args.pred_frames,
                                       dim=-2)


class Static(Structure):
    """
    It is a special prediction model that only used to debug.
    It could not actually predict trajectories but returns all agents'
    observed positions at the current observation step.
    """
    is_trainable = False

    def create_model(self):
        self.model = StaticModel(self.args, structure=self)
