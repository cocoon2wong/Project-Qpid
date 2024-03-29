"""
@Author: Conghao Wong
@Date: 2022-07-15 20:13:07
@LastEditors: Conghao Wong
@LastEditTime: 2024-03-18 16:09:06
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.args import DYNAMIC, Args, EmptyArgs
from qpid.model import Model, layers
from qpid.training import Structure


class LinearArgs(EmptyArgs):
    @property
    def weights(self) -> float:
        """
        The weights in the calculation of the mean squared error at 
        different moments of observation.
        Set to `0.0` to disable this function.
        """
        return self._arg('weights', default=0.0, argtype=DYNAMIC, short_name='w')


class LinearModel(Model):
    def __init__(self, Args: Args, structure=None, *args, **kwargs):
        super().__init__(Args, structure, *args, **kwargs)

        self.l_args = self.args.register_subargs(LinearArgs, 'linearArgs')
        self.linear = layers.LinearLayerND(obs_frames=self.args.obs_frames,
                                           pred_frames=self.args.pred_frames,
                                           diff=self.l_args.weights)

    def forward(self, inputs: list[torch.Tensor], training=None, *args, **kwargs):
        trajs = inputs[0]
        return self.linear(trajs)


class Linear(Structure):
    """
    Linear trajectory prediction model.
    It calculates future trajectories by least squares.
    For each trajectory dimension (like x and y in the 2D coordinates),
    it computs trajectories in the future period via `x = a_x * t + b_x`
    and `y = a_y * t + b_y`.
    """
    is_trainable = False

    def create_model(self, *args, **kwargs):
        self.model = LinearModel(self.args, structure=self)
