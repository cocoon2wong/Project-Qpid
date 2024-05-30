"""
@Author: Conghao Wong
@Date: 2022-07-15 20:13:07
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-29 21:38:06
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.args import DYNAMIC, EmptyArgs
from qpid.constant import INPUT_TYPES
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
    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        self.args._set('model_name', 'Linear Least Squares Prediction Model')

        # This model dose not use any preprocess methods
        self.set_preprocess()

        self.l_args = self.args.register_subargs(LinearArgs, 'linearArgs')
        self.linear = layers.LinearLayerND(obs_frames=self.args.obs_frames,
                                           pred_frames=self.args.pred_frames,
                                           diff=self.l_args.weights)

    def forward(self, inputs: list[torch.Tensor], training=None, *args, **kwargs):
        trajs = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        return self.linear(trajs)


class Linear(Structure):
    """
    Training structure for the linear trajectory prediction model.
    It calculates future trajectories by least squares.
    For each trajectory dimension (like x and y in the 2D coordinates),
    it computs trajectories in the future period via `x = a_x * t + b_x`
    and `y = a_y * t + b_y`.
    """
    is_trainable = False
    MODEL_TYPE = LinearModel
