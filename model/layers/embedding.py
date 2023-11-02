"""
@Author: Conghao Wong
@Date: 2021-12-21 15:25:47
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-01 19:54:07
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from .basic import Dense
from .transfroms import _BaseTransformLayer


class TrajEncoding(torch.nn.Module):
    """
    Encode trajectories into the traj feature
    """

    def __init__(self, input_units: int,
                 output_units: int = 64,
                 activation: type[torch.nn.Module] | None = None,
                 transform_layer: _BaseTransformLayer | None = None,
                 channels_first=True,
                 *args, **kwargs):
        """
        Init a trajectory encoding module.

        :param units: Feature dimension.
        :param activation: Activations used in the output layer.
        :param transform_layer: Controls if encode trajectories \
            with some transform methods (like FFTs).
        :param channels_first: Controls if running computations on \
            the last dimension of the inputs.
        """

        super().__init__(*args, **kwargs)

        self.Tlayer = None
        self.channels_first = channels_first
        fc_input_units = input_units

        if transform_layer:
            fc_input_units = output_units
            self.Tlayer = transform_layer
            Tchannels = self.Tlayer.Tshape[1]
            self.fc2 = Dense(Tchannels, output_units, torch.nn.ReLU)

        self.fc1 = Dense(fc_input_units, output_units, activation)

    def forward(self, trajs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode trajectories into the high-dimension features.

        :param trajs: Trajs, shape = `(batch, N, 2)`.
        :return features: Features, shape = `(batch, N, units)`.      
            NOTE: If the transform layer was set, it will return a feature 
            with the `shape = (batch, Tsteps, units)`.
        """
        if self.Tlayer:
            t = self.Tlayer(trajs)  # (batch, Tsteps, Tchannels)

            if not self.channels_first:
                i = list(range(t.ndim))
                # (batch, Tchannels, Tsteps)
                t = torch.permute(t, i[:-2] + [i[-1], i[-2]])

            fc2 = self.fc2(t)
            return self.fc1(fc2)

        else:
            return self.fc1(trajs)
