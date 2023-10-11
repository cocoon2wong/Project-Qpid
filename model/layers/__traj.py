"""
@Author: Conghao Wong
@Date: 2021-12-21 15:25:47
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-11 12:57:05
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from ...utils import MAP_HALF_SIZE, POOLING_BEFORE_SAVING
from .__base import Dense
from .transfroms import _BaseTransformLayer


class TrajEncoding(torch.nn.Module):
    """
    Encode trajectories into the traj feature
    """

    def __init__(self, input_units: int,
                 output_units: int = 64,
                 activation: type[torch.nn.Module] = None,
                 transform_layer: _BaseTransformLayer = None,
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


class ContextEncoding(torch.nn.Module):
    """
    Encode context maps into the context feature
    """

    def __init__(self, output_channels: int,
                 units: int = 64,
                 activation=None,
                 *args, **kwargs):
        """
        Init a context encoding module.
        The context encoding layer finally outputs a `torch.Tensor`
        with shape `(batch_size, output_channels, units)`.

        :param output_channels: Output channels.
        :param units: Output feature dimension.
        :param activation: Activations used in the output layer.
        """

        super().__init__(*args, **kwargs)

        if not POOLING_BEFORE_SAVING:
            self.pool = torch.nn.MaxPool2d([5, 5])

        self.flatten = torch.nn.Flatten()

        feature_len = (MAP_HALF_SIZE*2 / 5)**2
        self.fc = Dense(feature_len, output_channels * units, activation)

        self.reshape_size = (output_channels, units)

    def forward(self, context_map: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode context maps into context features.

        :param context_map: Maps, shape = `(batch, a, a)`.
        :return feature: Features, shape = `(batch, output_channels, units)`.
        """
        if not POOLING_BEFORE_SAVING:
            context_map = self.pool(context_map[:, None])

        flat = self.flatten(context_map)
        fc = self.fc(flat)
        return torch.reshape(fc, self.reshape_size)
