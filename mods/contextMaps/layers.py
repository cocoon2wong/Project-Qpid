"""
@Author: Conghao Wong
@Date: 2023-10-17 17:53:27
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-17 17:54:30
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model.layers import Dense

from .settings import MAP_HALF_SIZE, POOLING_BEFORE_SAVING


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

        feature_len = (MAP_HALF_SIZE*2 // 5)**2
        self.fc = Dense(feature_len, output_channels * units, activation)

        self.reshape_size = (-1, output_channels, units)

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
