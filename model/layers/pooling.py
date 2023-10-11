"""
@Author: Conghao Wong
@Date: 2022-11-21 10:15:13
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-11 10:32:50
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from ...__root import BaseObject


class _BasePooling2D(torch.nn.Module, BaseObject):
    """
    The base pooling layer that supports both CPU and GPU.
    """

    pool_function: type[torch.nn.MaxPool2d] = None

    def __init__(self, pool_size=(2, 2), strides=None, padding=0,
                 data_format: str = 'channels_first',
                 *args, **kwargs):

        torch.nn.Module.__init__(self, *args, **kwargs)
        BaseObject.__init__(
            self, name=f'{type(self).__name__}({hex(id(self))})')

        self.transpose = False
        self.data_format = data_format

        # Torch only supports pooling with `channels_first`
        if self.data_format == 'channels_last':
            self.transpose = True

        self.pool_layer = self.pool_function(pool_size, strides,
                                             padding, **kwargs)

    def forward(self, inputs: torch.Tensor, *args, **kwargs):
        """
        Run the 2D pooling operation.

        :param inputs: The input tensor, shape = `(..., channels, a, b)`
        """
        reshape = False
        if inputs.ndim >= 5:
            reshape = True
            s = list(inputs.shape)
            inputs = torch.reshape(inputs, [-1] + s[-3:])

        # Torch only supports pooling with `channels_first`
        if self.transpose:
            # Reshape inputs with shape (..., a, b, channels)
            i_reshape = torch.permute(inputs, [0, 3, 1, 2])
            pooled = self.pool_layer(i_reshape)
            res = torch.permute(pooled, [0, 2, 3, 1])
        else:
            res = self.pool_layer(inputs)

        if reshape:
            s1 = list(res.shape)
            res = torch.reshape(res, s[:-3] + s1[1:])

        return res


class MaxPooling2D(_BasePooling2D):
    pool_function = torch.nn.MaxPool2d
