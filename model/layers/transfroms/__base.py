"""
@Author: Conghao Wong
@Date: 2023-05-09 20:25:36
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-11 10:34:16
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from typing import Union

import torch


class _BaseTransformLayer(torch.nn.Module):
    """
    Calculate some Transform on (a batch of) trajectories.
    Method `set_Tshape` and `kernel_function` should be re-write
    when subclassing this class.
    """

    def __init__(self, Oshape: tuple[int, int],
                 *args, **kwargs):
        """
        :param Oshape: The original shape of the layer inputs.\
            It does not contain the `batch_size`.
        """

        super().__init__(*args, **kwargs)

        self.steps = Oshape[0]
        self.channels = Oshape[1]

        # original and transformed shapes
        self.__Oshape = Oshape
        self.__Tshape = self.set_Tshape()

        # switch implementation mode
        # canbe `bigbatch` or `repeat` or `direct`
        self.mode = 0

        self.trainable = False

    @property
    def Oshape(self) -> tuple[int, int]:
        """
        The original shape of the input sequences.
        It does not contain the `batch_size` item.
        For example, `(steps, channels)`.
        """
        return (self.__Oshape[0], self.__Oshape[1])

    @property
    def Tshape(self) -> tuple[int, int]:
        """
        Shape after applying transforms on the input sequences.
        It does not contain the `batch_size` item.
        For example, `(steps, channels)`.
        """
        return (self.__Tshape[0], self.__Tshape[1])

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        """
        Set output shape after the given transform on the input \
            trajectories, whose shapes are `(batch, steps, channels)`.
        It should be calculated with `self.steps` and `self.channels`.
        """
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor, *args, **kwargs):

        # calculate with a big batch size
        if self.mode == 0:
            shape_original = None
            if inputs.ndim >= 4:
                shape_original = list(inputs.shape)
                inputs = torch.reshape(inputs, [-1] + shape_original[-2:])

            outputs = self.kernel_function(inputs, *args, **kwargs)

            if shape_original is not None:
                shape_new = list(outputs.shape)
                outputs = torch.reshape(outputs,
                                        shape_original[:-2] + shape_new[-2:])

        # calculate recurrently
        elif self.mode == 1:
            reshape = False
            if inputs.ndim == 3:
                reshape = True
                inputs = inputs[None, :, :, :]

            outputs = []
            for batch_inputs in inputs:
                outputs.append(self.kernel_function(
                    batch_inputs,
                    *args, **kwargs))

            if reshape:
                outputs = outputs[0]

            outputs = torch.stack(outputs)

        # calculate directly
        elif self.mode == 2:
            outputs = self.kernel_function(inputs, *args, **kwargs)

        else:
            raise ValueError('Mode does not exist.')

        return outputs

    def kernel_function(self, inputs: torch.Tensor,
                        *args, **kwargs) -> torch.Tensor:
        """
        Calculate any kind of transform on a batch of trajectories.

        :param inputs: a batch of agents' trajectories, \
            shape is `(batch, steps, channels)`
        :return r: the transform of trajectories
        """
        raise NotImplementedError


class NoneTransformLayer(_BaseTransformLayer):
    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return [self.steps, self.channels]

    def kernel_function(self, inputs: torch.Tensor, *args, **kwargs):
        return inputs
