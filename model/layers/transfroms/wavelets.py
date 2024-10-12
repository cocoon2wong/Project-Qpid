"""
@Author: Conghao Wong
@Date: 2023-05-09 20:28:47
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-12 10:42:39
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from typing import Union

import torch

from ..linear_pred import LinearLayerND
from .__base import _BaseTransformLayer
from .wavetf import WaveTFFactory


class Haar1D(_BaseTransformLayer):
    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):

        self.even_fix = False
        if Oshape[0] % 2 == 1:
            self.even_fix = True

        super().__init__(Oshape, *args, **kwargs)

        if self.even_fix:
            self.linear = LinearLayerND(obs_frames=Oshape[0], pred_frames=1,
                                        diff=0, return_full_trajectory=True)

        self.haar = WaveTFFactory.build(kernel_type='haar',
                                        dim=1,
                                        inverse=False)

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        if not self.even_fix:
            return (self.steps//2, self.channels*2)
        else:
            return ((self.steps+1)//2, self.channels*2)

    def kernel_function(self, inputs: torch.Tensor, *args, **kwargs):
        if self.even_fix:
            inputs = self.linear(inputs)

        # (batch, steps, channels) -> (batch, steps//2, 2*channels)
        haar = self.haar(inputs)

        return haar


class InverseHaar1D(_BaseTransformLayer):

    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):

        self.even_fix = False
        if Oshape[0] % 2 == 1:
            self.even_fix = True

        super().__init__(Oshape, *args, **kwargs)

        self.haar = WaveTFFactory.build(kernel_type='haar',
                                        dim=1,
                                        inverse=True)

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        if not self.even_fix:
            return (self.steps//2, self.channels*2)
        else:
            return ((self.steps+1)//2, self.channels*2)

    def kernel_function(self, inputs: torch.Tensor,
                        *args, **kwargs) -> torch.Tensor:

        # (batch, steps//2, 2*channels) -> (batch, steps, channels)
        r = self.haar(inputs)

        if self.even_fix:
            r = r[..., :-1, :]

        return r


class DB2_1D(_BaseTransformLayer):

    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.daub = WaveTFFactory.build(kernel_type='db2',
                                        dim=1,
                                        inverse=False)

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return (self.steps//2, self.channels*2)

    def kernel_function(self, inputs: torch.Tensor, *args, **kwargs):
        return self.daub(inputs)


class InverseDB2_1D(_BaseTransformLayer):

    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.daub = WaveTFFactory.build(kernel_type='db2',
                                        dim=1,
                                        inverse=True)

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return (self.steps//2, self.channels*2)

    def kernel_function(self, inputs: torch.Tensor, *args, **kwargs):
        return self.daub(inputs)
