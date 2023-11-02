"""
@Author: Conghao Wong
@Date: 2023-05-09 20:30:01
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-01 19:49:08
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from typing import Union

import torch

from .__base import _BaseTransformLayer


class FFTLayer(_BaseTransformLayer):
    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.mode = 0

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return [self.steps, 2*self.channels]

    def kernel_function(self, inputs: torch.Tensor,
                        *args, **kwargs) -> torch.Tensor:
        """
        Run FFT on a batch of trajectories.

        :param inputs: batch inputs, \
            shape = `(batch, steps, channels)`
        :return fft: fft results (real and imag), \
            shape = `(batch, steps, 2*channels)`
        """

        real = []
        imag = []
        for index in range(0, inputs.shape[-1]):
            x_fft = torch.fft.fft(inputs[..., index])[..., None]
            real.append(x_fft.real)
            imag.append(x_fft.imag)

        real = torch.concat(real, dim=-1)
        imag = torch.concat(imag, dim=-1)
        return torch.concat([real, imag], dim=-1)


class FFT2DLayer(_BaseTransformLayer):
    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.mode = 0

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return [self.steps, 2*self.channels]

    def kernel_function(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Run 2D FFT on a batch of trajectories.

        :param inputs: A batch of input trajectories, \
            shape = `(batch, steps, channels)`.
        :return fft: 2D fft results, including real and imag parts, \
            shape = `(batch, steps, 2*channels)`.
        """

        seq = torch.fft.fft2(inputs)
        return torch.concat([seq.real, seq.imag], dim=-1)


class IFFTLayer(_BaseTransformLayer):
    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.mode = 0

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return [self.steps, 2*self.channels]

    def kernel_function(self, inputs: torch.Tensor, *args, **kwargs):
        real = inputs[..., 0: self.channels]
        imag = inputs[..., self.channels: 2*self.channels]

        ffts = []
        for index in range(0, real.shape[-1]):
            r = real[..., index]
            i = imag[..., index]
            ffts.append((torch.fft.ifft(torch.complex(r, i)).real)[..., None])

        return torch.concat(ffts, dim=-1)


class IFFT2Dlayer(_BaseTransformLayer):
    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.mode = 0

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return [self.steps, 2*self.channels]

    def kernel_function(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        real = inputs[..., :self.channels]
        imag = inputs[..., self.channels:]

        seq = torch.complex(real, imag)
        fft = torch.fft.ifft2(seq)

        return fft.real
