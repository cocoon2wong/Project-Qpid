"""
@Author: Conghao Wong
@Date: 2023-05-09 20:24:48
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-11 19:42:30
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from .__base import NoneTransformLayer, _BaseTransformLayer
from .fft import FFT2DLayer, FFTLayer, IFFT2Dlayer, IFFTLayer
from .wavelets import DB2_1D, Haar1D, InverseDB2_1D, InverseHaar1D


def get_transform_layers(Tname: str) -> \
        tuple[type[_BaseTransformLayer],
              type[_BaseTransformLayer]]:
    """
    Set transformation layers used when encoding or 
    decoding trajectories.

    :param Tname: name of the transform, canbe
        - `'none'`
        - `'fft'`
        - `'fft2d'`
        - `'haar'`
        - `'db2'`
    """

    if Tname == 'none':
        Tlayer = NoneTransformLayer
        ITlayer = NoneTransformLayer

    elif Tname == 'fft':
        Tlayer = FFTLayer
        ITlayer = IFFTLayer

    elif Tname == 'fft2d':
        Tlayer = FFT2DLayer
        ITlayer = IFFT2Dlayer

    elif Tname == 'haar':
        Tlayer = Haar1D
        ITlayer = InverseHaar1D

    elif Tname == 'db2':
        Tlayer = DB2_1D
        ITlayer = InverseDB2_1D

    else:
        raise ValueError(f'Transform `{Tname}` not supported.')

    return Tlayer, ITlayer
