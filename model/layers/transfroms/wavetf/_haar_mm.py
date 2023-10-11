# Copyright 2020 CRS4 (http://www.crs4.it/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch

from ._base_wavelets import DirWaveLayer1D, InvWaveLayer1D

########################################################################
# 1D wavelet
########################################################################

########################################################################
# Direct wavelet transform
# input: (b, x, c) --> output: (b, nx, 2*c)


class HaarWaveLayer1D(DirWaveLayer1D):
    """1D direct Haar trasform"""
    ########################################################################
    # Init (with wavelet kernels)
    ########################################################################

    def __init__(self, **kwargs):
        # Haar kernel
        s2 = math.sqrt(2) * .5  # 1/sqrt(2)
        self.haar_ker = torch.tensor(
            [s2, s2, s2, -s2]).reshape((2, 2)).to(torch.float32)

        # call constructor
        super(DirWaveLayer1D, self).__init__(**kwargs)

    ########################################################################
    # Haar wavelet
    ########################################################################
    def haar_0(self, t1: torch.Tensor):
        # t1: (b, c, x) with (x % 2) == 0
        # out: (b, c, nx, 2)
        return torch.reshape(t1, [-1, self.cn, self.nx, 2])

    def haar_1(self, t1: torch.Tensor):
        # t1: (b, c, x) with (x % 2) == 1
        # anti-symmetric-reflect padding, a.k.a. asym in matlab
        col1_xb = 2.0 * t1[:, :, -1:]
        col1_b = col1_xb - t1[:, :, -2:-1]  # 2*x_{n-1} - x_{n-2}
        s1 = torch.concat([t1, col1_b], dim=-1)

        # group by 2
        s1 = torch.reshape(s1, [-1, self.cn, self.nx, 2])  # out: (b, c, nx, 2)
        return s1

    def kernel_function(self, input: torch.Tensor, device: torch.device):

        haar_ker = self.haar_ker.to(device)
        mod_x = self.ox % 2

        # input: (b, x, c)
        t1 = torch.permute(input, [0, 2, 1])  # out: (b, c, x)

        # prepare data
        if (mod_x == 0):
            s1 = self.haar_0(t1)
        else:
            s1 = self.haar_1(t1)

        # s1: (b, c, nx, 2)
        # apply kernel to rows
        r = s1 @ haar_ker  # out: (b, c, nx, 2)
        r = torch.permute(r, [0, 2, 3, 1])  # out: (b, nx, 2, c)
        r = torch.reshape(r, [self.bs, self.nx, 2*self.cn])
        return r

########################################################################
# Inverse wavelet transforms
# input: (b, x, 2*c) --> output: (b, 2*x, c)
# (with input[b,x,:c] being the channels for L wavelet)


class InvHaarWaveLayer1D(InvWaveLayer1D):
    """1D inverse Haar trasform"""
    ########################################################################
    # Init (with wavelet kernels)
    ########################################################################

    def __init__(self, **kwargs):
        # Haar kernel
        s2 = math.sqrt(2) * .5  # 1/sqrt(2)
        self.haar_ker = torch.tensor(
            [s2, s2, s2, -s2]).reshape((2, 2)).to(torch.float32)

        # call constructor
        super(InvWaveLayer1D, self).__init__(**kwargs)

    ########################################################################
    # Haar wavelet
    ########################################################################
    def kernel_function(self, input: torch.Tensor, device: torch.device):
        # input: (b, x, 2*c)
        haar_ker = self.haar_ker.to(device)

        # out: (b, x, 2, c)
        t1 = torch.reshape(input, [self.bs, self.nx, 2, self.cn])

        # out: (b, c, x, 2)
        t1 = torch.permute(t1, [0, 3, 1, 2])

        # apply kernel to rows
        r = t1 @ haar_ker  # out: (b, c, x, 2)
        r = torch.reshape(r, [self.bs, self.cn, self.ox])  # out: (b, c, 2*x)
        r = torch.permute(r, [0, 2, 1])
        # out: (b, 2*x, c)
        return r
