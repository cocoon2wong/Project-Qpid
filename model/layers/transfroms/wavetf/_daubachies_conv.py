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
import torch.nn.functional as F

from ._base_wavelets import DirWaveLayer1D, InvWaveLayer1D

########################################################################
# 1D wavelet
########################################################################

########################################################################
# Direct wavelet transform
# input: (b, x, c) --> output: (b, nx, 2*c)


class DaubWaveLayer1D(DirWaveLayer1D):
    """1D direct Daubechies-N=2 trasform"""
    ########################################################################
    # Init (with wavelet kernels)
    ########################################################################

    def __init__(self, **kwargs):
        # Daubechies kernel
        d = math.sqrt(2) * .125  # 1/(4*sqrt(2))
        r3 = math.sqrt(3)
        h0 = d * (1+r3)
        h1 = d * (3+r3)
        h2 = d * (3-r3)
        h3 = d * (1-r3)
        g0 = h3
        g1 = -h2
        g2 = h1
        g3 = -h0

        self.daubechies_ker = torch.tensor([
            h0, g0,
            h1, g1,
            h2, g2,
            h3, g3]).reshape((4, 2)).to(torch.float32)

        # call constructor
        super(DirWaveLayer1D, self).__init__(**kwargs)

    ########################################################################
    # Daubechies wavelet
    ########################################################################
    def daub_cols1(self, t1: torch.Tensor):
        # anti-symmetric-reflect padding, a.k.a. asym in matlab
        col1_xa = 2.0 * t1[:, :, 0:1]
        col1_xb = 2.0 * t1[:, :, -1:]

        col1_a = col1_xa - t1[:, :, 1:2]  # 2*x_0 - x_1
        col1_b = col1_xb - t1[:, :, -2:-1]  # 2*x_{n-1} - x_{n-2}
        col1_c = col1_xb - t1[:, :, -3:-2]  # 2*x_{n-1} - x_{n-3}
        return [col1_a, col1_b, col1_c]

    def daub_0(self, t1: torch.Tensor):
        # t1: (b, c, x) with (x % 2) == 0
        col1_a, col1_b, _ = self.daub_cols1(t1)
        s1 = torch.concat([col1_a, t1, col1_b], dim=-1)  # out: (b, c, 2*nx)
        return s1

    def daub_1(self, t1: torch.Tensor):
        # t1: (b, c,x) with (x % 2) == 1
        col1_a, col1_b, col1_c = self.daub_cols1(t1)
        s1 = torch.concat([col1_a, t1, col1_b, col1_c], dim=-1)
        return s1

    def kernel_function(self, input: torch.Tensor, device: torch.device):
        daubechies_ker = self.daubechies_ker.to(device)
        mod_x = self.ox % 4

        # input: (b, x, c)
        t1 = torch.permute(input, [0, 2, 1])  # out: (b, c, x)

        # prepare data
        if (mod_x == 0):
            s1 = self.daub_0(t1)
        else:
            s1 = self.daub_1(t1)

        # s1: (b, c, 2*nx)
        nx_dim = s1.shape[2]
        s1 = torch.reshape(s1, [-1, 1, nx_dim])  # out: (b*c, 1, 2*nx')

        # build kernels and apply to rows
        k1l = torch.reshape(daubechies_ker[:, 0], (1, 1, 4))
        k1h = torch.reshape(daubechies_ker[:, 1], (1, 1, 4))

        rl = F.conv1d(s1, k1l, stride=2, padding='valid')
        rh = F.conv1d(s1, k1h, stride=2, padding='valid')

        r = torch.concat((rl, rh), dim=-2)
        r = torch.transpose(r, -1, -2)  # out: (b*c, nx, 2)

        # out: (b, c, nx, 2)
        r = torch.reshape(r, [self.bs, self.cn, self.nx, 2])
        r = torch.permute(r, [0, 2, 3, 1])  # out: (b, nx, 2, c)

        # out: (b, nx, 2*c)
        r = torch.reshape(r, [self.bs, self.nx, 2*self.cn])
        return r

########################################################################
# Inverse wavelet transforms
# input: (b, x) --> output: (b, x), x must be even


class InvDaubWaveLayer1D(InvWaveLayer1D):
    """1D inverse Daubechies-N=2 trasform"""
    ########################################################################
    # Init (with wavelet kernels)
    ########################################################################

    def __init__(self, **kwargs):
        # Daubechies kernel
        d = math.sqrt(2) * .125  # 1/(4*sqrt(2))
        r3 = math.sqrt(3)
        h0 = d * (1+r3)
        h1 = d * (3+r3)
        h2 = d * (3-r3)
        h3 = d * (1-r3)
        g0 = h3
        g1 = -h2
        g2 = h1
        g3 = -h0

        self.daubechies_ker = torch.tensor([
            h2, h3,
            g2, g3,
            h0, h1,
            g0, g1]).reshape((4, 2)).to(torch.float32)

        # matrix for border effect 0 (begin)
        ker_0 = torch.tensor([
            h0, h1, h2,
            g0, g1, g2,
            0, 0, h0,
            0, 0, g0]).reshape((4, 3)).to(torch.float32)
        pad_0 = torch.tensor([
            2., -1.,
            1., 0.,
            0., 1.]).reshape((3, 2)).to(torch.float32)

        self.inv_bor_0 = torch.permute(
            torch.linalg.pinv(ker_0 @ pad_0), [1, 0])

        # matrix for border effect 1 (end)
        ker_1 = torch.tensor([
            h3, 0, 0,
            g3, 0, 0,
            h1, h2, h3,
            g1, g2, g3]).reshape((4, 3)).to(torch.float32)
        pad_1 = torch.tensor([
            1., 0.,
            0., 1.,
            -1., 2.]).reshape((3, 2)).to(torch.float32)
        self.inv_bor_1 = torch.permute(
            torch.linalg.pinv(ker_1 @ pad_1), [1, 0])

        # call constructor
        super(InvWaveLayer1D, self).__init__(**kwargs)

    def kernel_function(self, input: torch.Tensor, device: torch.device):
        # input: (b, nx, 2*c)
        daubechies_ker = self.daubechies_ker.to(device)
        inv_bor_0 = self.inv_bor_0.to(device)
        inv_bor_1 = self.inv_bor_1.to(device)

        #######################################
        # reshape
        #######################################
        # out: (b, ox, c)
        t1 = torch.reshape(input, [self.bs, self.ox, self.cn])
        t1 = torch.permute(t1, [0, 2, 1])  # out: (b, c, ox)

        #######################################
        # compute borders
        #######################################
        # border 0
        b_0 = t1[:, :, :4]
        r2_0 = b_0 @ inv_bor_0

        # border 1
        b_1 = t1[:, :, -4:]
        r2_1 = b_1 @ inv_bor_1

        #######################################
        # transform core
        #######################################
        t1 = torch.reshape(t1, [-1, 1, self.ox])  # out: (b*c, 1, ox)

        # apply kernel to rows
        k1l = torch.reshape(daubechies_ker[:, 0], (1, 1, 4))
        k1h = torch.reshape(daubechies_ker[:, 1], (1, 1, 4))

        rl = F.conv1d(t1, k1l, stride=2, padding='valid')
        rh = F.conv1d(t1, k1h, stride=2, padding='valid')

        r1 = torch.concat((rl, rh), dim=-2)
        r1 = torch.transpose(r1, -1, -2)    # out: (b*c, qx, 4)

        # out: (b, c, ox-2)
        r1 = torch.reshape(r1, [self.bs, self.cn, self.ox-2])

        #######################################
        # merge core and borders
        #######################################
        # out: (b, c, nx)
        r = torch.concat((r2_0, r1[:, :, 1:-1], r2_1), dim=-1)
        r = torch.permute(r, [0, 2, 1])  # out: (b, nx, c)

        return r
