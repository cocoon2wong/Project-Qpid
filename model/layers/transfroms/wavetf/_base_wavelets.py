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

########################################################################
# 1D wavelet
########################################################################

########################################################################
# Direct wavelet transform
# input: (b, x, c) --> output: (b, nx, 2*c)


class DirWaveLayer1D(torch.nn.Module):
    """Abstract class with general methods for 1D wavelet transforms"""
    # in : (b, x, c) --> out: (b, nx, 2*c)

    def forward(self, batch: torch.Tensor):
        """Call the direct 1D wavelet

        :param batch: tensor of shape (batch_size, dim_x, chans)
        :returns: tensor of shape (batch_size, ceil(dim_x/2), 2*chans)
        :rtype: tensor

        """
        self.bs, self.ox, self.cn = batch.shape

        if (self.bs is None):
            self.bs = -1

        self.nx = math.ceil(self.ox / 2)
        self.qx = math.ceil(self.nx / 2)

        return self.kernel_function(batch, device=batch.device)


########################################################################
# Inverse wavelet transforms
# input: (b, x, 2*c) --> output: (b, 2*x, c)
# (with input[b,x,:c] being the channels for L wavelet)


class InvWaveLayer1D(torch.nn.Module):
    """Abstract class with general methods for 1D inverse wavelet transforms"""
    # in : (b, x, 2*c) --> out: (b, 2x, c)

    def forward(self, batch: torch.Tensor):
        """Call the inverse 1D wavelet

        :param batch: tensor of shape (batch_size, dim_x, 2*chans)
        :returns: tensor of shape (batch_size, 2*dim_x, chans)
        :rtype: tensor

        """
        self.bs, self.nx, self.cn = batch.shape

        if (self.bs is None):
            self.bs = -1

        self.ox = self.nx * 2
        self.cn = self.cn // 2
        return self.kernel_function(batch, device=batch.device)
