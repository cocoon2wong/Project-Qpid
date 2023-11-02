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

# use MM implementations for 1D Haar, since they're faster than conv
from ._daubachies_conv import DaubWaveLayer1D, InvDaubWaveLayer1D
from ._haar_mm import HaarWaveLayer1D, InvHaarWaveLayer1D


class WaveTFFactory(object):
    """Factory for different wavelet transforms (1D/2D, haar/db2)"""
    @staticmethod
    def build(kernel_type='db2', dim=2, inverse=False):
        """Build chosen wavelet layer

        :param kernel_type: 'haar' or 'db2'
        :param dim: 1 or 2
        :param inverse: True if computing anti-transform
        :returns: Chosen wavelet layer
        :rtype: torch.nn.Module

        """
        l = None
        if (dim != 1 and dim != 2):
            raise ValueError(
                'Only 1- and 2-dimensional wavelet supported yet.')
        elif (kernel_type not in ['haar', 'db2']):
            raise ValueError('Kernel type can be either "haar" or "db2".')
        # direct wavelet
        elif (inverse == False):
            if (kernel_type == 'haar'):
                if (dim == 1):
                    l = HaarWaveLayer1D()
                else:
                    pass
                    # l = HaarWaveLayer2D()
            elif (kernel_type == 'db2'):
                if (dim == 1):
                    l = DaubWaveLayer1D()
                else:
                    pass
                    # l = DaubWaveLayer2D()
        # inverse wavelet
        else:
            if (kernel_type == 'haar'):
                if (dim == 1):
                    l = InvHaarWaveLayer1D()
                else:
                    pass
                    # l = InvHaarWaveLayer2D()
            elif (kernel_type == 'db2'):
                if (dim == 1):
                    l = InvDaubWaveLayer1D()
                else:
                    pass
                    # l = InvDaubWaveLayer2D()

        if l is None:
            raise NotImplementedError

        return l
