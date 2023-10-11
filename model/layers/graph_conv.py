"""
@Author: Conghao Wong
@Date: 2021-12-21 15:20:57
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-10 20:44:21
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from .basic import Dense


class GraphConv(torch.nn.Module):
    """
    Graph conv layer
    """

    def __init__(self, input_units: int,
                 output_units: int,
                 activation: type[torch.nn.Module] = None,
                 *args, **kwargs):
        """
        Init a graph convolution layer

        :param units: output feature dimension
        :param activation: activations used in the output layer
        """
        super().__init__(*args, **kwargs)

        self.fc = Dense(input_units, output_units, activation)

    def forward(self, features: torch.Tensor,
                adjMatrix: torch.Tensor,
                *args, **kwargs) -> torch.Tensor:
        """
        Run the graph convolution operation

        :param features: feature sequences, shape = (batch, N, M)
        :param adjMatrix: adj matrix, shape = (batch, N, N)
        :return outputs: shape = (batch, N, units)
        """

        dot = torch.matmul(adjMatrix, features)
        return self.fc(dot)
