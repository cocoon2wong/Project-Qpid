"""
@Author: Conghao Wong
@Date: 2023-10-10 15:25:33
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-11 10:37:43
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import torch


class Dense(torch.nn.Module):
    def __init__(self, input_units: int,
                 output_units: int,
                 activation: type[torch.nn.Module] = None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(input_units, output_units)
        self.activation = activation() if activation else None

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x


class Dropout(torch.nn.Module):
    def __init__(self, p: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.p = p

    def forward(self, x: torch.Tensor, training=None):
        return torch.nn.functional.dropout(x, self.p, training)
