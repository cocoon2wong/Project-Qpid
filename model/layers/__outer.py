"""
@Author: Conghao Wong
@Date: 2022-06-20 21:50:44
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-10 17:26:46
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch


class OuterLayer(torch.nn.Module):
    """
    Compute the outer product of two vectors.

    :param a_dim: the last dimension of the first input feature
    :param b_dim: the last dimension of the second input feature
    :param reshape: if `reshape == True`, output shape = `(..., a_dim * b_dim)`
        else output shape = `(..., a_dim, b_dim)`
    """

    def __init__(self, a_dim: int, b_dim: int,
                 reshape=False,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.M = a_dim
        self.N = b_dim
        self.reshape = reshape

    def forward(self, tensorA: torch.Tensor, tensorB: torch.Tensor):
        """
        Compute the outer product of two vectors.

        :param tensorA: shape = (..., M)
        :param tensorB: shape = (..., N)
        :return outer: shape = (..., M, N) if `reshape` is `False`,
            else its output shape = (..., M*N)
        """

        _a = tensorA[..., None]
        _b = tensorB[..., None, :]

        _a = torch.repeat_interleave(_a, self.N, dim=-1)
        _b = torch.repeat_interleave(_b, self.M, dim=-2)

        outer = _a * _b

        if not self.reshape:
            return outer
        else:
            return torch.reshape(outer, list(outer.shape[:-2]) + [self.M*self.N])
