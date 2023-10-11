"""
@Author: Conghao Wong
@Date: 2022-11-28 21:16:28
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-11 13:33:07
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch


class NewtonInterpolation(torch.nn.Module):
    """
    Newton interpolation layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, index: torch.Tensor,
                value: torch.Tensor,
                ord: int, interval: float = 1):
        """
        Newton interpolation.
        The results do not contain the start point.

        :param index: Indexes of keypoints, shape = `(n)`.
        :param value: Keypoints values, shape = `(..., n, dim)`.
        :param ord: The order to calculate interpolations.
        :param interval: The interpolation interval.

        :return yp: Interpolations, shape = `(..., m, dim)`, where
        `m = index[-1] - index[0]`.
        """

        x = index.to(torch.int32)
        y = value
        device = x.device

        diff_quotient = [y]
        for i in range(ord):
            last_res = diff_quotient[i]
            diff_y = last_res[..., :-1, :] - last_res[..., 1:, :]
            diff_x = (x[:-1-i] - x[1+i:])[:, None]
            diff_quotient.append(diff_y/diff_x)

        # shape = (m)
        x_p = torch.arange(x[0]+1, x[-1]+1, step=interval).to(device)

        # shape = (ord+1, ..., dim)
        coe = torch.stack([d[..., 0, :] for d in diff_quotient]).to(device)

        # shape = (m, n)
        xs = x_p[:, None] - x

        xs_prod = [torch.ones_like(x_p)[:, None]]
        for i in range(ord):
            xs_prod.append(torch.prod(xs[:, :i+1], dim=-1, keepdim=True))

        # shape = (m, ord+1)
        xs_prod = torch.concat(xs_prod, dim=-1).to(torch.float32)

        xs_prod = xs_prod[None, None]
        coe = torch.permute(coe, [1, 2, 0, 3])
        return xs_prod @ coe
