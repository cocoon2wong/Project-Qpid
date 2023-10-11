"""
@Author: Conghao Wong
@Date: 2022-11-28 21:03:40
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-11 12:38:01
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch


class LinearSpeedInterpolation(torch.nn.Module):
    """
    Piecewise linear interpolation on the speed.
    For a trajectory `y(t)`, this interpolation method considers
    the speed as `v(t) = v0 + delta_v * t`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, index: torch.Tensor, 
             value: torch.Tensor,
             init_speed: torch.Tensor):
        """
        Piecewise linear interpolation on the speed.
        The results do not contain the start point.

        :param index: Indexes of keypoints, shape = `(n)`.
        :param value: Keypoints values, shape = `(..., n, dim)`.
        :param init_speed: The initial speed on the last observed
        time step. It should has the shape `(..., 1, dim)`.

        :return yp: Interpolations, shape = `(..., m, dim)`, where
        `m = index[-1] - index[0]`.
        """

        x = index.to(torch.int32)
        y = value

        speeds = [init_speed]
        results = [y[..., 0:1, :]]

        for output_index in range(len(x) - 1):
            x_start = x[output_index]
            x_end = x[output_index+1]
            n = x_end - x_start

            # shape = (..., 1, dim)
            y_start = y[..., output_index:output_index+1, :]
            y_end = y[..., output_index+1:output_index+2, :]
            delta_y = y_end - y_start

            if not x_end - x_start > 1:
                results += [y_end]
                speeds += [y_end - y_start]
                continue

            v0 = speeds[-1]
            delta_v = 2 * (delta_y - n*v0) / (n * (n+1))

            for _ in range(x_start + 1, x_end + 1):
                speeds.append(speeds[-1] + delta_v)
                results.append(results[-1] + speeds[-1])

        return torch.concat(results[1:], dim=-2)
