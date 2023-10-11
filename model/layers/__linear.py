"""
@Author: Conghao Wong
@Date: 2021-12-21 15:19:11
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-10 20:46:59
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch


class LinearLayer(torch.nn.Module):
    def __init__(self, obs_frames: int, pred_frames: int,
                 diff=0.95, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.h = obs_frames
        self.f = pred_frames
        self.diff = diff

        if self.diff == 0:
            P = torch.diag(torch.ones(self.h))
        else:
            P = torch.diag(torch.softmax(torch.tensor(
                [(i+1)**self.diff for i in range(self.h)]), dim=-1))

        self.x = torch.arange(self.h, dtype=torch.float32)
        self.x_p = torch.arange(self.f, dtype=torch.float32) + self.h
        A = torch.t(torch.stack([
            torch.ones([self.h]),
            self.x
        ]))
        self.A_p = torch.t(torch.stack([
            torch.ones([self.f]),
            self.x_p
        ]))
        self.W = torch.inverse(torch.t(A) @ P @ A) @ torch.t(A) @ P

    def forward(self, inputs: torch.Tensor, **kwargs):
        """
        Linear prediction

        :param inputs: input trajs, shape = (batch, obs, 2)
        :param results: linear pred, shape = (batch, pred, 2)
        """
        x = inputs[:, :, 0:1]
        y = inputs[:, :, 1:2]

        Bx = self.W @ x
        By = self.W @ y

        results = torch.stack([
            self.A_p @ Bx,
            self.A_p @ By,
        ])

        results = torch.permute(results[:, :, :, 0], [1, 2, 0])
        return results[:, -self.f:, :]


class LinearLayerND(LinearLayer):
    def __init__(self, obs_frames: int, pred_frames: int,
                 diff=0.95, *args, **kwargs):

        super().__init__(obs_frames, pred_frames, diff, *args, **kwargs)

    def call(self, inputs: torch.Tensor, **kwargs):
        """
        Linear prediction

        :param inputs: input trajs, shape = (batch, obs, dim)
        :param results: linear pred, shape = (batch, pred, dim)
        """
        dim = inputs.shape[-1]

        results = []
        for d in range(dim):
            x = inputs[..., [d]].to(dtype=torch.float32)
            Bx = self.W @ x
            results.append(self.A_p @ Bx)

        results = torch.concat(results, dim=-1)
        return results[..., -self.f:, :]


class LinearInterpolation(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Piecewise linear interpolation
        (Results do not contain the start point)
        """

        super().__init__(*args, **kwargs)

    def call(self, index, value):
        """
        Piecewise linear interpolation
        (Results do not contain the start point)

        :param index: index, shape = `(n)`, where `m = index[-1] - index[0]`
        :param value: values, shape = `(..., n, 2)`
        :return yp: linear interpolations, shape = `(..., m, 2)`
        """

        x = index
        y = value

        linear_results = []
        for output_index in range(x.shape[0] - 1):
            p_start = x[output_index]
            p_end = x[output_index+1]

            # shape = (..., 2)
            start = y[..., output_index, :]
            end = y[..., output_index+1, :]

            for p in range(p_start+1, p_end+1):
                linear_results.append(torch.unsqueeze(
                    (end - start) * (p - p_start) / (p_end - p_start)
                    + start, dim=-2))   # (..., 1, 2)

        # shape = (..., n, 2)
        return torch.concat(linear_results, dim=-2)
