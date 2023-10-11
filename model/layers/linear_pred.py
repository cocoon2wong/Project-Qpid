"""
@Author: Conghao Wong
@Date: 2021-12-21 15:19:11
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-11 19:36:13
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

    def forward(self, inputs: torch.Tensor, **kwargs):
        """
        Linear prediction

        :param inputs: input trajs, shape = (batch, obs, dim)
        :param results: linear pred, shape = (batch, pred, dim)
        """
        dim = inputs.shape[-1]
        device = inputs.device

        results = []
        for d in range(dim):
            x = inputs[..., [d]].to(dtype=torch.float32)
            Bx = self.W.to(device) @ x
            results.append(self.A_p.to(device) @ Bx)

        results = torch.concat(results, dim=-1)
        return results[..., -self.f:, :]
