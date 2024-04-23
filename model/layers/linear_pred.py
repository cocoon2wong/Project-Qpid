"""
@Author: Conghao Wong
@Date: 2021-12-21 15:19:11
@LastEditors: Conghao Wong
@LastEditTime: 2024-04-23 09:39:46
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch


class LinearLayer(torch.nn.Module):
    def __init__(self, obs_frames: int,
                 pred_frames: int,
                 diff: float = 0.95,
                 return_full_trajectory=False,
                 *args, **kwargs):
        """
        Init a linear trajectory prediction layer.

        :param obs_frames: The number of observed frames ($t_h$).
        :param pred_frames: The number of predicted frames ($t_f$).
        :param diff: The re-weight parameter to assign more attention weights \
            on the frames that are close to the current observation frame \
            when computing the least squares. Set it to `0` to disable this \
            function, and set it to `1.0` to focus only on the current frame.
        :param return_full_trajectory: Set it to `True` to predict trajectories \
            on all time steps, including the observation steps, i.e., the \
            predicted trajectory has the shape `(batch, t_h + t_f, dim)`. \
            Set it to `False` to predict trajectories only on the future steps.
        """

        super().__init__(*args, **kwargs)

        self.t_h = obs_frames
        self.t_f = pred_frames
        self.diff = diff
        self.return_full_trajectory = return_full_trajectory

        if self.diff == 0:
            P = torch.diag(torch.ones(self.t_h))
        else:
            P = torch.diag(torch.softmax(torch.tensor(
                [(i+1)**self.diff for i in range(self.t_h)]), dim=-1))

        self.x = torch.arange(self.t_h, dtype=torch.float32)
        if not return_full_trajectory:
            self.x_p = torch.arange(self.t_f, dtype=torch.float32) + self.t_h
        else:
            self.x_p = torch.arange(self.t_h + self.t_f, dtype=torch.float32)

        A = torch.t(torch.stack([
            torch.ones([len(self.x)]),
            self.x
        ]))
        self.A_p = torch.t(torch.stack([
            torch.ones([len(self.x_p)]),
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

        if not self.return_full_trajectory:
            return results[:, -self.t_f:, :]
        else:
            return results


class LinearLayerND(LinearLayer):
    def __init__(self, obs_frames: int, pred_frames: int,
                 diff=0.95, return_full_trajectory=False,
                 *args, **kwargs):

        super().__init__(obs_frames, pred_frames, diff,
                         return_full_trajectory, *args, **kwargs)

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

        if not self.return_full_trajectory:
            return results[..., -self.t_f:, :]
        else:
            return results
