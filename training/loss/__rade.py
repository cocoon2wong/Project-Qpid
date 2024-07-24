"""
@Author: Conghao Wong
@Date: 2024-07-24 15:27:25
@LastEditors: Conghao Wong
@LastEditTime: 2024-07-24 16:17:22
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from ...constant import INPUT_TYPES
from .__ade import ADE_2D
from .__layers import BaseLossLayer, pick_keypoints_from_gt


class RADE(BaseLossLayer):
    """
    Relative ADE.
    For each agent, `RADE = ADE/OBS`,
    where the `OBS` is the moving length during the observation period.
    """

    def forward(self, outputs: list[torch.Tensor], labels: list[torch.Tensor],
                inputs: list[torch.Tensor], points: int = -1,
                mask=None, training=None, *args, **kwargs):

        pred = outputs[0]
        label = self.model.get_label(labels, INPUT_TYPES.GROUNDTRUTH_TRAJ)
        obs = self.model.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)

        # Compute move length
        l = torch.norm(obs[..., -1, :] - obs[..., 0, :], dim=-1)

        if (pred.shape[-2] == label.shape[-2]) and (points > 0):
            pred = pred[..., :points, :]
            label = label[..., :points, :]
        else:
            label = pick_keypoints_from_gt(self, pred, label)

        # Expand to (..., K, pred, dim)
        if pred.ndim == label.ndim:
            pred = pred[..., None, :, :]

        ade = []
        picker = self.picker.get_coordinate_series
        for _pred, _label in zip(picker(pred), picker(label)):
            coe = 1/(l + 1e-8)
            coe = coe * (coe < 1e6)
            ade.append(ADE_2D(_pred, _label, coe, mask))

        return torch.mean(torch.stack(ade))


class RFDE(RADE):
    """
    Relative FDE.
    For each agent, `RFDE = FDE/OBS`,
    where the `OBS` is the moving length during the observation period.
    """

    def forward(self, outputs: list, labels: list, inputs: list,
                index: int = -1,
                mask=None, training=None, *args, **kwargs):

        pred = outputs[0]
        label = self.model.get_label(labels, INPUT_TYPES.GROUNDTRUTH_TRAJ)
        if (pred.shape[-2] != label.shape[-2]):
            index = -1
            label = pick_keypoints_from_gt(self, pred, label)

        return super().forward([pred[..., index, None, :]],
                               [label[..., index, None, :]],
                               inputs,
                               mask=mask, training=training,
                               *args, **kwargs)
