"""
@Author: Conghao Wong
@Date: 2023-06-19 19:16:49
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-31 10:17:36
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import torch

from ...base import BaseManager
from ...dataset import Annotation as Picker
from .__ade import ADE_2D


class BaseLossLayer(torch.nn.Module):

    # Config whether this loss or metric is with some units.
    # For example, l2 loss is with units (like 0.99 meters),
    # while IoU has no units (or its unit is `1`).
    # The unit is the same as datasets' annotations.
    HAS_UNIT = False

    def __init__(self, coe: float,
                 manager: BaseManager,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.trainable = False
        self.manager = manager
        self.coe = coe
        self.picker: Picker = self.manager.picker

    def forward(self, outputs: list, labels: list,
                inputs: list, mask=None,
                training=None, *args, **kwargs):

        raise NotImplementedError


class l2(BaseLossLayer):
    """
    l2 loss on the keypoints.
    Support M-dimensional trajectories.
    """
    HAS_UNIT = True

    def forward(self, outputs: list, labels: list,
                inputs: list, mask=None,
                training=None, *args, **kwargs):

        return ADE_2D(outputs[0], labels[0], coe=self.coe, mask=mask)


class ADE(BaseLossLayer):
    """
    l2 (single-point-wise) loss.
    Support M-dimensional trajectories.
    """
    HAS_UNIT = True

    def forward(self, outputs: list, labels: list,
                inputs: list, points: int = -1,
                mask=None, training=None, *args, **kwargs):

        pred = outputs[0]
        label = labels[0]
        obs = inputs[0]

        if points > 0:
            pred = pred[..., :points, :]
            label = label[..., :points, :]
            obs = label[..., :points, :]

        # Expand to (..., K, pred, dim)
        if pred.ndim == obs.ndim:
            pred = pred[..., None, :, :]

        ade = []
        picker = self.picker.get_coordinate_series
        for _pred, _label in zip(picker(pred), picker(label)):
            ade.append(ADE_2D(_pred, _label, self.coe, mask))

        return torch.mean(torch.stack(ade))


class FDE(ADE):
    """
    l2 (single-point-wise) loss on the last prediction point.
    Support M-dimensional trajectories.
    """
    HAS_UNIT = True

    def forward(self, outputs: list, labels: list, inputs: list,
                index: int = -1,
                mask=None, training=None, *args, **kwargs):

        return super().forward([outputs[0][..., index, None, :]],
                               [labels[0][..., index, None, :]],
                               inputs,
                               mask=mask, training=training,
                               *args, **kwargs)


class avgCenter(BaseLossLayer):
    """
    Average displacement error on the center of each prediction.
    """
    HAS_UNIT = True
    def forward(self, outputs: list, labels: list, inputs: list,
                mask=None, training=None, *args, **kwargs):

        return ADE_2D(self.picker.get_center(outputs[0]),
                      self.picker.get_center(labels[0]),
                      self.coe, mask)


class finalCenter(avgCenter):
    """
    Final displacement error on the center of each prediction.
    """
    HAS_UNIT = True

    def forward(self, outputs: list, labels: list, inputs: list,
                mask=None, training=None, *args, **kwargs):

        return super().forward([outputs[0][..., -1, None, :]],
                               [labels[0][..., -1, None, :]],
                               inputs,
                               mask=mask, training=training,
                               *args, **kwargs)
