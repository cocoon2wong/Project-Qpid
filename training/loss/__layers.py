"""
@Author: Conghao Wong
@Date: 2023-06-19 19:16:49
@LastEditors: Conghao Wong
@LastEditTime: 2024-07-24 10:07:31
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import torch

from ...base import BaseManager
from ...constant import INPUT_TYPES
from ...dataset import Annotation as Picker
from ...model import Model
from .__ade import ADE_2D


class BaseLossLayer(torch.nn.Module):

    # Config whether this loss or metric is with some units.
    # For example, l2 loss is with units (like 0.99 meters),
    # while IoU has no units (or its unit is `1`).
    # The unit is the same as datasets' annotations.
    has_unit = False

    def __init__(self, coe: float,
                 manager: BaseManager,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.name = self.__class__.__name__
        self.trainable = False
        self.manager = manager
        self.coe = coe

    @property
    def model(self) -> Model:
        return self.manager.manager.model

    @property
    def picker(self) -> Picker:
        """
        The `Picker` object to transfer annotations.
        """
        return self.manager.picker

    def forward(self, outputs: list[torch.Tensor],
                labels: list[torch.Tensor],
                inputs: list[torch.Tensor], mask=None,
                training=None, *args, **kwargs):

        raise NotImplementedError


class l2(BaseLossLayer):
    """
    l2 loss on the keypoints.
    Support M-dimensional trajectories.
    """
    has_unit = True

    def forward(self, outputs: list, labels: list,
                inputs: list, mask=None,
                training=None, *args, **kwargs):

        label = self.model.get_label(labels, INPUT_TYPES.GROUNDTRUTH_TRAJ)
        label = pick_keypoints_from_gt(self, pred := outputs[0], label)
        weights = self.model.get_input(inputs, INPUT_TYPES.LOSS_WEIGHT)
        coe = self.coe * weights if training else self.coe
        return ADE_2D(pred, label, coe, mask=mask)


class ADE(BaseLossLayer):
    """
    l2 (single-point-wise) loss.
    Support M-dimensional trajectories.
    """
    has_unit = True

    def forward(self, outputs: list, labels: list,
                inputs: list, points: int = -1,
                mask=None, training=None, *args, **kwargs):

        pred = outputs[0]
        label = self.model.get_label(labels, INPUT_TYPES.GROUNDTRUTH_TRAJ)

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
            ade.append(ADE_2D(_pred, _label, self.coe, mask))

        return torch.mean(torch.stack(ade))


class FDE(ADE):
    """
    l2 (single-point-wise) loss on the last prediction point.
    Support M-dimensional trajectories.
    """
    has_unit = True

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


class avgCenter(BaseLossLayer):
    """
    Average displacement error on the center of each prediction.
    """
    has_unit = True
    def forward(self, outputs: list, labels: list, inputs: list,
                mask=None, training=None, *args, **kwargs):

        label = self.model.get_label(labels, INPUT_TYPES.GROUNDTRUTH_TRAJ)
        label = pick_keypoints_from_gt(self, pred := outputs[0], label)
        return ADE_2D(self.picker.get_center(pred),
                      self.picker.get_center(label),
                      self.coe, mask)


class finalCenter(avgCenter):
    """
    Final displacement error on the center of each prediction.
    """
    has_unit = True

    def forward(self, outputs: list, labels: list, inputs: list,
                mask=None, training=None, *args, **kwargs):

        label = self.model.get_label(labels, INPUT_TYPES.GROUNDTRUTH_TRAJ)
        label = pick_keypoints_from_gt(self, pred := outputs[0], label)
        return super().forward([pred[..., -1, None, :]],
                               [label[..., -1, None, :]],
                               inputs,
                               mask=mask, training=training,
                               *args, **kwargs)


def pick_keypoints_from_gt(layer: BaseLossLayer,
                           pred: torch.Tensor,
                           gt: torch.Tensor):
    """
    Pick keypoints from the entire ground truth trajectory.
    """
    if pred.shape[-2] < gt.shape[-2]:
        if not layer.name.endswith('(keypoints)'):
            layer.name += '(keypoints)'
        gt = gt[..., layer.model.output_pred_steps, :]
    return gt
