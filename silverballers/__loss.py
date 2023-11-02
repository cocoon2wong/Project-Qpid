"""
@Author: Conghao Wong
@Date: 2023-09-06 18:51:14
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-02 19:10:00
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from ..training import loss
from .__baseSubnetwork import BaseSubnetworkStructure as _BS


class keyl2(loss.l2):
    """
    l2 loss on the future keypoints.
    Support M-dimensional trajectories.
    """

    def forward(self, outputs: list, labels: list, inputs: list,
                mask=None, training=None, *args, **kwargs):

        indices = self.manager.get_manager(_BS).model.key_indices_future
        return super().forward(outputs, [labels[0][..., indices, :]],
                               inputs, mask, training, *args, **kwargs)


class keyl2_past(loss.l2):
    """
    l2 loss on the future keypoints.
    Support M-dimensional trajectories.
    """

    def forward(self, outputs: list, labels: list, inputs: list,
                mask=None, training=None, *args, **kwargs):

        if (indices := self.manager.get_manager(_BS).model.key_indices_past) is None:
            return 0

        labels_pickled = inputs[0][..., indices, :]
        return super().forward([outputs[1]], [labels_pickled],
                               inputs, mask, training, *args, **kwargs)


class avgKey(loss.ADE):
    """
    l2 (2D-point-wise) loss on the future keypoints.
    """

    def forward(self, outputs: list, labels: list, inputs: list,
                mask=None, training=None, *args, **kwargs):

        pred = outputs[0]
        label = labels[0]

        if pred.ndim == label.ndim:
            pred = pred[..., None, :, :]

        indices = self.manager.get_manager(_BS).model.key_indices_future
        if pred.shape[-2] != len(indices):
            pred = pred[..., indices, :]

        labels_key = label[..., indices, :]

        return super().forward([pred], [labels_key], inputs,
                               mask=mask, training=training, *args, **kwargs)
