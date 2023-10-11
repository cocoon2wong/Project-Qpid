"""
@Author: Conghao Wong
@Date: 2022-10-12 10:50:35
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-11 10:46:39
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from .__layers import BaseLossLayer


class AIOU(BaseLossLayer):
    """
    Calculate the average IoU on predicted bounding boxes among the `time` axis.
    It is only used for models with `anntype == 'boundingbox'`.
    Each dimension of the predictions should be `(xl, yl, xr, yr)`.
    """

    def forward(self, outputs: list, labels: list, inputs: list,
                mask=None, training=None, *args, **kwargs):

        pred = outputs[0]
        GT = labels[0]
        mask_count = torch.sum(mask)

        # reshape to (..., K, steps, dim)
        if pred.ndim == GT.ndim:
            pred = pred[..., None, :, :]

        K = pred.shape[-3]
        GT = torch.repeat_interleave(GT[..., None, :, :], K, dim=-3)

        dim = pred.shape[-1]
        if dim == 4:
            func = IoU_single_2Dbox
        elif dim == 6:
            func = IoU_single_3Dbox
        else:
            raise ValueError(dim)

        iou = func(pred, GT)
        iou = torch.mean(iou, dim=-1)
        iou = torch.max(iou, dim=-1)[0]
        iou = torch.maximum(iou, 0.0)
        return torch.sum(iou * mask)/mask_count


class FIOU(AIOU):
    """
    Calculate the IoU on the final prediction time step.
    It is only used for models with `anntype == 'boundingbox'`.
    Each dimension of the predictions should be `(xl, yl, xr, yr)`.
    """

    def forward(self, outputs: list, labels: list, inputs: list,
                mask=None, training=None, *args, **kwargs):

        return super().forward([outputs[0][..., -1, None, :]],
                               [labels[0][..., -1, None, :]],
                               inputs, mask, training, *args, **kwargs)


def IoU_single_3Dbox(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU on pred and GT.
    Boxes must have the same shape.

    - box1:

    |  0  |  1  |  2  |  3  |  4  |  5  |
    |-----|-----|-----|-----|-----|-----|
    | xl1 | yl1 | zl1 | xr1 | yr1 | zr1 |

    - box2:

    |  0  |  1  |  2  |  3  |  4  |  5  |
    |-----|-----|-----|-----|-----|-----|
    | xl2 | yl2 | zl2 | xr2 | yr2 | zr2 |

    :param box1: Shape = (..., 6).
    :param box2: Shape = (..., 6).
    """

    len1_x, len2_x, len_inter_x = \
        __get_len_1Dbox(box1[..., [0, 3]], box2[..., [0, 3]])

    len1_y, len2_y, len_inter_y = \
        __get_len_1Dbox(box1[..., [1, 4]], box2[..., [1, 4]])

    len1_z, len2_z, len_inter_z = \
        __get_len_1Dbox(box1[..., [2, 5]], box2[..., [2, 5]])

    v_inter = len_inter_x * len_inter_y * len_inter_z
    v_box1 = len1_x * len1_y * len1_z
    v_box2 = len2_x * len2_y * len2_z
    iou = v_inter / (v_box1 + v_box2 - v_inter)
    return iou


def IoU_single_2Dbox(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU on pred and GT.
    Boxes must have the same shape.

    - box1:

    |  0  |  1  |  2  |  3  |
    |-----|-----|-----|-----|
    | xl1 | yl1 | xr1 | yr1 |

    - box2:

    |  0  |  1  |  2  |  3  |
    |-----|-----|-----|-----|
    | xl2 | yl2 | xr2 | yr2 |

    :param box1: Shape = (..., 4).
    :param box2: Shape = (..., 4).
    """

    len1_x, len2_x, len_inter_x = \
        __get_len_1Dbox(box1[..., [0, 2]], box2[..., [0, 2]])

    len1_y, len2_y, len_inter_y = \
        __get_len_1Dbox(box1[..., [1, 3]], box2[..., [1, 3]])

    s_inter = len_inter_x * len_inter_y
    s_all = len1_x * len1_y + len2_x * len2_y
    iou = s_inter / (s_all - s_inter)
    return iou


def __get_len_1Dbox(box1: torch.Tensor, box2: torch.Tensor):
    """
    The shape of each box should be `(..., 2)`.
    Boxes should have the same shape.
    """
    len1 = torch.abs(box1[..., 0] - box1[..., 1])
    len2 = torch.abs(box2[..., 0] - box2[..., 1])

    len_all = [len1, len2]
    for i in [0, 1]:
        for j in [0, 1]:
            len_all.append(box1[..., i] - box2[..., j])

    len_max = torch.max(torch.abs(len_all), dim=0)[0]
    len_inter = torch.maximum(len1 + len2 - len_max, 0.0)

    return len1, len2, len_inter
