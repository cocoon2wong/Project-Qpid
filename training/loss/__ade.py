"""
@Author: Conghao Wong
@Date: 2022-10-12 09:06:50
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-01 21:01:43
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch


def ADE_2D(pred: torch.Tensor,
           GT: torch.Tensor,
           coe: float = 1.0,
           mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Calculate `ADE` or `minADE`.

    :param pred: The predicted trajectories with shape = \
        `(batch, K, steps, 2)` or `(batch, steps, 2)`.
    :param GT: Ground truth future trajectory, shape = `(batch, steps, 2)`.

    :return loss_ade:
        Return `ADE` when input_shape = [batch, pred_frames, 2];
        Return `minADE` when input_shape = [batch, K, pred_frames, 2].
    """
    if pred.ndim == GT.ndim:      # (batch, steps, dim)
        pred = pred[..., None, :, :]

    # Shape of all_ade: (..., K)
    all_ade = torch.mean(
        torch.norm(
            pred - GT[..., None, :, :],
            p=2, dim=-1
        ), dim=-1)

    best_ade = torch.min(all_ade, dim=-1)[0]

    if mask is not None:
        best_ade *= mask
        count = torch.sum(mask)
    else:
        count = torch.sum(torch.ones_like(best_ade))

    return coe * torch.sum(best_ade) / count
