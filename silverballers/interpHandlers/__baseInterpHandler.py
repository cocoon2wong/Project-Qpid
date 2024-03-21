"""
@Author: Conghao Wong
@Date: 2024-03-20 17:27:04
@LastEditors: Conghao Wong
@LastEditTime: 2024-03-21 09:14:38
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.args import Args
from qpid.constant import INPUT_TYPES
from qpid.model import Model


class _BaseInterpHandlerModel(Model):
    """
    The basic interpolation handler model.
    Subclass this class and rewrite the `interp` method to add 
    different interpolation layers.
    """

    INTERP_LAYER_TYPE: type[torch.nn.Module] | None = None

    def __init__(self, Args: Args, structure=None, *args, **kwargs):

        if self.INTERP_LAYER_TYPE is None:
            raise ValueError

        # Init with the force-args
        Args._set('preprocess', '000')
        super().__init__(Args, structure, *args, **kwargs)

        # It only accept interpolation models
        if self.input_pred_steps is None:
            raise ValueError

        # Type hinting
        self.input_pred_steps: torch.Tensor

        # Set model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.GROUNDTRUTH_TRAJ)

        # Layers
        self.interp_layer = self.INTERP_LAYER_TYPE()

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        trajs = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        keypoints = self.get_input(inputs, INPUT_TYPES.GROUNDTRUTH_KEYPOINTS)

        if keypoints.ndim >= 4:     # (batch, K, steps, dim)
            K = keypoints.shape[-3]
            trajs = torch.repeat_interleave(trajs[..., None, :, :], K, dim=-3)

        # Concat keypoints with the last observed point
        keypoints_index = torch.concat([torch.tensor([-1], device=keypoints.device),
                                        self.input_pred_steps.to(keypoints.device)], dim=0)
        keypoints = torch.concat([trajs[..., -1:, :], keypoints], dim=-2)

        return self.interp(keypoints_index, keypoints, obs_traj=trajs)

    def interp(self, index: torch.Tensor,
               value: torch.Tensor,
               obs_traj: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError('Please rewrite this method.')
