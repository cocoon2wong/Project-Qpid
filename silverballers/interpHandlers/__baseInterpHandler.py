"""
@Author: Conghao Wong
@Date: 2022-11-29 09:26:00
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-11 12:32:02
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from ...constant import INPUT_TYPES
from ...training import Structure
from ..__handlerModel import BaseHandlerModel, HandlerArgs


class _BaseInterpHandlerModel(BaseHandlerModel):
    """
    The basic interpolation handler model.
    Subclass this class and rewrite the `interp` method to add 
    different interpolation layers.
    """

    is_interp_handler = True
    INTERP_LAYER_TYPE: type[torch.nn.Module] = None

    def __init__(self, Args: HandlerArgs,
                 as_single_model: bool = True,
                 structure: Structure = None,
                 *args, **kwargs):

        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        self.args._set('T', 'none')
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.GROUNDTRUTH_TRAJ)
        self.set_preprocess()

        self.accept_batchK_inputs = True
        self.interp_layer = None if not self.INTERP_LAYER_TYPE \
            else self.INTERP_LAYER_TYPE()

        self.ext_traj_wise_outputs = {}
        self.ext_agent_wise_outputs = {}

    def forward(self, inputs: list[torch.Tensor],
                keypoints: torch.Tensor,
                keypoints_index: torch.Tensor,
                training=None, mask=None):

        # Unpack inputs
        trajs = inputs[0]

        if keypoints.ndim >= 4:     # (batch, K, steps, dim)
            K = keypoints.shape[-3]
            trajs = torch.repeat_interleave(trajs[..., None, :, :], K, dim=-3)

        # Concat keypoints with the last observed point
        keypoints_index = torch.concat([torch.tensor([-1], device=keypoints.device),
                                        keypoints_index], dim=0)
        keypoints = torch.concat([trajs[..., -1:, :], keypoints], dim=-2)

        return self.interp(keypoints_index, keypoints, obs_traj=trajs)

    def interp(self, index: torch.Tensor,
               value: torch.Tensor,
               obs_traj: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError('Please rewrite this method.')
