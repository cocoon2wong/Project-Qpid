"""
@Author: Conghao Wong
@Date: 2022-09-01 10:40:50
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-10 16:38:41
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from ...constant import ANN_TYPES, INPUT_TYPES, OUTPUT_TYPES
from ...utils import SCALE_THRESHOLD
from .__base import BaseProcessLayer


class Scale(BaseProcessLayer):
    """
    Scaling length of trajectories' direction vector into 1.
    It will take the last observation point as the reference point when 
    running preprocess, and take the first predicted point as the
    reference point when running postprocess.
    """

    def __init__(self, anntype: str, *args, **kwargs):

        super().__init__(anntype,
                         preprocess_input_types=[INPUT_TYPES.OBSERVED_TRAJ,
                                                 INPUT_TYPES.NEIGHBOR_TRAJ],
                         postprocess_input_types=[OUTPUT_TYPES.PREDICTED_TRAJ])

        if anntype != ANN_TYPES.CO_2D:
            raise NotImplementedError(
                f'Rotate is not supported on `{anntype}`.')

        self._scales = None

    @property
    def scales(self) -> torch.Tensor:
        """
        Scales of all observed trajectories.
        Shape is `(batch, 1, 1)`
        """
        return self._scales

    def update_paras(self, inputs: dict[str, torch.Tensor]) -> None:
        trajs = inputs[INPUT_TYPES.OBSERVED_TRAJ]
        # Move vector, shape = (batch, 2)
        vectors = trajs[..., -1, :] - trajs[..., 0, :]

        # Move distance, shape = (batch)
        scales = torch.norm(vectors, dim=-1)

        # Reshape into (batch, 1, 1)
        scales = scales[:, None, None]

        # Ignore trajectories with small movements
        scales = torch.maximum(scales, torch.tensor(SCALE_THRESHOLD))

        self._scales = scales

    def preprocess(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Scaling length of trajectories' direction vector into 1.
        The reference point when scaling is the `last` observation point.
        """
        outputs = {}
        for _type, _input in inputs.items():
            if _type == INPUT_TYPES.OBSERVED_TRAJ:
                outputs[_type] = self.scale(_input, autoref_index=-1)
            elif _type == INPUT_TYPES.NEIGHBOR_TRAJ:
                outputs[_type] = self.scale_neighbors(_input)
            else:
                raise ValueError(_type)
        return outputs

    def postprocess(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Scale trajectories back to their original.
        The reference point is the `first` prediction point.
        """
        outputs = {}
        for _type, _input in inputs.items():
            if _type == OUTPUT_TYPES.PREDICTED_TRAJ:
                outputs[_type] = self.scale(_input, inverse=True,
                                            autoref_index=0)
            else:
                raise ValueError(_type)
        return outputs

    def scale(self, trajs: torch.Tensor, inverse=False, autoref_index: int = None):
        scales = self.scales
        if inverse:
            scales = 1.0 / scales

        ref_points = trajs[..., None, autoref_index, :]

        while scales.ndim < trajs.ndim:
            scales = scales[..., None]

        return (trajs - ref_points) / scales + ref_points

    def scale_neighbors(self, nei_trajs: torch.Tensor):
        scales = self.scales
        while scales.ndim < nei_trajs.ndim:
            scales = scales[..., None]

        return nei_trajs / scales
