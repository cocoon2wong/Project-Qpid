"""
@Author: Conghao Wong
@Date: 2023-06-12 10:16:03
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-02 15:44:13
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import torch

from ...model.layers import LinearLayerND
from ...utils import get_loss_mask
from ..__base import BaseInputObject

LINEAR_LAYER = None


class Frame(BaseInputObject):
    """
    Frame
    ---
    Structure to manage data of one `frame-based` training sample.
    It could be managed by `FrameFilesManager` or `FrameObjectmanager`.

    Universal Properties
    ---
    ```python
    (property) agent_ids: (self: Self@Frame) -> (ndarray | None)
    (property) agent_types: (self: Self@Frame) -> (ndarray | None)
    (property) frames: (self: Self@BaseInputObject) -> ndarray
    (property) frames_future: (self: Self@BaseInputObject) -> ndarray
    ```

    Trajectory-Related Properties
    ---
    ```python
    # Mask
    (property) mask: (self: Self@BaseInputObject) -> ndarray

    # Observed trajectory
    (property) traj: (self: Self@Frame) -> ndarray

    # Observed trajectory (without paddings)
    (property) traj_masked: (self: Self@BaseInputObject) -> ndarray

    # Predicted trajectory
    (property) pred: (self: Self@BaseInputObject) -> (ndarray | None)

    # Predicted trajectory (without paddings)
    (property) pred_masked: (self: Self@BaseInputObject) -> (ndarray | None)

    # Linearly predicted trajectory
    (property) pred_linear: (self: Self@Frame) -> (ndarray | None)

    # Ground truth future trajectory
    (property) groundtruth: (self: Self@Frame) -> (ndarray | None)

    # Ground truth futhre trajectory (without paddings)
    (property) groundtruth_masked: (self: Self@BaseInputObject) -> (ndarray | None)

    # Ground truth destination (the last point of ground truth trajectory)
    (property) destination: (self: Self@BaseInputObject) -> (ndarray | None)
    ```
    """

    __version__ = 0.1
    _save_items = ['__version__',
                   '_traj', '_traj_future',
                   '_traj_pred', '_traj_linear',
                   '_init_position',
                   '_id', '_agent_ids', '_agent_types',
                   '_frames', '_frames_future',
                   'linear_predict',
                   'obs_length', 'total_frame']

    def __init__(self):

        super().__init__()

        self._type = 'Frame'
        self._agent_ids: np.ndarray | None = None
        self._agent_types: np.ndarray | None = None
        self._init_position: float | None = None

    @property
    def traj(self) -> np.ndarray:
        """
        Trajectory matrix of all observed frames.
        Shape is `(n_agent, obs, dim)`.
        The position of agents that are not in the scene will be
        represented as a big float number (`init_position` in this
        object).
        """
        if self._traj is None:
            raise ValueError
        return self.padding(self.pickers.get(self._traj))

    @property
    def agent_ids(self) -> np.ndarray | None:
        return self._agent_ids

    @property
    def agent_types(self) -> np.ndarray | None:
        return self._agent_types

    @property
    def groundtruth(self) -> np.ndarray | None:
        """
        ground truth future trajectory.
        shape = (n_agent, pred, dim)
        """
        if self._traj_future is None:
            return None
        return self.padding(self.pickers.get(self._traj_future))

    @property
    def pred_linear(self) -> np.ndarray | None:
        """
        linear prediction.
        shape = (n_agent, pred, dim)
        """
        if self._traj_linear is None:
            return None
        return self.padding(self.pickers.get(self._traj_linear))

    def write_pred(self, pred: np.ndarray):
        self._traj_pred = self._get_masked_traj(pred)

    def init_data(self, id: str,
                  traj: np.ndarray,
                  frames: np.ndarray,
                  agent_ids: np.ndarray,
                  agent_types: np.ndarray,
                  start_frame: int,
                  obs_frame: int,
                  end_frame: int,
                  init_position: float,
                  frame_step: int = 1,
                  linear_predict=True):
        """
        Make one training data.

        NOTE that `start_frame`, `obs_frame`, `end_frame` are
        indices of frames, not their ids.
        Length (time steps) of `target_traj` and `neighbors_traj`
        are `(end_frame - start_frame) // frame_step`.
        """

        self.linear_predict = linear_predict
        self._init_position = init_position

        # Trajectory info
        self.obs_length = (obs_frame - start_frame) // frame_step
        self.total_frame = (end_frame - start_frame) // frame_step

        # Write trajectories
        _x = traj[..., :self.obs_length, :]
        _y = traj[..., self.obs_length:, :]
        _mask = get_loss_mask(_x, _y)
        _valid_indices = np.where(_mask)[0]

        self._traj = _x[_valid_indices]
        self._traj_future = _y[_valid_indices]

        self._id = id
        self._agent_ids = agent_ids[_valid_indices]
        self._agent_types = agent_types[_valid_indices]

        frames = np.array(frames)
        self._frames = frames[:self.obs_length]
        self._frames_future = frames[self.obs_length:]

        if linear_predict:
            pred_frames = self.total_frame - self.obs_length

            global LINEAR_LAYER
            if not LINEAR_LAYER:
                LINEAR_LAYER = LinearLayerND(obs_frames=self.obs_length,
                                             pred_frames=pred_frames)

            self._traj_linear = LINEAR_LAYER(
                torch.from_numpy(self._traj)).numpy()

        return self
