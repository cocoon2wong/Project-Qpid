"""
@Author: Conghao Wong
@Date: 2023-06-12 15:11:35
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-02 15:39:52
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import copy
from typing import overload

import numpy as np
import torch
from tqdm import tqdm

from ...base import BaseManager
from ...utils import INIT_POSITION, get_loss_mask
from .__picker import AnnotationManager


class BaseInputObject():

    __version__ = 0.0
    _save_items = []

    def __init__(self) -> None:

        self._id: str = 'Not Assigned'
        self._type: str = 'Not Assigned'

        self._traj: np.ndarray | None = None
        self._traj_future: np.ndarray | None = None

        self._traj_pred: np.ndarray | None = None
        self._traj_linear: np.ndarray | None = None

        self._frames: np.ndarray | None = None
        self._frames_future: np.ndarray | None = None

        self.obs_length = 0
        self.total_frame = 0

        self._mask: np.ndarray | None = None
        self.linear_predict = False

        self.manager: BaseManager | None = None

    def copy(self):
        return copy.deepcopy(self)

    @property
    def pickers(self) -> AnnotationManager:
        return self.manager.pickers

    def init_data(self):
        raise NotImplementedError

    def zip_data(self) -> dict[str, object]:
        zipped = {}
        for item in self._save_items:
            zipped[item] = getattr(self, item)
        return zipped

    def load_data(self, zipped_data: dict[str, object]):
        for item in self._save_items:
            if not item in zipped_data.keys():
                continue
            else:
                setattr(self, item, zipped_data[item])
        return self

    def padding(self, trajs: np.ndarray) -> np.ndarray:
        """
        Padding all agents' trajectories.
        Shape should be `(n_agent, steps, dim)`.
        """
        n = len(trajs)

        if not self.manager:
            raise ValueError('Manager object not assigned!')

        m = self.manager.args.max_agents

        if n <= m:
            zero_pad = np.pad(trajs,
                              ((0, m-n), (0, 0), (0, 0)))
            zero_pad[n:, :, :] = INIT_POSITION
        else:
            zero_pad = trajs[:m]

        return zero_pad

    ##########################################
    # Trajectories data
    # (observations, labels, predictions)
    ##########################################
    @property
    def traj(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def traj_neighbor(self):
        return None

    @property
    def traj_masked(self) -> np.ndarray:
        """
        Masked observed trajectories (without paddings).
        """
        return self._get_masked_traj(self.traj)

    @property
    def groundtruth(self) -> np.ndarray | None:
        raise NotImplementedError

    @property
    def destination(self) -> np.ndarray | None:
        if self.groundtruth is None:
            return None
        return self.groundtruth[..., -1:, :]

    @property
    def groundtruth_masked(self):
        """
        Masked groundtruth future trajectories (without paddings).
        """
        return self._get_masked_traj(self.groundtruth)

    @property
    def pred(self) -> np.ndarray | None:
        """
        predicted trajectory, shape = (..., pred, dim).
        """
        return self._traj_pred

    @property
    def pred_linear(self):
        raise NotImplementedError

    @property
    def pred_masked(self):
        """
        Masked future predicted trajectories (without paddings).
        """
        return self._traj_pred

    def write_pred(self, pred: np.ndarray):
        raise NotImplementedError

    @property
    def mask(self) -> np.ndarray:
        """
        The mask matrix to show whether the trajectory is valid.
        """
        if self._mask is None:
            gt_real = self.groundtruth
            gt = gt_real if gt_real is not None else self.traj
            self._mask = get_loss_mask(self.traj, gt, return_numpy=True)
        return self._mask

    ##########################################
    # Agent data
    # (frames, id)
    ##########################################
    @property
    def id(self) -> str:
        """
        Agent ID
        """
        return self._id

    @property
    def type(self) -> str:
        """
        Agent type
        """
        return self._type

    @property
    def frames(self) -> np.ndarray:
        """
        a list of frame indexes during observation and prediction time.
        shape = (obs + pred)
        """
        return np.concatenate([self._frames, self._frames_future])

    @property
    def frames_future(self) -> np.ndarray:
        """
        a list of frame indexes during prediction time.
        shape = (pred)
        """
        if self._frames_future is None:
            raise ValueError

        return self._frames_future

    @overload
    def _get_masked_traj(self, traj: None) -> None: ...

    @overload
    def _get_masked_traj(self, traj: np.ndarray) -> np.ndarray: ...

    def _get_masked_traj(self, traj):
        if traj is None:
            return None

        if ((not issubclass(type(self.mask), np.ndarray)) or
                (self.mask.ndim == 0)):
            return traj
        else:
            index = np.where(self.mask)[0]
            return traj[index]


def get_attributes(objects: BaseInputObject | list[BaseInputObject],
                   name: str,
                   tqdm_description: str | None = None) -> torch.Tensor:
    """
    Get a specific attribute from the list of `BaseInputObject`s,
    and make them into a single `torch.Tensor` tensor to train or test.
    """
    if isinstance(objects, BaseInputObject):
        objects = [objects]

    items: list[np.ndarray] = []
    repeats = tqdm(objects, tqdm_description) if tqdm_description else objects
    for _object in repeats:
        items.append(getattr(_object, name))

    return torch.from_numpy(np.array(items, dtype=np.float32))
