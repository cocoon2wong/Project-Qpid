"""
@Author: Conghao Wong
@Date: 2022-06-21 09:26:56
@LastEditors: Conghao Wong
@LastEditTime: 2024-03-22 09:42:19
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np

from ...utils import INIT_POSITION
from ..__base import BaseInputObject


class Agent(BaseInputObject):
    """
    Agent
    ---
    Structure to manage data of one `agent-based` training sample.
    It could be managed by `AgentFilesManager` or `AgentObjectmanager`.

    Universal Properties
    ---
    ```python
    (property) id: (self: Self@BaseInputObject) -> str
    (property) type: (self: Self@BaseInputObject) -> str
    (property) frames: (self: Self@BaseInputObject) -> ndarray
    (property) frames_future: (self: Self@BaseInputObject) -> ndarray
    ```

    Trajectory-Related Properties
    ---
    ```python
    # Mask
    (property) mask: (self: Self@BaseInputObject) -> ndarray

    # Observed trajectory
    (property) traj: (self: Self@Agent) -> ndarray

    # Observed trajectory (without paddings)
    (property) traj_masked: (self: Self@BaseInputObject) -> ndarray

    # All neighbors' observed trajectories
    (property) traj_neighbor: (self: Self@Agent) -> ndarray

    # Predicted trajectory
    (property) pred: (self: Self@BaseInputObject) -> (ndarray | None)

    # Predicted trajectory (without paddings)
    (property) pred_masked: (self: Self@BaseInputObject) -> (ndarray | None)

    # Linearly predicted trajectory
    (property) pred_linear: (self: Self@Agent) -> (ndarray | None)

    # All neighbors' linearly predicted trajectories
    (property) pred_linear_neighbor: (self: Self@Agent) -> (ndarray | None)

    # Ground truth future trajectory
    (property) groundtruth: (self: Self@Agent) -> (ndarray | None)

    # Ground truth futhre trajectory (without paddings)
    (property) groundtruth_masked: (self: Self@BaseInputObject) -> (ndarray | None)

    # Ground truth destination (the last point of ground truth trajectory)
    (property) destination: (self: Self@BaseInputObject) -> (ndarray | None)
    ```
    """

    __version__ = 8.0
    _save_items = ['__version__',
                   '_traj', '_traj_future',
                   '_traj_pred', '_traj_linear',
                   '_id', '_type',
                   '_frames', '_frames_future',
                   'linear_predict',
                   'obs_length', 'total_frame',
                   'neighbor_number',
                   '_traj_neighbor',
                   '_traj_linear_neighbor']

    def __init__(self):

        super().__init__()
        self.neighbor_number = 0
        self._traj_neighbor: np.ndarray | None = None
        self._traj_neighbor_force = None
        self._traj_linear_neighbor: np.ndarray | None = None

    @property
    def traj(self) -> np.ndarray:
        """
        historical trajectory, shape = (obs, dim)
        """
        if self._traj is None:
            raise ValueError
        return self.pickers.get(self._traj)

    @property
    def traj_neighbor(self) -> np.ndarray:
        """
        All neighbors' historical trajectories (after padding), 
        shape = (max_agents, obs, dim).
        NOTE: Returned trajectories are all reletive, corresponding to
        current agents' last observed point.
        """
        if self._traj_neighbor_force is not None:
            return self._traj_neighbor_force[..., :self.obs_length, :]

        if self._traj_neighbor is None:
            raise ValueError

        ref = self.traj[..., -1:, :]
        return self.padding(self.pickers.get(self._traj_neighbor[..., :self.obs_length, :])) - ref

    @property
    def groundtruth_neighbor(self) -> np.ndarray:
        """
        All neighbors' groundtruth trajectories (after padding), 
        shape = (max_agents, pred, dim).
        NOTE: Returned trajectories are all reletive, corresponding to
        current agents' last observed point.
        """
        if self._traj_neighbor_force is not None:
            return self._traj_neighbor_force[..., self.obs_length:, :]

        if self._traj_neighbor is None:
            raise ValueError

        ref = self.traj[..., -1:, :]
        return self.padding(self.pickers.get(self._traj_neighbor[..., self.obs_length:, :])) - ref

    @traj_neighbor.setter
    def traj_neighbor(self, value):
        self._traj_neighbor_force = value

    @property
    def pred_linear(self) -> np.ndarray | None:
        """
        linear prediction.
        shape = (pred, dim)
        """
        return self.pickers.get(self._traj_linear)

    @property
    def pred_linear_neighbor(self) -> np.ndarray | None:
        """
        linear prediction of neighbors' trajectories.
        shape = (n, pred, dim)
        """
        return self.pickers.get(self._traj_linear_neighbor)

    @property
    def groundtruth(self) -> np.ndarray | None:
        """
        ground truth future trajectory.
        shape = (pred, dim).
        """
        return self.pickers.get(self._traj_future)

    def write_pred(self, pred: np.ndarray):
        self._traj_pred = pred

    def init_data(self, id: str,
                  type: str,
                  target_traj: np.ndarray,
                  neighbors_traj: np.ndarray,
                  frames: np.ndarray,
                  start_frame, obs_frame, end_frame,
                  frame_step=1,
                  linear_predict=True):
        """
        Make one training data.

        NOTE that `start_frame`, `obs_frame`, `end_frame` are
        indexes of frames, not their ids.
        Length (time steps) of `target_traj` and `neighbors_traj`
        are `(end_frame - start_frame) // frame_step`.
        """

        self.linear_predict = linear_predict

        # Trajectory info
        self.obs_length = (obs_frame - start_frame) // frame_step
        self.total_frame = (end_frame - start_frame) // frame_step

        self._id = id
        self._type = type
        self._traj = target_traj[:self.obs_length]
        self._traj_future = target_traj[self.obs_length:]

        frames = np.array(frames)
        self._frames = frames[:self.obs_length]
        self._frames_future = frames[self.obs_length:]

        # Neighbor info
        self.clear_all_neighbor_info()

        traj_neighbor_fixed = []
        for _n_traj in neighbors_traj.copy():
            if _n_traj.max() == INIT_POSITION:
                index = np.where(_n_traj.T[0] != INIT_POSITION)[0]
                _n_traj[:index[0], :] = _n_traj[index[0]]
                _n_traj[index[-1]:, :] = _n_traj[index[-1]]

                if _n_traj.max() == INIT_POSITION:
                    continue

            traj_neighbor_fixed.append(_n_traj)

        self._traj_neighbor = np.array(traj_neighbor_fixed)
        self.neighbor_number = len(traj_neighbor_fixed)

        if linear_predict:
            pred_frames = self.total_frame - self.obs_length
            n = self.neighbor_number

            self._traj_linear = linear_pred(self._traj,
                                            self.obs_length,
                                            pred_frames)

            _n_pred = linear_pred(
                np.concatenate(
                    self._traj_neighbor[..., :self.obs_length, :], axis=-1),
                self.obs_length,
                pred_frames
            )

            _n_pred = np.reshape(_n_pred, [pred_frames, n, -1])
            _n_pred = np.transpose(_n_pred, [1, 0, 2])
            self._traj_linear_neighbor = _n_pred

        return self

    def clear_all_neighbor_info(self):
        self._traj_neighbor = None
        self._traj_linear_neighbor = None


class Trajectory():
    """
    Entire Trajectory
    -----------------
    Manage one agent's entire trajectory in datasets.

    Properties
    ----------
    ```python
    >>> self.id
    >>> self.traj
    >>> self.neighbors
    >>> self.frames
    >>> self.start_frame
    >>> self.end_frame
    ```
    """

    def __init__(self, agent_id: str,
                 agent_type: str,
                 trajectory: np.ndarray,
                 neighbors: list[list[int]],
                 frames: np.ndarray,
                 init_position: float):
        """
        init

        :param agent_index: ID of the trajectory.
        :param agent_type: The type of the agent.
        :param neighbors: A list of lists that contain agents' ids \
            who appear in each frame. \
            index are frame indexes.
        :param trajectory: The target trajectory, \
            shape = `(all_frames, 2)`.
        :param frames: A list of frame ids, \
            shape = `(all_frames)`.
        :param init_position: The default position that indicates \
            the agent has gone out of the scene.
        """

        self._id = agent_id
        self._type = agent_type
        self._traj = trajectory
        self._neighbors = neighbors
        self._frames = frames

        base = np.concatenate([[init_position],
                               self.traj.T[0],
                               [init_position]])
        diff = base[:-1] - base[1:]

        appear = np.where(diff > init_position/2)[0]
        # disappear in the next step
        disappear = np.where(diff < -init_position/2)[0]

        self._start_frame = appear[0] if len(appear) else 0
        self._end_frame = disappear[0] if len(disappear) else len(base - 2)

    @property
    def id(self):
        return self._id

    @property
    def type(self):
        return self._type

    @property
    def traj(self):
        """
        Trajectory, shape = `(frames, 2)`
        """
        return self._traj

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def frames(self):
        """
        frame id that the trajectory appears.
        """
        return self._frames

    @property
    def start_frame(self):
        """
        index of the first observed frame
        """
        return self._start_frame

    @property
    def end_frame(self):
        """
        index of the last observed frame
        """
        return self._end_frame

    def sample(self, start_frame, obs_frame, end_frame,
               matrix,
               frame_step=1,
               max_neighbor=15) -> Agent:
        """
        Sample training data from the trajectory.

        NOTE that `start_frame`, `obs_frame`, `end_frame` are
        indexes of frames, not their ids.
        """
        neighbors = np.array(self.neighbors[obs_frame - frame_step])

        if len(neighbors) > max_neighbor + 1:
            nei_pos = matrix[obs_frame - frame_step, list(neighbors), :]
            tar_pos = self.traj[obs_frame - frame_step, np.newaxis, :]
            dis = calculate_length(nei_pos - tar_pos)
            neighbors = neighbors[np.argsort(dis)[1:max_neighbor+1]]

        nei_traj = matrix[start_frame:end_frame:frame_step, list(neighbors), :]
        nei_traj = np.transpose(nei_traj, [1, 0, 2])
        tar_traj = self.traj[start_frame:end_frame:frame_step, :]

        return Agent().init_data(
            id=self.id,
            type=self.type,
            target_traj=tar_traj,
            neighbors_traj=nei_traj,
            frames=self.frames[start_frame:end_frame:frame_step],
            start_frame=start_frame,
            obs_frame=obs_frame,
            end_frame=end_frame,
            frame_step=frame_step,
        )


def calculate_length(vec1):
    return np.linalg.norm(vec1, axis=-1)


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)


def __predict_linear(x, y, x_p, P):
    """
    Linear prediction.

    :param x: shape = (batch, obs).
    :param y: shape = (batch, pred).
    :param P: shape = (obs, obs).
    """

    A = np.stack([np.ones_like(x), x]).T        # (obs, 2)
    A_p = np.stack([np.ones_like(x_p), x_p]).T  # (pred, 2)
    Y = y.T  # (obs)
    B = np.linalg.inv(A.T @ P @ A) @ A.T @ P @ Y
    Y_p = A_p @ B
    return Y_p


def linear_pred(inputs, obs_frames, pred_frames,
                diff_weights=0.95) -> np.ndarray:

    if diff_weights == 0:
        P = np.diag(np.ones(shape=[obs_frames]))
    else:
        P = np.diag(softmax([(i+1)**diff_weights for i in range(obs_frames)]))

    t = np.arange(obs_frames)
    t_p = np.arange(obs_frames + pred_frames)
    dim = inputs.shape[-1]

    inputs = np.transpose(inputs, [1, 0])
    pred = __predict_linear(t, inputs, t_p, P)
    return pred[obs_frames:]
