"""
@Author: Conghao Wong
@Date: 2023-06-06 16:45:56
@LastEditors: Conghao Wong
@LastEditTime: 2023-12-06 16:02:47
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.args import Args

from ..args import Args
from ..dataset import AgentManager
from ..model import Model
from ..model.process import PROCESS_TYPES
from ..training import Structure
from .__baseArgs import BaseSilverballersArgs


class BaseSubnetwork(Model):

    def __init__(self, Args: BaseSilverballersArgs,
                 as_single_model: bool = True,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, structure, *args, **kwargs)

        # For type hinting
        self.args: BaseSilverballersArgs
        self.structure: Structure

        # Parameters
        self.as_single_model = as_single_model

        # Preprocess
        preprocess = []
        for index, operation in enumerate([PROCESS_TYPES.MOVE,
                                           PROCESS_TYPES.SCALE,
                                           PROCESS_TYPES.ROTATE]):
            if self.args.preprocess[index] == '1':
                preprocess.append(operation)

        self.set_preprocess(*preprocess)

        # Keypoints and their indices
        indices = [int(i) for i in self.args.key_points.split('_')]
        self.__ki = torch.tensor(indices, dtype=torch.int32)

        self.n_key_past = int(torch.sum((self.__ki < 0).to(torch.int32)))
        self.n_key_future = int(torch.sum((self.__ki >= 0).to(torch.int32)))
        self.n_key = self.n_key_past + self.n_key_future

    @property
    def d(self) -> int:
        """
        Feature dimension used in most layers.
        """
        return self.args.feature_dim

    @property
    def d_id(self) -> int:
        """
        Dimension of the noise vectors.
        """
        return self.args.depth

    @property
    def dim(self) -> int:
        """
        Dimension of the predicted trajectory.
        For example, `dim = 4` for 2D bounding boxes.
        """
        return self.structure.ann_manager.dim

    @property
    def key_indices_future(self) -> torch.Tensor:
        """
        Indices of the future keypoints.
        Data type is `torch.int32`.
        """
        return self.__ki[self.n_key_past:]

    @property
    def key_indices_past(self) -> torch.Tensor:
        """
        Indices of the past keypoints.
        Data type is `torch.int32`.
        It starts with `0`.
        """
        if self.n_key_past:
            k_i = self.args.obs_frames + self.__ki[:self.n_key_past]
            return torch.tensor(k_i, dtype=torch.int32)
        else:
            return torch.tensor([], dtype=torch.int32)

    @property
    def picker(self):
        """
        Trajectory picker (from the top manager object).
        """
        return self.get_top_manager().get_member(AgentManager).picker


class BaseSubnetworkStructure(Structure):

    ARG_TYPE: type[BaseSilverballersArgs] = BaseSilverballersArgs
    MODEL_TYPE: type[BaseSubnetwork] | None = None

    def __init__(self, terminal_args: list[str] | Args,
                 manager=None):

        name = 'Train Manager'
        if isinstance(terminal_args, Args):
            init_args = terminal_args
            name += ' (subnetwork)'
        else:
            init_args = self.ARG_TYPE(terminal_args)

        super().__init__(args=init_args,
                         manager=manager,
                         name=name)

        # For type hinting
        self.args: BaseSilverballersArgs
        self.model: BaseSubnetwork

    def create_model(self, as_single_model=True):
        if self.MODEL_TYPE is None:
            raise ValueError

        self.model = self.MODEL_TYPE(self.args, as_single_model,
                                     structure=self)

    def set_model_type(self, new_type: type[BaseSubnetwork]):
        self.MODEL_TYPE = new_type
