"""
@Author: Conghao Wong
@Date: 2022-08-03 10:50:46
@LastEditors: Conghao Wong
@LastEditTime: 2024-03-21 21:08:04
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..base import BaseManager
from ..constant import INPUT_TYPES
from .__base import (Annotation, AnnotationManager, BaseExtInputManager,
                     BaseInputObject, get_attributes)
from .__splitManager import SplitManager
from .agent_based import AgentFilesManager
from .frame_based import FrameFilesManager


class TrajectoryDataset(Dataset):

    def __init__(self, input_types: list[str], label_types: list[str]):

        self.input_types = input_types
        self.label_types = label_types

        self.data: dict[str, torch.Tensor] = {}
        self.dataset_wise_data: dict[str, list[torch.Tensor]] = {}

        self.dataset_wise_types: list[str] = []

    def add_data(self, data_type: str,
                 data: np.ndarray,
                 is_dataset_wise: bool = False,
                 data_length: int = -1):

        if not is_dataset_wise:
            self._concat_data(data_type, data)

        elif data_length > 0:
            self.dataset_wise_types.append(data_type)
            self._concat_dataset_wise_data(data_type, data)

            key = len(self.dataset_wise_data[data_type]) - 1
            self._concat_data(data_type, key*np.ones(data_length), torch.int32)

        else:
            raise ValueError(data_length)

    def __get(self, data_type: str, index: int) -> torch.Tensor:
        if data_type in self.dataset_wise_types:
            d_index = self.data[data_type][index]
            return self.dataset_wise_data[data_type][d_index]
        else:
            return self.data[data_type][index]

    def __getitem__(self, index: int):
        return (tuple([self.__get(T, index) for T in self.input_types]),
                tuple([self.__get(T, index) for T in self.label_types]))

    def __len__(self):
        return len(self.data[self.input_types[0]])

    def _concat_data(self, key: str, data: np.ndarray,
                     dtype=torch.float32):

        _data = torch.from_numpy(data).to(dtype)
        if not key in self.data.keys():
            self.data[key] = _data
        else:
            self.data[key] = np.concatenate(
                [self.data[key], _data], axis=0)

    def _concat_dataset_wise_data(self, key: str, data: np.ndarray,
                                  dtype=torch.float32):

        _data = torch.from_numpy(data).to(dtype)
        if not key in self.dataset_wise_data.keys():
            self.dataset_wise_data[key] = []

        self.dataset_wise_data[key].append(_data)


class AgentManager(BaseManager):
    """
    AgentManager
    ---
    Structure to manage several training and test `Agent` or `Frame` objects.
    The `AgentManager` object is managed by the `Structure` object.

    Members
    ---
    - `SplitManager`: Manage the dataset split;
    - `AgentFilesManager` (only for `agent-based` models): Manage the save or load
        process of trajectory data (with multiple `Agent` and `AgentManager` objects);
    - `FrameFilesManager` (only for `frame-based` models): Manage the save or load
        process of trajectory data (with multiple `Frame` and `FrameObjectManager` objects);
    - A list of `ExtInputManager` objects to manage all other needed model inputs.
    ```
    """

    def __init__(self, manager: BaseManager, name='Agent Manager'):
        super().__init__(manager=manager, name=name)

        # Dataset split and basic inputs
        self.split_manager = SplitManager(manager=self,
                                          dataset=self.args.dataset,
                                          split=self.args.split)

        if (t := self.args.model_type) == 'agent-based':
            self.file_manager = AgentFilesManager(self)
        elif t == 'frame-based':
            self.file_manager = FrameFilesManager(self)
        else:
            self.log(f'Wrong model type `{t}`!',
                     level='error', raiseError=ValueError)

        # file root paths
        self.base_path: str = 'Not Assigned'
        self.npz_path: str = 'Not Assigned'
        self.maps_dir: str = 'Not Assigned'

        # Settings and variations
        self._agents: list[BaseInputObject] = []
        self.metadata: dict[str, TrajectoryDataset] = {}
        self.model_inputs: list[str] = []
        self.model_labels: list[str] = []
        self.processed_clips: dict[str, list[str]] = {'train': [], 'test': []}

        # Managers for extra model inputs
        self.ext_mgrs: list[BaseExtInputManager] = []

    @property
    def agents(self):
        return self._agents

    @agents.setter
    def agents(self, value: list[BaseInputObject]):
        self._agents = self.update_agents(value)

    @property
    def picker(self) -> Annotation:
        return self.pickers.annotations[self.args.anntype]

    @property
    def pickers(self) -> AnnotationManager:
        return self.manager.get_member(AnnotationManager)

    def set_path(self, npz_path: str):
        self.npz_path = npz_path
        self.base_path = npz_path.split('.np')[0]
        self.maps_dir = self.base_path + '_maps'

    def update_agents(self, agents: list[BaseInputObject]):
        for a in agents:
            a.manager = self
        return agents

    def append(self, target: list[BaseInputObject]):
        self._agents += self.update_agents(target)

    def set_types(self, input_types: list[str], label_types: list[str]):
        """
        Set the type of model inputs and outputs.
        Accept all types in `INPUT_TYPES`.
        """
        if INPUT_TYPES.MAP in input_types:
            from qpid.mods import contextMaps as maps
            p = maps.settings.POOLING_BEFORE_SAVING
            args = self.args.register_subargs(maps.ContextMapsArgs,
                                              name=maps.__name__)

            self.ext_mgrs.append(maps.MapParasManager(self))
            if not args.use_seg_maps:
                self.ext_mgrs.append(maps.TrajMapManager(self, p))
            else:
                self.ext_mgrs.append(maps.TrajMapManager_seg(self, p))
            self.ext_mgrs.append(maps.SocialMapManager(self, p))

        if INPUT_TYPES.SEG_MAP in input_types:
            from qpid.mods import segMaps
            self.ext_mgrs.append(segMaps.SegMapManager(self))

        if INPUT_TYPES.SEG_MAP_PARAS in input_types:
            from qpid.mods import segMaps
            self.ext_mgrs.append(segMaps.SegMapParasManager(self))

        self.model_inputs = input_types
        self.model_labels = label_types

    def clean(self):
        """
        Clean all loaded data and agent objects in this manager.
        """
        self.agents = []
        return self

    def make(self, clips: str | list[str], training: bool) -> DataLoader:
        """
        Load train samples and make the `torch.utils.data.DataLoader`
        object to train.

        :param clips: Clips to load.
        :param training: The load mode.
        :return dataset: The loaded `torch.utils.data.DataLoader` object.
        """
        if isinstance(clips, str):
            clips = [clips]

        # shuffle agents and video clips when training
        if training:
            mode = 'train'
            random.shuffle(clips)
        else:
            mode = 'test'

        # Init data-holder
        self.metadata[mode] = TrajectoryDataset(input_types=self.model_inputs,
                                                label_types=self.model_labels)

        # load agent data in each video clip
        for clip_name in self.timebar(clips):
            # get clip info
            clip = self.split_manager.clips_dict[clip_name]

            # update time bar
            s = f'Prepare data of {mode} agents in `{clip.clip_name}`...'
            self.update_timebar(s, pos='start')

            # Get new agents
            agents = self.file_manager.run(clip)
            self.append(agents)

            # Load extra model inputs
            trajs = None
            for mgr in self.ext_mgrs:
                key = mgr.INPUT_TYPE
                if key is None:
                    raise ValueError

                if trajs is None:
                    trajs = np.array([a.traj for a in agents])

                dir_path = f'{self.file_manager.get_temp_file_path(clip)}.{key}'
                dir_name = os.path.basename(dir_path)
                value = mgr.run(clip=clip,
                                root_dir=dir_name,
                                agents=agents,
                                trajs=trajs)
                self.metadata[mode].add_data(key, value,
                                             mgr.is_dataset_wise_input,
                                             data_length=len(trajs))

        self.processed_clips[mode] += clips

        # Making into the dataset object
        p = f'Prepare {mode}' + ' {}...'
        [self._gather(T, mode, p) for T in self.model_inputs]
        [self._gather(T, mode, p) for T in self.model_labels]
        dataset = DataLoader(dataset=self.metadata[mode],
                             batch_size=self.args.batch_size,
                             drop_last=True if training else False,
                             shuffle=True if training else False)
        return dataset

    def _gather(self, type_name: str, mode: str, desc_pattern: str):
        """
        Gather needed inputs from `self.agents` and `self.ext_inputs`
        and stake them into a `torch.Tensor` tensor for training or test.
        """
        match type_name:
            case INPUT_TYPES.OBSERVED_TRAJ:
                name, string = ['traj', 'trajectories']
            case INPUT_TYPES.NEIGHBOR_TRAJ:
                name, string = ['traj_neighbor', 'neighbors']
            case INPUT_TYPES.DESTINATION_TRAJ:
                name, string = ['destination', 'destinations']
            case INPUT_TYPES.GROUNDTRUTH_TRAJ:
                name, string = ['groundtruth', 'groundtruth']
            case INPUT_TYPES.NEIGHBOR_GROUNDTRUTH_TRAJ:
                name, string = ['groundtruth_neighbor', "neighbors' groundtruth"]
            case _:
                return

        value = get_attributes(self.agents, name, desc_pattern.format(string))
        self.metadata[mode].add_data(type_name, value)

    def print_info(self, **kwargs):
        t_info: dict = {'Dataset name': self.split_manager.dataset_name,
                        'Dataset annotation type': self.split_manager.anntype,
                        'Split name': self.split_manager.split}

        for mode in ['train', 'test']:
            if len(t := self.processed_clips[mode]):
                t_info[f'Clips to {mode}'] = t

        return super().print_info(**t_info, **kwargs)
