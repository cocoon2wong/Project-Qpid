"""
@Author: Conghao Wong
@Date: 2022-08-03 10:50:46
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-02 19:14:36
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..base import BaseManager
from ..constant import INPUT_TYPES
from ..utils import dir_check
from .__base import (Annotation, AnnotationManager, BaseExtInputManager,
                     BaseInputObject, get_attributes)
from .__splitManager import SplitManager
from .agent_based import AgentFilesManager
from .frame_based import FrameFilesManager


class TrajectoryDataset(Dataset):

    def __init__(self, inputs: list[torch.Tensor],
                 labels: list[torch.Tensor]) -> None:
        super().__init__()

        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, index):
        return (tuple([i[index] for i in self.inputs]),
                tuple([l[index] for l in self.labels]))

    def __len__(self):
        return len(self.inputs[0])


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
        self.model_inputs: list[str] = []
        self.model_labels: list[str] = []
        self.processed_clips: dict[str, list[str]] = {'train': [], 'test': []}

        # Managers for extra model inputs
        self.ext_mgrs: list[BaseExtInputManager] = []
        self.ext_types: list[str] = []
        self.ext_inputs: dict[str, dict[str, np.ndarray]] = {}

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

    def set_types(self, inputs_type: list[str], labels_type: list[str]):
        """
        Set the type of model inputs and outputs.
        Accept all types in `INPUT_TYPES`.
        """
        if (t := INPUT_TYPES.MAP) in inputs_type:
            from qpid.mods import contextMaps as maps
            p = maps.settings.POOLING_BEFORE_SAVING
            self.ext_types.append(t)
            self.ext_mgrs.append(maps.MapParasManager(self))
            self.ext_mgrs.append(maps.TrajMapManager(self, p))
            self.ext_mgrs.append(maps.TrajMapManager_seg(self, p))
            self.ext_mgrs.append(maps.SocialMapManager(self, p))

            if (t := INPUT_TYPES.MAP_PARAS) in inputs_type:
                self.ext_types.append(t)

        self.model_inputs = inputs_type
        self.model_labels = labels_type

    def clean(self):
        """
        Clean all loaded data and agent objects in this manager.
        """
        self.agents = []
        self.ext_inputs = {}
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
            random.shuffle(clips)

        mode = 'train' if training else 'test'

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
            self.ext_inputs[clip_name] = {}
            for mgr in self.ext_mgrs:
                key = mgr.INPUT_TYPE
                if key is None:
                    raise ValueError

                if trajs is None:
                    trajs = np.array([a.traj for a in agents])

                dir_path = f'{self.file_manager.get_temp_file_path(clip)}.{key}'
                dir_name = dir_check(dir_path).split('/')[-1]
                value = mgr.run(clip=clip,
                                root_dir=dir_name,
                                agents=agents,
                                trajs=trajs)

                if not key in self.ext_inputs[clip_name].keys():
                    self.ext_inputs[clip_name][key] = value
                else:
                    self.ext_inputs[clip_name][key] += value

        self.processed_clips[mode] += clips

        # Making into the dataset object
        p = f'Prepare {mode}' + ' {}...'
        x = [self._gather(T, p) for T in self.model_inputs]
        y = [self._gather(T, p) for T in self.model_labels]
        dataset = DataLoader(dataset=TrajectoryDataset(x, y),
                             batch_size=self.args.batch_size,
                             drop_last=True if training else False,
                             shuffle=True if training else False)
        return dataset

    def _gather(self, type_name: str,
                tqdm_desc_pattern: str) -> torch.Tensor:
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
            case _:
                if (t := type_name) in self.ext_types:
                    res = None
                    for _res in self.ext_inputs.values():
                        if res is None:
                            res = _res[t]
                        else:
                            res = np.concatenate([res, _res[t]], axis=0)
                    return torch.from_numpy(res)
                else:
                    raise ValueError(t)

        return get_attributes(self.agents, name,
                              tqdm_desc_pattern.format(string))

    def print_info(self, **kwargs):
        t_info = {}
        for mode in ['train', 'test']:
            if len(t := self.processed_clips[mode]):
                t_info.update({'T' + f'{mode} agents come from'[1:]: t})

        return super().print_info(**t_info, **kwargs)
