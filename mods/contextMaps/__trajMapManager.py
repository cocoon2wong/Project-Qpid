"""
@Author: Conghao Wong
@Date: 2023-05-22 16:26:35
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-30 12:12:01
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from typing import Any

import cv2
import numpy as np

from qpid.base import BaseManager, SecondaryBar
from qpid.dataset import Clip
from qpid.dataset.__base import BaseExtInputManager
from qpid.dataset.agent_based import Agent
from qpid.utils import dir_check

from .__mapParasManager import MapParasManager
from .settings import INPUT_TYPES, MAP_HALF_SIZE
from .utils import add, cut, pooling2D


class TrajMapManager(BaseExtInputManager):
    """
    Trajectory Map Manager
    ---
    The trajectory map is a map that builds from all agents'
    observed trajectories. It indicates all possible walkable
    areas around the target agent. The value of the trajectory map
    is in the range `[0, 1]`. A higher value indicates that
    the area may not walkable.
    """

    TEMP_FILES: dict[str, str] = {'FILE': 'trajMap.npy',
                                  'GLOBAL_FILE': 'trajMap.png'}

    MAP_NAME = 'Trajectory Map'
    INPUT_TYPE = INPUT_TYPES.MAP

    def __init__(self, manager: BaseManager,
                 pool_maps=False,
                 name='Trajectory Map Manager'):

        super().__init__(manager, name)

        self.POOL = pool_maps

        # Configs
        self.HALF_SIZE = MAP_HALF_SIZE

        # Map variables
        self.global_map: np.ndarray | None = None

        if pool_maps:
            self.TEMP_FILES['FILE_WITH_POOLING'] = 'trajMap_pooling.npy'

    @property
    def map_mgr(self) -> MapParasManager:
        return self.manager.get_member(MapParasManager)

    def run(self, clip: Clip,
            trajs: np.ndarray,
            agents: list[Agent],
            *args, **kwargs):

        if self.map_mgr.use_seg_map:
            return 0

        return super().run(clip=clip, trajs=trajs, agents=agents,
                           *args, **kwargs)

    def save(self, trajs: np.ndarray,
             agents: list[Agent],
             *args, **kwargs) -> Any:

        # Build and save global trajectory map
        self.global_map = self.build_and_save_global_map(trajs)
        self.build_and_save_local_maps(agents)

    def load(self, *args, **kwargs):
        if not (temp_files := self.temp_files):
            raise ValueError(temp_files)

        # Load maps from the saved file
        if not self.POOL:
            f = self.temp_files['FILE']
        else:
            f = self.temp_files['FILE_WITH_POOLING']
        return 0.5 * np.load(f, allow_pickle=True)

    def build_and_save_global_map(self, trajs: np.ndarray,
                                  source: np.ndarray | None = None):
        """
        Build and save the global trajectory map.

        - Saved files: `GLOBAL_FILE`, `GLOBAL_CONFIG_FILE`.
        """
        if source is None:
            source = self.map_mgr.void_map
        else:
            source = source.copy()

        # Get 2D center points
        trajs = self.C(trajs)

        # build the global trajectory map
        source = add(source,
                     self.real2grid(trajs),
                     amplitude=[1],
                     radius=7)

        source = np.minimum(source, 30)
        source = 1 - source / np.max(source)

        # save global trajectory map
        cv2.imwrite(self.temp_files['GLOBAL_FILE'], 255 * source)
        return source

    def build_and_save_local_maps(self, agents: list[Agent],
                                  source: np.ndarray = None):
        """
        Build and save local trajectory maps for all agents.

        - Required files: `self.map` (`GLOBAL_FILE` and `GLOBAL_CONFIG_FILE`).
        - Saved file: `FILE`.
        """
        maps = []
        for agent in SecondaryBar(agents,
                                  manager=self.manager,
                                  desc=f'Building {self.MAP_NAME}...'):

            # Cut the local trajectory map from the global map
            # Center point: the last observed point
            center_real = self.C(agent.traj[-1:, :])
            center_pixel = self.real2grid(center_real)
            local_map = cut(self.global_map, center_pixel, self.HALF_SIZE)[0]
            maps.append(local_map)

        # Save maps
        dir_check(self.temp_dir)
        np.save(self.temp_files['FILE'], maps)

        # Pool the maps
        maps_pooling = pooling2D(np.array(maps))
        np.save(self.temp_files['FILE_WITH_POOLING'], maps_pooling)

    def real2grid(self, traj: np.ndarray) -> np.ndarray:
        return self.map_mgr.real2grid(traj)

    def C(self, trajs: np.ndarray) -> np.ndarray:
        """
        Get the 2D center point of the input M-dimensional trajectory.
        """
        return self.map_mgr.C(trajs)
