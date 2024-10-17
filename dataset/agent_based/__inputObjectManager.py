"""
@Author: Conghao Wong
@Date: 2023-05-19 14:38:26
@LastEditors: Conghao Wong
@LastEditTime: 2024-03-22 09:42:31
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import numpy as np

from ...base import BaseManager, SecondaryBar
from ...utils import INIT_POSITION
from ..__base import BaseInputObjectManager
from .__inputObject import Agent, Trajectory


class AgentObjectManager(BaseInputObjectManager):
    """
    AgentObjectManager
    ---
    The basic class for managing `Agent` objects.
    It is used to load data from the meta dataset files, and make them into the
    zipped `npz` files in the `./temp_files` for training or test usages.
    It is managed by the `AgentFilesManager` object.
    """

    def __init__(self, manager: BaseManager,
                 name='Trajectory Manager'):

        super().__init__(manager, name)

    def load(self, **kwargs) -> list[Agent]:
        # load from saved files
        if self.temp_file is None:
            raise ValueError

        dat = np.load(self.temp_file, allow_pickle=True)

        matrix = dat['matrix']
        neighbor_indexes = dat['neighbor_indexes']
        frame_ids = dat['frame_ids']
        names_and_types = dat['person_ids']

        agent_count = matrix.shape[1]

        trajs = [Trajectory(agent_id=names_and_types[agent_index][0],
                            agent_type=names_and_types[agent_index][1],
                            trajectory=matrix[:, agent_index, :],
                            neighbors=neighbor_indexes,
                            frames=frame_ids,
                            init_position=INIT_POSITION,
                            ) for agent_index in range(agent_count)]

        sample_rate, frame_rate = self.working_clip.paras
        frame_step = int(self.args.interval / (sample_rate / frame_rate))
        train_samples = []

        # The timebar is open by the `AgentManager` object
        for agent_index in SecondaryBar(range(agent_count),
                                        manager=self.manager.manager,
                                        desc='Process dataset files...'):

            trajectory = trajs[agent_index]
            start_frame = trajectory.start_frame
            end_frame = trajectory.end_frame

            for p in range(start_frame, end_frame,
                           int(np.ceil(self.args.step * frame_step))):
                # Normal mode
                if self.args.pred_frames > 0:
                    if p + (self.args.obs_frames + self.args.pred_frames - 1) * frame_step >= end_frame:
                        break

                    obs = p + self.args.obs_frames * frame_step
                    end = p + (self.args.obs_frames +
                               self.args.pred_frames) * frame_step

                # Infinity mode, only works for destination models
                elif self.args.pred_frames == -1:
                    if p + (self.args.obs_frames + 1) * frame_step > end_frame:
                        break

                    obs = p + self.args.obs_frames * frame_step
                    end = end_frame

                else:
                    self.log('`pred_frames` should be a positive integer or -1, ' +
                             f'got `{self.args.pred_frames}`', level='error')
                    raise ValueError

                train_samples.append(trajectory.sample(start_frame=p,
                                                       obs_frame=obs,
                                                       end_frame=end,
                                                       matrix=matrix,
                                                       frame_step=frame_step,
                                                       max_neighbor=self.args.max_agents))

        return train_samples
