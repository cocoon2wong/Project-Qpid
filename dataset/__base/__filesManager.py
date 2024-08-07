"""
@Author: Conghao Wong
@Date: 2023-06-12 18:44:58
@LastEditors: Conghao Wong
@LastEditTime: 2024-08-06 10:20:20
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import os

import numpy as np

from ..__splitManager import Clip
from .__inputManager import BaseInputManager, BaseManager
from .__inputObject import BaseInputObject
from .__inputObjectManager import BaseInputObjectManager


class BaseFilesManager(BaseInputManager):
    """
    BaseFilesManager
    ---
    A manager to save processed dataset files (types are `BaseInputObject`).

    - Load items: A list of `BaseInputObject` to save;
    - Run items: Load files and save them into `npz` files.
        If the saved file exists, it will load these files and make
        the corresponding `BaseInputObject` objects.
    """

    FILE_PREFIX: str = ''
    DATA_MGR: type[BaseInputObjectManager] = BaseInputObjectManager
    DATA_TYPE: type[BaseInputObject] = BaseInputObject

    def __init__(self, manager: BaseManager,
                 name='Agent Files Manager'):

        super().__init__(manager, name)
        self.data_manager = self.DATA_MGR(self)

    def get_temp_file_path(self, clip: Clip) -> str:
        base_dir = clip.temp_dir
        f_name = '{}_{}to{}_interval{}_samplestep{}.npz'
        f_name = f_name.format(self.FILE_PREFIX,
                               self.args.obs_frames,
                               self.args.pred_frames,
                               self.args.interval,
                               self.args.step)
        return os.path.join(base_dir, f_name)

    # For type hinting
    def run(self, clip: Clip, agents: list[BaseInputObject] | None = None,
            *args, **kwargs) -> list[BaseInputObject]:

        return super().run(clip=clip, agents=agents, *args, **kwargs)

    def save(self, *args, **kwargs) -> None:
        save_dict = {}
        agents = self.data_manager.run(self.working_clip)
        for index, agent in enumerate(agents):
            save_dict[str(index)] = agent.zip_data()

        if self.temp_file is None:
            raise ValueError

        np.savez(self.temp_file, **save_dict)

    def load(self, *args, **kwargs) -> list[BaseInputObject]:
        if self.temp_file is None:
            raise ValueError

        saved: dict = np.load(self.temp_file, allow_pickle=True)

        if not len(saved):
            self.log(f'Please delete file `{self.temp_file}` and re-run the program.',
                     level='error', raiseError=FileNotFoundError)

        if (v := saved['0'].tolist()['__version__']) < (
                v1 := self.DATA_TYPE.__version__):
            self.log((f'Saved {self.FILE_PREFIX} managers\' version is {v}, ' +
                      f'which is lower than the required {v1}. Please delete' +
                      ' them (you can directly delete the `temp_files` dir)' +
                      ' and re-run this program, or there could' +
                      ' happen something wrong.'),
                     level='warning')

        return [self.DATA_TYPE().load_data(v.tolist()) for v in saved.values()]
