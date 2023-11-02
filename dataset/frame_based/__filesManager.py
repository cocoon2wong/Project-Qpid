"""
@Author: Conghao Wong
@Date: 2023-06-12 14:44:44
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-02 15:25:04
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from ...base import BaseManager
from ..__base import BaseFilesManager
from .__inputObject import Frame
from .__inputObjectManager import FrameObjectManager


class FrameFilesManager(BaseFilesManager):
    """
    FrameFilesManager
    ---
    A manager to save processed trajectory `npz` files.
    It is managed by the `AgentManager` object.

    Members
    ---
    - `FrameObjectManager`: Manage to make and save `Frame` objects.

    Others
    ---
    - Load items: A list of frame agents (type is `list[Frame]`) to save;
    - Run items: Load agents and save them into `npz` files.
        If the saved file exists, it will load these files into frame agents.
    """

    FILE_PREFIX = 'frame'
    DATA_MGR = FrameObjectManager
    DATA_TYPE = Frame

    def __init__(self, manager: BaseManager,
                 name='Frame Files Manager'):

        super().__init__(manager, name)
