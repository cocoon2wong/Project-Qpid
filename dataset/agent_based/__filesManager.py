"""
@Author: Conghao Wong
@Date: 2023-05-19 16:05:54
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-02 15:16:03
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from ...base import BaseManager
from ..__base import BaseFilesManager
from .__inputObject import Agent
from .__inputObjectManager import AgentObjectManager


class AgentFilesManager(BaseFilesManager):
    """
    AgentFilesManager
    ---
    A manager to save processed trajectory `npz` files.
    It is managed by the `AgentManager` object.

    Members
    ---
    - `AgentObjectManager`: Manage to make and save `Agent` objects.

    Others
    ---
    - Load items: A list of agents (type is `list[Agent]`) to save;
    - Run items: Load agents and save them into `npz` files.
        If the saved file exists, it will load these files into agents.
    """

    FILE_PREFIX = 'agent'
    DATA_MGR = AgentObjectManager
    DATA_TYPE = Agent

    def __init__(self, manager: BaseManager,
                 name='Agent Files Manager'):

        super().__init__(manager, name)
