"""
@Author: Conghao Wong
@Date: 2023-11-07 16:47:51
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-30 12:24:57
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import qpid
from qpid.dataset import AgentManager

from .__segMapManager import SegMapManager
from .__segMapParasManager import SegMapParasManager
from .settings import INPUT_TYPES


def map_handler(agent_mgr: AgentManager):
    agent_mgr.ext_mgrs.append(SegMapManager(agent_mgr))


def map_paras_handler(agent_mgr: AgentManager):
    agent_mgr.ext_mgrs.append(SegMapParasManager(agent_mgr))


qpid.register_input_type(INPUT_TYPES.SEG_MAP, map_handler)
qpid.register_input_type(INPUT_TYPES.SEG_MAP_PARAS, map_paras_handler)
