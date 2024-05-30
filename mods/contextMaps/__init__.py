"""
@Author: Conghao Wong
@Date: 2022-11-10 09:26:50
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-30 13:08:11
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import qpid
from qpid.dataset import AgentManager

from . import settings
from .__args import ContextMapsArgs
from .__mapParasManager import MapParasManager
from .__socialMapManager import SocialMapManager
from .__trajMapManager import TrajMapManager
from .__trajMapManager_seg import TrajMapManager_seg
from .layers import ContextEncoding
from .settings import INPUT_TYPES


def map_handler(agent_mgr: AgentManager):
    p = settings.POOLING_BEFORE_SAVING
    args = agent_mgr.args.register_subargs(ContextMapsArgs, name=__name__)

    agent_mgr.ext_mgrs.append(MapParasManager(agent_mgr))
    if not args.use_seg_maps:
        agent_mgr.ext_mgrs.append(TrajMapManager(agent_mgr, p))
    else:
        agent_mgr.ext_mgrs.append(TrajMapManager_seg(agent_mgr, p))
    agent_mgr.ext_mgrs.append(SocialMapManager(agent_mgr, p))


qpid.register_args(ContextMapsArgs, 'Context Maps Args')
qpid.register_input_type(INPUT_TYPES.MAP, handler=map_handler)
qpid.register_input_type(INPUT_TYPES.MAP_PARAS)
