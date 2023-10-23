"""
@Author: Conghao Wong
@Date: 2022-09-29 09:49:43
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-23 18:17:25
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import qpid as __qpid

from .__args import VisArgs
from .__vis import Visualization

__qpid.register_new_args(VisArgs, 'Visualization Args', __package__)
__qpid._log_mod_loaded(__package__)
