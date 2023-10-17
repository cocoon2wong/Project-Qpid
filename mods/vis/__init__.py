"""
@Author: Conghao Wong
@Date: 2022-09-29 09:49:43
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-17 11:13:39
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import qpid as __qpid

from .__args import VisArgs
from .__vis import Visualization

__qpid._log_mod_loaded(__package__)
__qpid.args.register_new_args(VisArgs._get_args_names(), __package__)
