"""
@Author: Conghao Wong
@Date: 2023-11-07 16:47:51
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-08 10:59:19
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import qpid as __qpid

from .__args import SegMapArgs
from .__segMapManager import SegMapManager
from .__segMapParasManager import SegMapParasManager

__qpid.register_new_args(SegMapArgs, 'Segmentation Maps Args', __package__)
__qpid._log_mod_loaded(__package__)
