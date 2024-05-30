"""
@Author: Conghao Wong
@Date: 2022-09-29 09:49:43
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-30 11:22:03
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import qpid

from .__args import VisArgs
from .__vis import Visualization

qpid.register_args(VisArgs, 'Visualization Args')
