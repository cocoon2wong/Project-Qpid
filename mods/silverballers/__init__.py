"""
@Author: Conghao Wong
@Date: 2022-06-20 22:09:51
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-30 11:01:18
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import qpid

from . import interpHandlers
from .__baseArgs import SilverballersArgs
from .__MKII_model import SilverballersMKII, SilverballersModel

# Register new args
qpid.register_args(SilverballersArgs, 'Silverballers Args')

# Register new models
qpid.register(
    MKII=[SilverballersMKII, SilverballersModel],
)
