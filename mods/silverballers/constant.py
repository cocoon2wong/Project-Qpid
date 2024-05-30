"""
@Author: Conghao Wong
@Date: 2024-05-30 09:51:21
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-30 09:51:33
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""


class INTERPOLATION_TYPES():
    """
    Names of all interpolation methods.
    """

    LINEAR = 'l'
    LINEAR_SPEED = 'speed'
    LINEAR_ACC = 'acc'
    NEWTON = 'newton'

    @classmethod
    def get_type(cls, s: str):
        for _s in [cls.LINEAR, cls.LINEAR_ACC,
                   cls.LINEAR_SPEED, cls.NEWTON]:
            if s == _s:
                return _s
        return None
