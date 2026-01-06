"""
@Author: Conghao Wong
@Date: 2026-01-06 18:39:27
@LastEditors: Conghao Wong
@LastEditTime: 2026-01-06 19:21:08
@Github: https://cocoon2wong.github.io
@Copyright 2026 Conghao Wong, All Rights Reserved.
"""

import sys

from .cli import entrance

if __name__ == '__main__':
    entrance(sys.argv, train_or_test=True)
    