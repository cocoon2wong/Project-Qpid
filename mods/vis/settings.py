"""
@Author: Conghao Wong
@Date: 2023-10-17 11:20:36
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-17 21:16:21
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import numpy as np

# Visualization configs
IF_USE_SMALL_POINTS = True

# Configs for drawing distribution
# Color bar in BGR format
# rgb(0, 0, 178) -> rgb(252, 0, 0) -> rgb(255, 255, 10)
DISTRIBUTION_COLORBAR = np.column_stack([
    np.interp(np.arange(256),
              np.array([0, 127, 255]),
              np.array([178, 0, 10])),
    np.interp(np.arange(256),
              np.array([0, 127, 255]),
              np.array([0, 0, 255])),
    np.interp(np.arange(256),
              np.array([0, 127, 255]),
              np.array([0, 252, 255])),
])

# Whether draw lines and text in images and videos
IF_DRAW_LINES = False
IF_PUT_TEXT_IN_VIDEOS = True
IF_PUT_TEXT_IN_IMAGES = True
