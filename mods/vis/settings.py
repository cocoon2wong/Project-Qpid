"""
@Author: Conghao Wong
@Date: 2023-10-17 11:20:36
@LastEditors: Conghao Wong
@LastEditTime: 2025-06-17 21:11:19
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from ...utils import get_relative_path

# Whether put text on images and videos
IF_PUT_TEXT_IN_VIDEOS = True
IF_PUT_TEXT_IN_IMAGES = True

# Distribution visualization settings
DISTRIBUTION_TEMP_PATH = 'vis'
DISTRIBUTION_TEMP_NAME = 'dis.png'

OBS_IMAGE = 'obs_small.png'
NEI_OBS_IMAGE = 'neighbor_small.png'
CURRENT_IMAGE = 'neighbor_current.png'
GT_IMAGE = 'gt_small.png'
PRED_IMAGE = 'pred_small.png'

OBS_IMAGE = get_relative_path(__file__, OBS_IMAGE)
NEI_OBS_IMAGE = get_relative_path(__file__, NEI_OBS_IMAGE)
CURRENT_IMAGE = get_relative_path(__file__, CURRENT_IMAGE)
GT_IMAGE = get_relative_path(__file__, GT_IMAGE)
PRED_IMAGE = get_relative_path(__file__, PRED_IMAGE)

DRAW_ON_VIDEO = 0
DRAW_ON_IMAGE = 1
DRAW_ON_PLTCANVAS = 2

PLT_CANVAS_TITLE = 'Visualized Predictions'
