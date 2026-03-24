"""
@Author: Conghao Wong
@Date: 2026-03-24 16:29:51
@LastEditors: Conghao Wong
@LastEditTime: 2026-03-24 17:20:26
@Github: https://cocoon2wong.github.io
@Copyright 2026 Conghao Wong, All Rights Reserved.
"""

import numpy as np
from PIL import Image

OUTPUT_PATH = './neighbor_small.png'
DEFAULT_COLOR = [128, 128, 128, 255]
BORDER_COLOR = [255, 255, 255, 255]
DEFAULT_SIZE = [10, 10, 4]


def generate_marker():
    img_array = np.full(DEFAULT_SIZE,
                        DEFAULT_COLOR,
                        dtype=np.uint8)

    img_array[0, :] = BORDER_COLOR
    img_array[-1, :] = BORDER_COLOR

    img_array[:, 0] = BORDER_COLOR
    img_array[:, -1] = BORDER_COLOR

    img = Image.fromarray(img_array, 'RGBA')
    img.save(OUTPUT_PATH)

    print(f"File successfully generated: {OUTPUT_PATH}")


if __name__ == '__main__':
    generate_marker()
