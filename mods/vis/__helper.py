"""
@Author: Conghao Wong
@Date: 2022-09-29 09:53:58
@LastEditors: Conghao Wong
@LastEditTime: 2024-11-11 13:53:44
@Description: png content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

import cv2
import numpy as np

from ...constant import ANN_TYPES
from ...utils import INIT_POSITION, ROOT_TEMP_DIR, dir_check
from .settings import (DISTRIBUTION_TEMP_NAME, DISTRIBUTION_TEMP_PATH,
                       IF_DRAW_LINES)


class BaseVisHelper():
    def __init__(self):
        self.draw_lines = IF_DRAW_LINES

    def draw_single(self, source: np.ndarray,
                    inputs: np.ndarray,
                    png: np.ndarray):
        """
        Draw a single trajectory point on the source image.

        :param source: The background image.
        :param inputs: A single observation, shape = (dim).
        :param png: The target png image to put on the image.
        """
        raise NotImplementedError

    def draw_traj(self, source: np.ndarray,
                  inputs: np.ndarray,
                  png: np.ndarray,
                  color=(255, 255, 255),
                  width=3,
                  draw_lines: bool | int = True):
        """
        Draw one trajectory on the source image.

        :param source: The background image.
        :param inputs: A single observation, shape = (steps, dim).
        :param png: The target png image to put on the image.
        """
        raise NotImplementedError

    def draw_dis(self, source: np.ndarray,
                 inputs: np.ndarray,
                 alpha: float,
                 steps: str = 'all',
                 plt_mode: bool = False):
        """
        Draw model predicted trajectories in the distribution way.

        :param source: The background image.
        :param inputs: Model predictions, shape = ((K), steps, dim).
        :param alpha: Transparency (from 0 to 1).
        :param steps: Indices of the predicted steps to be visualized. \
            It accepts a string that contains several integers split by `_`. \
            For example, `'0_6_11'`.
        :param plt_mode: (Do not change it manually)
        """

        import seaborn as sns
        from matplotlib import pyplot as plt

        if steps != 'all':
            _indices = [int(ii) for ii in steps.split('_') if len(ii)]
            _indices = np.array(_indices)
            dat = inputs[..., _indices, :]
        else:
            dat = inputs

        dat = dat.reshape([-1, 2])   # inputs shape: ((K), steps, dim)

        if not plt_mode:
            h, w = source.shape[:2]
            plt.figure(figsize=(h/100, w/100), dpi=100)

        sns.kdeplot(x=dat[..., 0], y=dat[..., 1], fill=True)

        if plt_mode:
            return source

        plt.subplots_adjust(top=1, bottom=0, right=1,
                            left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.xlim(0, h)
        plt.ylim(0, w)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])

        dir_check(r := os.path.join(ROOT_TEMP_DIR, DISTRIBUTION_TEMP_PATH))
        plt.savefig(path := os.path.join(r, DISTRIBUTION_TEMP_NAME))
        plt.close()

        f_dis = cv2.imread(path)
        f_dis = np.transpose(f_dis, [1, 0, 2])[:, ::-1, :]
        f_dis_alpha = (255 * (np.sum(f_dis, axis=-1) < 255*3)).astype(int)
        f_dis_png = np.concatenate([f_dis, f_dis_alpha[..., None]], axis=-1)
        source = ADD(source, f_dis_png, alpha=alpha, type='auto')
        return source


class CoordinateHelper(BaseVisHelper):
    def __init__(self):
        super().__init__()

    def draw_single(self, source: np.ndarray,
                    inputs: np.ndarray,
                    png: np.ndarray):

        return ADD(source, png, (inputs[1], inputs[0]))

    def draw_traj(self, source: np.ndarray,
                  inputs: np.ndarray,
                  png: np.ndarray,
                  color: None | tuple[int, int, int] = None,
                  width=3,
                  draw_lines=True):

        if inputs.max() >= 0.5 * INIT_POSITION:
            return source

        steps, dim = inputs.shape[-2:]
        _color = (255, 255, 255) if color is None else color

        # draw lines
        if draw_lines and self.draw_lines and steps >= 2:
            _lines = np.zeros_like(source)
            traj = np.column_stack([inputs.T[1], inputs.T[0]])
            for left, right in zip(traj[:-1], traj[1:]):
                cv2.line(img=_lines,
                         pt1=(left[0], left[1]),
                         pt2=(right[0], right[1]),
                         color=_color, thickness=width)

            # Add the alpha channel
            _lines_alpha = 255 * (np.sum(_lines[..., :3], axis=-1) > 0)
            _lines[..., -1] = _lines_alpha

            # Add lines to the image
            source = ADD(source, _lines, type='auto')

        # recolor the marker
        if color is not None:
            if png.shape[-1] == 4:
                _c = np.concatenate([np.array(color), [255]], axis=-1)
            else:
                _c = np.array(color)

            png = (png.astype(float) *
                   _c[np.newaxis, np.newaxis] / 255).astype(np.uint8)

        # draw points
        if inputs.ndim > 2:
            inputs = np.reshape(inputs, [-1, dim])

        for input in inputs:
            source = self.draw_single(source, input, png)

        return source


class BoundingboxHelper(BaseVisHelper):
    def __init__(self):
        super().__init__()

    def draw_single(self, source: np.ndarray,
                    inputs: np.ndarray,
                    png: np.ndarray,
                    color=(255, 255, 255),
                    width=3,
                    draw_center: bool | int = True):

        (y1, x1, y2, x2) = inputs[:4]
        color = tuple([int(c) for c in color])
        cv2.rectangle(img=source,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      color=color,
                      thickness=width)
        source[:, :, -1] = 255 * np.sign(source[:, :, 0])

        if draw_center:
            center = ((x1 + x2)//2, (y1 + y2)//2)
            source = ADD(source, png, center)

        return source

    def draw_traj(self, source: np.ndarray,
                  inputs: np.ndarray,
                  png: np.ndarray,
                  color=(255, 255, 255),
                  width=3,
                  draw_lines=True):

        if inputs.max() >= 0.5 * INIT_POSITION:
            return source

        if inputs.ndim == 3:
            inputs = np.reshape(inputs, [-1, inputs.shape[-1]])

        for box in inputs:
            source = self.draw_single(
                source, box, png, color, width, draw_center=draw_lines)
        return source


def get_helper(anntype: str) -> BaseVisHelper:
    if anntype == ANN_TYPES.CO_2D:
        h = CoordinateHelper
    elif anntype == ANN_TYPES.BB_2D:
        h = BoundingboxHelper
    else:
        raise NotImplementedError(anntype)

    return h()


def ADD(source: np.ndarray,
        png: np.ndarray,
        position: np.ndarray | list | tuple | None = None,
        alpha=1.0,
        type=None):
    """
    Add a png to the source image

    :param source: The source image, shape = `(H, W, 3)` or `(H, W, 4)`.
    :param png: The png image, shape = `(H, W, 3)` or `(H, W, 4)`.
    :param position: The pixel-level position in the source image, shape = `(2)`.
    :param alpha: Transparency.
    """
    if position is None:
        if type == 'auto':
            position = [source.shape[1]//2, source.shape[0]//2]
        else:
            raise ValueError

    yc, xc = position
    xp, yp, _ = png.shape
    xs, ys, _ = source.shape
    x0, y0 = [xc-xp//2, yc-yp//2]

    if png.shape[-1] == 4:
        png_mask = png[:, :, 3:4]/255
        png_file = png[:, :, :3]
    else:
        png_mask = np.ones_like(png[:, :, :1])
        png_file = png

    if x0 >= 0 and y0 >= 0 and x0 + xp <= xs and y0 + yp <= ys:
        source[x0:x0+xp, y0:y0+yp, :3] = \
            (1.0 - alpha * png_mask) * source[x0:x0+xp, y0:y0+yp, :3] + \
            alpha * png_mask * png_file

        if source.shape[-1] == 4:
            source[x0:x0+xp, y0:y0+yp, 3:4] = \
                np.minimum(source[x0:x0+xp, y0:y0+yp, 3:4] +
                           255 * alpha * png_mask, 255)
    return source
