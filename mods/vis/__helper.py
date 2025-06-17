"""
@Author: Conghao Wong
@Date: 2022-09-29 09:53:58
@LastEditors: Conghao Wong
@LastEditTime: 2025-06-17 21:10:24
@Description: png content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
from typing import TypeVar

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ...constant import ANN_TYPES
from ...utils import INIT_POSITION, ROOT_TEMP_DIR, dir_check
from .__args import VisArgs
from .settings import DISTRIBUTION_TEMP_NAME, DISTRIBUTION_TEMP_PATH

T = TypeVar('T', np.ndarray, Figure)


class BaseVisHelper():
    def __init__(self, args: VisArgs):
        self.args = args

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
                  do_not_draw_lines: None | bool = None) -> np.ndarray:
        """
        Draw one trajectory on the source image.

        :param source: The background image.
        :param inputs: A single observation, shape = (steps, dim).
        :param png: The target png image to put on the image.
        """
        raise NotImplementedError

    def draw_dis(self, source: T,
                 pred: np.ndarray,
                 alpha: float,
                 *args, **kwargs) -> T:
        """
        Draw model predicted trajectories in the distribution way.

        :param source: The background image or the plt canvas.
        :param pred: Model predictions, shape = ((K), steps, dim).
        :param alpha: Transparency (from 0 to 1).
        """

        # Prepare model predictions
        if (steps := self.args.distribution_steps) != 'all':
            _indices = [int(ii) for ii in steps.split('_') if len(ii)]
            _indices = np.array(_indices)
            dat = pred[..., _indices, :]
        else:
            dat = pred

        dat = dat.reshape([-1, 2])   # inputs shape: ((K), steps, dim)

        # Prepare canvas
        if not isinstance(source, Figure):
            h, w = source.shape[:2]
            plt.figure(figsize=(h/100, w/100), dpi=100)

        # Draw distribution
        import seaborn as sns
        sns.kdeplot(x=dat[..., 0], y=dat[..., 1], fill=True)

        if isinstance(source, Figure):
            return source

        # Post process images
        # (by adding distributions to the original image)
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

    def draw_single(self, source: np.ndarray,
                    inputs: np.ndarray,
                    png: np.ndarray):

        return ADD(source, png, (inputs[1], inputs[0]))

    def draw_traj(self, source: np.ndarray,
                  inputs: np.ndarray,
                  png: np.ndarray,
                  color: None | tuple[int, int, int] = None,
                  width=3,
                  do_not_draw_lines: None | bool = None):

        if inputs.max() >= 0.5 * INIT_POSITION:
            return source

        steps, dim = inputs.shape[-2:]
        _color = (255, 255, 255) if color is None else color

        # draw lines
        if (self.args.draw_lines and steps >= 2) and (not do_not_draw_lines):
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
                  do_not_draw_lines: None | bool = None):

        if inputs.max() >= 0.5 * INIT_POSITION:
            return source

        if inputs.ndim == 3:
            inputs = np.reshape(inputs, [-1, inputs.shape[-1]])

        for box in inputs:
            source = self.draw_single(source, box, png, color, width,
                                      draw_center=self.args.draw_lines)
        return source


def get_helper(anntype: str, vis_args: VisArgs) -> BaseVisHelper:
    if anntype == ANN_TYPES.CO_2D:
        h = CoordinateHelper
    elif anntype == ANN_TYPES.BB_2D:
        h = BoundingboxHelper
    else:
        raise NotImplementedError(anntype)

    return h(vis_args)


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
