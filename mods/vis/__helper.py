"""
@Author: Conghao Wong
@Date: 2022-09-29 09:53:58
@LastEditors: Conghao Wong
@LastEditTime: 2024-03-18 15:49:54
@Description: png content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import cv2
import numpy as np
import torch

from ...constant import ANN_TYPES
from ...utils import INIT_POSITION
from .settings import DISTRIBUTION_COLORBAR, IF_DRAW_LINES


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
                 png: np.ndarray,
                 alpha: float):
        """
        Draw model predicted trajectories in the distribution way.

        :param source: The background image.
        :param inputs: Model predictions, shape = (steps, (K), dim).
        """
        # reshape into (K, steps, dim)
        if inputs.ndim == 2:
            inputs = inputs[np.newaxis, :, :]

        steps, dim = inputs.shape[-2:]
        h, w = source.shape[:2]
        f_empty = np.zeros([h, w, 4])
        f_dis = f_empty

        for step in range(steps):
            step_input = inputs[:, step, :]  # (K, dim)

            f = np.zeros([h, w, 4])
            for _input in step_input:
                f = self.draw_single(f, _input, png)

            f_alpha = f[:, :, -1] ** 0.2
            f_alpha = (255 * f_alpha/f_alpha.max()).astype(np.int32)
            color_map = (step+1)/steps * DISTRIBUTION_COLORBAR[f_alpha]
            color_map = color_map.astype(np.int32)

            f_alpha = f_alpha[:, :, np.newaxis]
            f = np.concatenate([color_map, f_alpha], axis=-1)
            f_dis = ADD(f_dis, f, type='auto')

        # smooth distribution image
        f_dis = torch.conv2d(
            torch.from_numpy(
                np.transpose(f_dis.astype(np.float32),
                             [2, 0, 1])[:, np.newaxis]
            ),
            weight=torch.ones([1, 1, 20, 20])/(20**2),
            padding='same'
        ).numpy()
        f_dis = np.transpose(f_dis[:, 0], [1, 2, 0])

        source = ADD(source, f_dis, alpha=alpha, type='auto')
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
                  color=(255, 255, 255),
                  width=3,
                  draw_lines=True):

        if inputs.max() >= 0.5 * INIT_POSITION:
            return source

        steps, dim = inputs.shape[-2:]

        # draw lines
        if draw_lines and self.draw_lines and steps >= 2:
            traj = np.column_stack([inputs.T[1], inputs.T[0]])
            for left, right in zip(traj[:-1], traj[1:]):
                cv2.line(img=source,
                         pt1=(left[0], left[1]),
                         pt2=(right[0], right[1]),
                         color=color, thickness=width)
                source[:, :, -1] = 255 * np.sign(source[:, :, 0])

        # recolor the image
        if 255 not in color:
            if png.shape[-1] == 4:
                color = np.concatenate([np.array(color), [255]], axis=-1)
            png = (png.astype(float) *
                   color[np.newaxis, np.newaxis] / 255).astype(np.uint8)

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
