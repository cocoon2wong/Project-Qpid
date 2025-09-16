"""
@Author: Conghao Wong
@Date: 2025-09-16 19:44:09
@LastEditors: Conghao Wong
@LastEditTime: 2025-09-16 21:20:32
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import os
from typing import Any, TypeVar

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from qpid.args import Args
from qpid.constant import ANN_TYPES
from qpid.utils import (INIT_POSITION, ROOT_TEMP_DIR, dir_check,
                        get_relative_path)

from ..settings import DISTRIBUTION_TEMP_NAME, DISTRIBUTION_TEMP_PATH
from .__base import BaseCanvasManager, BaseVisHelper

T = TypeVar('T', np.ndarray, Figure)

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


class Normal2DCanvas(BaseCanvasManager):
    def __init__(self, args: Args | None = None,
                 manager: Any = None,
                 name: str | None = 'Visualization Helper'):

        super().__init__(args, manager, name)

        anntype = self.args.anntype

        # Get basic vis helper
        if anntype == ANN_TYPES.CO_2D:
            h = CoordinateHelper
        elif anntype == ANN_TYPES.BB_2D:
            h = BoundingboxHelper
        else:
            raise NotImplementedError(anntype)

        self.helper = h(self.vis_args)

        # Read png files
        self.obs_file = cv2.imread(OBS_IMAGE, -1)
        self.neighbor_file = cv2.imread(NEI_OBS_IMAGE, -1)
        self.current_file = cv2.imread(CURRENT_IMAGE, -1)
        self.pred_file = cv2.imread(PRED_IMAGE, -1)
        self.gt_file = cv2.imread(GT_IMAGE, -1)

    def init_canvas(self, init_image: Any | None = None, *args, **kwargs):
        if init_image is None:
            raise (ValueError)

        f = init_image
        if self.vis_args.draw_on_empty_canvas:
            f = 255 * np.ones_like(f)
        return f

    def text(self, source: np.ndarray,
             texts: list[str],
             x: int = 10,
             y: int = 40,
             font: int = cv2.FONT_HERSHEY_COMPLEX,
             size: float = 0.9,
             width: int = 2,
             line_height: int = 30,
             shadow_bias: int = 3,
             *args, **kwargs) -> np.ndarray:
        """
        Put text on one image
        """
        for index, text in enumerate(texts):
            source = cv2.putText(source, text,
                                 org=(x + shadow_bias, y + index *
                                      line_height + shadow_bias),
                                 fontFace=font,
                                 fontScale=size,
                                 color=(0, 0, 0),
                                 thickness=width)

            source = cv2.putText(source, text,
                                 org=(x, y + index * line_height),
                                 fontFace=font,
                                 fontScale=size,
                                 color=(255, 255, 255),
                                 thickness=width)

        return source

    def vis(self, source: np.ndarray,
            obs: np.ndarray | None = None,
            gt: np.ndarray | None = None,
            pred: np.ndarray | None = None,
            neighbor: np.ndarray | None = None,
            background: np.ndarray | None = None,
            pred_colors: np.ndarray | None = None,
            *args, **kwargs):
        """
        Draw one agent's observations, predictions, and groundtruths.

        :param source: The image file.
        :param obs: (optional) The observations in *pixel* scale.
        :param gt: (optional) The ground truth in *pixel* scale.
        :param pred: (optional) The predictions in *pixel* scale,\
            shape = `(K, steps, dim)`.
        :param neighbor: (optional) The observed neighbors' positions\
             in *pixel* scale, shape = `(batch, dim)`.
        :param draw_distribution: Controls whether to draw as a distribution.
        :param alpha: The alpha channel coefficient.
        """
        f = np.zeros([source.shape[0], source.shape[1], 4])

        # draw neighbors' observed trajectories
        if neighbor is not None:
            f = self.helper.draw_traj(f, neighbor[..., -1:, :],
                                      self.current_file,
                                      do_not_draw_lines=True)

            neighbor = neighbor if self.vis_args.draw_full_neighbors \
                else neighbor[..., -1:, :]
            for nei in neighbor:
                f = self.helper.draw_traj(f, nei, self.neighbor_file)

        # draw predicted trajectories
        if pred is not None:
            if self.vis_args.draw_distribution:
                alpha = 0.8 if not self.vis_args.draw_on_empty_canvas else 1.0
                f = self.helper.draw_dis(f, pred, alpha=alpha,
                                         steps=self.vis_args.distribution_steps)
            else:
                if pred_colors is None:
                    pred_colors = 255 * np.random.rand(pred.shape[0], 3)

                for (pred_k, color_k) in zip(pred, pred_colors):
                    f = self.helper.draw_traj(
                        f, pred_k, self.pred_file,
                        color=color_k)

        # draw observed and groundtruth trajectories
        if obs is not None:
            if obs.ndim == 2:
                obs = obs[np.newaxis]

            for _obs in obs:
                f = self.helper.draw_traj(f, _obs, self.obs_file)

        if gt is not None:
            if gt.ndim == 2:
                gt = gt[np.newaxis]

            for _gt in gt:
                f = self.helper.draw_traj(f, _gt, self.gt_file)

        # draw the background image
        if background is not None:
            f = ADD(background, f, [f.shape[1]//2, f.shape[0]//2])

        # add the original image
        f = ADD(source, f, [f.shape[1]//2, f.shape[0]//2])
        return f


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
