"""
@Author: Conghao Wong
@Date: 2025-09-16 19:44:09
@LastEditors: Conghao Wong
@LastEditTime: 2026-03-30 20:06:19
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import os
from typing import Any, TypeVar

import cv2
import numpy as np
from cv2.typing import MatLike
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
                    png: np.ndarray | MatLike | None):

        if png is None:
            return source

        return ADD(source, png, (inputs[1], inputs[0]))

    def draw_traj(self, source: np.ndarray,
                  inputs: np.ndarray,
                  png: np.ndarray | MatLike | None,
                  color: None | tuple[int, int, int] = None,
                  width=3,
                  do_not_draw_lines: None | bool = None):

        if inputs.max() >= 0.5 * INIT_POSITION:
            return source

        steps, dim = inputs.shape[-2:]
        _color = (255, 255, 255) if color is None else color

        # Draw lines.
        if (self.args.draw_lines and steps >= 2) and (not do_not_draw_lines):
            _lines = np.zeros_like(source)
            traj = np.column_stack([inputs.T[1], inputs.T[0]])
            for left, right in zip(traj[:-1], traj[1:]):
                cv2.line(img=_lines,
                         pt1=(left[0], left[1]),
                         pt2=(right[0], right[1]),
                         color=_color, thickness=width)

            # Add the alpha channel.
            _lines_alpha = 255 * (np.sum(_lines[..., :3], axis=-1) > 0)
            _lines[..., -1] = _lines_alpha

            # Add lines to the image.
            source = ADD(source, _lines, type='auto')

        # Recolor the marker.
        if color is not None and png is not None:
            if png.shape[-1] == 4:
                _c = np.concatenate([np.array(color), [255]], axis=-1)
            else:
                _c = np.array(color)

            png = (png.astype(float) *
                   _c[np.newaxis, np.newaxis] / 255).astype(np.uint8)

        # Draw points.
        if inputs.ndim > 2:
            inputs = np.reshape(inputs, [-1, dim])

        for input in inputs:
            source = self.draw_single(source, input, png)

        return source

    def draw_dis(self, source: T,
                 pred: np.ndarray,
                 alpha: float,
                 adjust: float = 1.0,
                 *args, **kwargs) -> T:
        """
        Draw forecasted trajectories as a spatial distribution.

        :param source: The background image or the matplotlib (plt) canvas.
        :param pred: Model predictions, shape = `((K), steps, dim)`.
        :param alpha: Transparency (from 0 to 1).
        :param adjust: Bandwidth adjustment factor for the KDE plot, 
            controlling the smoothness of the distribution (default is 1.0).
        """

        # Prepare model predictions.
        if (steps := self.args.distribution_steps) != 'all':
            _indices = [int(ii) for ii in steps.split('_') if len(ii)]
            _indices = np.array(_indices)
            dat = pred[..., _indices, :]
        else:
            dat = pred

        # Inputs shape: ((K), steps, dim)
        dat = dat.reshape([-1, 2])

        # Remove all illegal coordinates.
        dat_cond = np.abs(dat.astype(np.float32)).sum(-1) < 0.1 * INIT_POSITION
        dat = dat[np.where(dat_cond)[0]]

        # Prepare the canvas.
        if not isinstance(source, Figure):
            h, w = source.shape[:2]
            plt.figure(figsize=(h / 100, w / 100), dpi=100)

        # Draw the distribution.
        import seaborn as sns
        sns.kdeplot(x=dat[..., 0], y=dat[..., 1], fill=True, bw_adjust=adjust)

        if isinstance(source, Figure):
            return source

        # Post-process images by adding distributions to the original image.
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

        f_dis = np.array(cv2.imread(path))
        f_dis = np.transpose(f_dis, [1, 0, 2])[:, ::-1, :]
        f_dis_alpha = (255 * (np.sum(f_dis, axis=-1) < 255 * 3)).astype(int)
        f_dis_png = np.concatenate([f_dis, f_dis_alpha[..., None]], axis=-1)
        source = ADD(source, f_dis_png, alpha=alpha, type='auto')
        return source


class BoundingboxHelper(BaseVisHelper):

    def draw_single(self, source: np.ndarray,
                    inputs: np.ndarray,
                    png: np.ndarray | MatLike | None,
                    color=(255, 255, 255),
                    width=3,
                    draw_center: bool | int = True):

        if png is None:
            return source

        (y1, x1, y2, x2) = inputs[:4]
        color = tuple([int(c) for c in color])
        cv2.rectangle(img=source,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      color=color,
                      thickness=width)
        source[:, :, -1] = 255 * np.sign(source[:, :, 0])

        if draw_center:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            source = ADD(source, png, center)

        return source

    def draw_traj(self, source: np.ndarray,
                  inputs: np.ndarray,
                  png: np.ndarray | MatLike | None,
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

        # Get the basic visualization helper.
        if anntype == ANN_TYPES.CO_2D:
            h = CoordinateHelper
        elif anntype == ANN_TYPES.BB_2D:
            h = BoundingboxHelper
        else:
            raise NotImplementedError(anntype)

        self.helper = h(self.vis_args)

        # Read PNG files.
        self.obs_file = cv2.imread(OBS_IMAGE, -1)
        self.neighbor_file = cv2.imread(NEI_OBS_IMAGE, -1)
        self.current_file = cv2.imread(CURRENT_IMAGE, -1)
        self.pred_file = cv2.imread(PRED_IMAGE, -1)
        self.gt_file = cv2.imread(GT_IMAGE, -1)

        # Core canvas.
        self.__canvas: np.ndarray | None = None

    def get_canvas(self):
        """
        Get the canvas object. It will raise a ValueError when this canvas
        object is `None`.
        """
        if self.__canvas is None:
            raise ValueError
        else:
            return self.__canvas

    def update_canvas(self, new_canvas: np.ndarray | None):
        self.__canvas = new_canvas

    def init_canvas(self, init_image: Any | None = None, *args, **kwargs):
        if init_image is None:
            raise ValueError

        f = init_image
        c = self.vis_args.draw_on_empty_canvas

        if c != 'null':
            # Hardcode RGB string to BGR tuple.
            bgr = (int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16))

            f = np.empty_like(f)
            f[..., :3] = bgr
            if f.shape[-1] == 4:
                f[..., 3] = 255

        self.update_canvas(f)

    def text(self, texts: list[str],
             x: int = 22,
             y: int = 16,
             size: int = 18,
             line_height: int = 22,
             shadow_bias: int = 2,
             *args, **kwargs):
        """
        Overlay text using PIL with Matplotlib's built-in DejaVu Sans font.
        This avoids adding font files to the repository while ensuring 
        high-quality anti-aliasing across all platforms.
        """
        import os

        import matplotlib
        from PIL import Image, ImageDraw, ImageFont

        f = self.get_canvas()
        is_rgba = f.shape[-1] == 4

        # Determine text color and shadow visibility.
        bg = self.vis_args.draw_on_empty_canvas
        if bg == 'null':
            draw_shadow = True
            text_color = (255, 255, 255)
        else:
            draw_shadow = False
            # Calculate luminance to intelligently switch between black and white.
            try:
                r, g, b = int(bg[0:2], 16), int(bg[2:4], 16), int(bg[4:6], 16)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)
            except ValueError:
                text_color = (0, 0, 0)

        # Convert OpenCV array to PIL Image.
        f_uint8 = np.clip(f, 0, 255).astype(np.uint8)
        cv_color = cv2.COLOR_BGRA2RGBA if is_rgba else cv2.COLOR_BGR2RGB
        pil_img = Image.fromarray(cv2.cvtColor(f_uint8, cv_color))
        draw = ImageDraw.Draw(pil_img)

        # Dynamically locate Matplotlib's built-in font.
        try:
            mpl_data_dir = matplotlib.get_data_path()
            # DejaVuSans-Bold.ttf or DejaVuSans.ttf
            font_path = os.path.join(
                mpl_data_dir, 'fonts', 'ttf', 'DejaVuSans-Bold.ttf'
            )
            font = ImageFont.truetype(font_path, size)
        except Exception:
            font = ImageFont.load_default()

        # Draw texts.
        for index, text_line in enumerate(texts):
            current_y = y + index * line_height

            # Draw shadow only if drawing on the original video/image.
            if draw_shadow:
                draw.text((x + shadow_bias, current_y + shadow_bias),
                          text_line, font=font, fill=(0, 0, 0))

            # Draw main text.
            draw.text((x, current_y), text_line, font=font, fill=text_color)

        # Convert back to OpenCV format.
        cv_back = cv2.COLOR_RGBA2BGRA if is_rgba else cv2.COLOR_RGB2BGR
        f_out = cv2.cvtColor(np.array(pil_img), cv_back)

        self.update_canvas(f_out.astype(f.dtype))

    def vis(self, obs: np.ndarray | None = None,
            gt: np.ndarray | None = None,
            pred: np.ndarray | None = None,
            neighbor: np.ndarray | None = None,
            background: np.ndarray | None = None,
            pred_colors: np.ndarray | None = None,
            draw_neighbor_markers: bool = True,
            *args, **kwargs):
        """
        Draw one agent's observations, predictions, and ground-truths.

        :param obs: (optional) The observations in *pixel* scale.
        :param gt: (optional) The ground truth in *pixel* scale.
        :param pred: (optional) The predictions in *pixel* scale, 
            shape = `(K, steps, dim)`.
        :param neighbor: (optional) The observed neighbors' positions 
            in *pixel* scale, shape = `(batch, dim)`.
        :param draw_distribution: Controls whether to draw as a distribution.
        :param alpha: The alpha channel coefficient.
        """
        source = self.get_canvas()
        f = np.zeros([source.shape[0], source.shape[1], 4])

        # Draw neighbors' observed trajectories.
        if neighbor is not None:
            if draw_neighbor_markers:
                f = self.helper.draw_traj(f, neighbor[..., -1:, :],
                                          self.current_file,
                                          do_not_draw_lines=True)

            neighbor = neighbor if self.vis_args.draw_full_neighbors \
                else neighbor[..., -1:, :]
            for nei in neighbor:
                f = self.helper.draw_traj(f, nei, self.neighbor_file)

        # Draw predicted trajectories.
        if pred is not None:
            if self.vis_args.draw_distribution != 0:
                alpha = 0.8 if self.vis_args.draw_on_empty_canvas == 'null' else 0.9
                f = self.helper.draw_dis(f, pred, alpha=alpha,
                                         steps=self.vis_args.distribution_steps,
                                         adjust=self.vis_args.draw_distribution)
            else:
                if pred_colors is None:
                    pred_colors = 255 * np.random.rand(pred.shape[0], 3)

                for pred_k, color_k in zip(pred, pred_colors):
                    f = self.helper.draw_traj(
                        source=f,
                        inputs=pred_k,
                        png=self.pred_file,
                        color=color_k,
                    )

        # Draw observed and ground-truth trajectories.
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

        # Draw the background image.
        if background is not None:
            f = ADD(background, f, [f.shape[1] // 2, f.shape[0] // 2])

        # Add the original image.
        f = ADD(source, f, [f.shape[1] // 2, f.shape[0] // 2])

        self.update_canvas(f)

    def vis_neighbor_IDs(self, neighbor: np.ndarray, IDs: list[int]):
        for id, nei_pos in zip(IDs, neighbor[..., -1, :]):
            self.text([f'{id}'], x=nei_pos[1], y=nei_pos[0], size=22)


def ADD(source: np.ndarray,
        png: np.ndarray,
        position: np.ndarray | list | tuple | None = None,
        alpha=1.0,
        type=None):
    """
    Add a PNG to the source image.

    :param source: The source image, shape = `(H, W, 3)` or `(H, W, 4)`.
    :param png: The PNG image, shape = `(H, W, 3)` or `(H, W, 4)`.
    :param position: The pixel-level position in the source image, shape `(2)`.
    :param alpha: Transparency.
    """
    if position is None:
        if type == 'auto':
            position = [source.shape[1] // 2, source.shape[0] // 2]
        else:
            raise ValueError

    yc, xc = position
    xp, yp, _ = png.shape
    xs, ys, _ = source.shape
    x0, y0 = [xc - xp // 2, yc - yp // 2]

    if png.shape[-1] == 4:
        png_mask = png[:, :, 3:4] / 255
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
