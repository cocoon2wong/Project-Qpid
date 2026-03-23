"""
@Author: Conghao Wong
@Date: 2025-09-16 21:22:26
@LastEditors: Conghao Wong
@LastEditTime: 2026-03-23 18:09:38
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import random
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qpid.args.__args import Args

from .__base import BaseCanvasManager

# PLT Settings
PLT_CANVAS_TITLE = 'Visualized Predictions'
PLT_CANVAS_WIDTH = 30
PLT_CANVAS_HEIGHT = 10

# Skeleton Settings
IF_SHOW_AXES = False
LINE_WIDTH = 4
POINT_WIDTH = 8

LINES = [[0, 1],
         [1, 2],
         [2, 3],
         [0, 4],
         [4, 5],
         [5, 6],
         [0, 7],
         [7, 8],
         [8, 9],
         [9, 10],
         [8, 11],
         [11, 12],
         [12, 13],
         [8, 14],
         [14, 15],
         [15, 16]]


class PLT3DSkeleton17Canvas(BaseCanvasManager):

    def __init__(self, args: Args | None = None,
                 manager: Any = None,
                 name: str | None = 'Visualization Helper'):

        super().__init__(args, manager, name)
        self.line_colors = [get_random_colors() for _ in LINES]

        # Core canvas.
        self.__canvas: Figure | None = None

    def get_canvas(self):
        """
        Get the canvas object. It will raise a ValueError when this canvas
        object is `None`.
        """
        if self.__canvas is None:
            raise ValueError
        else:
            return self.__canvas

    def update_canvas(self, new_canvas: Figure | None):
        self.__canvas = new_canvas

    def init_canvas(self, init_image: Any | None = None, *args, **kwargs):
        plt.close(PLT_CANVAS_TITLE)
        f = plt.figure(PLT_CANVAS_TITLE,
                       figsize=(PLT_CANVAS_WIDTH, PLT_CANVAS_HEIGHT))

        f.subplots_adjust(left=0, right=1,
                          bottom=0, top=1,
                          hspace=0, wspace=0)

        self.update_canvas(f)

    def text(self, texts: list[str], *args, **kwargs):
        # self.get_canvas().suptitle(', '.join(texts))
        pass

    def vis(self, obs: np.ndarray | None = None,
            gt: np.ndarray | None = None,
            pred: np.ndarray | None = None,
            neighbor: np.ndarray | None = None,
            *args, **kwargs):

        # Canvas
        source = self.get_canvas()

        # Figures will be drawn in 3 rows, max(obs + pred) cols.
        r = 3
        c = max(self.args.obs_frames, self.args.pred_frames)

        # Shapes of both observations and ground truths should be `(*, 51)`.
        # Initialize colors for each edge.
        colors = self.line_colors

        # Draw observations (row 1).
        if obs is not None:
            for _i in range(self.args.obs_frames):
                ax = source.add_subplot(r, c, _i + 1, projection='3d')
                ax = draw_3D_skeleton(ax, obs[_i],
                                      point_config='o',
                                      point_color='cornflowerblue',
                                      line_colors=colors)

        # Draw ground truths (row 2).
        if gt is not None:
            for _i in range(self.args.pred_frames):
                ax = source.add_subplot(r, c, c + _i + 1, projection='3d')
                ax = draw_3D_skeleton(ax, gt[_i],
                                      point_config='s',
                                      point_color='lightgreen',
                                      line_colors=colors)

        # Draw predictions (row 3).
        # Only the first (stochastic) prediction will be drawn.
        if pred is not None:
            p = pred[0]
            for _i in range(self.args.pred_frames):
                ax = source.add_subplot(r, c, 2 * c + _i + 1, projection='3d')
                ax = draw_3D_skeleton(ax, p[_i],
                                      point_config='v',
                                      point_color='gold',
                                      line_colors=colors)


def draw_3D_skeleton(ax: Axes, dat: np.ndarray,
                     point_config: str,
                     point_color: str,
                     line_colors: list[str]) -> Axes:
    ax.clear()
    dat = dat.reshape([-1, 3])

    for [l1, l2], c in zip(LINES, line_colors):
        _p1 = dat[l1]
        _p2 = dat[l2]

        # Draw the line.
        ax.plot3D([_p1[0], _p2[0]],  # type: ignore
                  [_p1[1], _p2[1]],
                  [_p1[2], _p2[2]],
                  '-', color=c,
                  linewidth=LINE_WIDTH,
                  markersize=POINT_WIDTH)

        # Draw the points.
        ax.plot3D([_p1[0], _p2[0]],  # type: ignore
                  [_p1[1], _p2[1]],
                  [_p1[2], _p2[2]],
                  point_config,
                  color=point_color,
                  markersize=POINT_WIDTH)

    ax.axis('scaled')

    if not IF_SHOW_AXES:
        ax.axis('off')

    return ax


def get_random_colors():
    color_arr = ['1', '2', '3', '4', '5', '6', '7',
                 '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for _ in range(6):
        color += color_arr[random.randint(0, 14)]

    return "#" + color
