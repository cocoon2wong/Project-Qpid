"""
@Author: Conghao Wong
@Date: 2025-06-17 10:01:09
@LastEditors: Conghao Wong
@LastEditTime: 2026-03-30 16:23:13
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from qpid.args import Args
from qpid.utils import INIT_POSITION

from .__base import BaseCanvasManager

PLT_CANVAS_TITLE = 'Visualized Predictions'


class PLT2DCanvas(BaseCanvasManager):

    def __init__(self, args: Args | None = None,
                 manager: Any = None,
                 name: str | None = 'Visualization Helper'):

        super().__init__(args, manager, name)

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
        # Scale the canvas.
        s = self.vis_args.text_scale
        if s == -1:
            s = 1.0

        s = 1.0 / max(s, 0.2)

        plt.close(PLT_CANVAS_TITLE)
        f = plt.figure(PLT_CANVAS_TITLE, figsize=(6.4 * s, 4.8 * s))

        self.update_canvas(f)

    def text(self, texts: list[str],
             x: int = -1,
             y: int = -1,
             size: int = 10,
             text_mode='title',
             *args, **kwargs):

        if text_mode == 'title':
            self.get_canvas().gca().set_title(', '.join(texts))
            return

        plt.text(x + 0.14, y + 0.14,
                 ' '.join(texts),
                 fontsize=size,
                 ha='center',
                 va='bottom',
                 color='black',
                 bbox=dict(
                     boxstyle='round,pad=0.3',
                     facecolor='gray',
                     alpha=0.5,
                     edgecolor='none'
                 ))

    def vis(self, obs: np.ndarray | None = None,
            gt: np.ndarray | None = None,
            pred: np.ndarray | None = None,
            neighbor: np.ndarray | None = None,
            pred_colors: np.ndarray | None = None,
            *args, **kwargs):
        """
        Visualize trajectories with matplotlib (2D coordinates only).
        """
        _p = '-' if self.vis_args.draw_lines else ''
        f = self.get_canvas().gca()

        # Draw neighbors' trajectories.
        if neighbor is not None:
            neighbor = neighbor[None] if neighbor.ndim == 2 else neighbor
            for nei in neighbor:
                f.plot(nei[-1, 0], nei[-1, 1], _p + 'o',
                       color='darkorange', markersize=13)

                f.plot(nei[:, 0], nei[:, 1], _p + 's', color='purple')

        # Draw observations.
        if obs is not None:
            obs = obs[None] if obs.ndim == 2 else obs
            for o in obs:
                f.plot(o[:, 0], o[:, 1], _p + 's', color='cornflowerblue')

        # Draw ground truths.
        if gt is not None:
            gt = gt[None] if gt.ndim == 2 else gt
            for g in gt:
                f.plot(g[:, 0], g[:, 1], _p + 's', color='lightgreen')

        # Draw predictions.
        if pred is not None:
            pred = pred[None] if pred.ndim == 2 else pred

            # Draw as the trajectory distribution.
            if False:
                # TODO: Implement distribution drawing
                pass

            # Draw as multiple trajectory points.
            else:
                if pred_colors is None:
                    pred_colors = 255 * np.random.rand(pred.shape[0], 3)

                for p, c in zip(pred, pred_colors):
                    if np.abs(p).max() < 0.5 * INIT_POSITION:
                        f.plot(p[:, 0], p[:, 1], _p + 's', color=c / 255)

        f.axis('equal')

    def vis_neighbor_IDs(self, neighbor: np.ndarray):
        for id, nei_pos in enumerate(neighbor[..., -1, :]):
            self.text([f'{id}'], x=nei_pos[0], y=nei_pos[1], text_mode='0')
