"""
@Author: Conghao Wong
@Date: 2025-06-17 10:01:09
@LastEditors: Conghao Wong
@LastEditTime: 2025-09-16 21:02:04
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .__base import BaseCanvasManager

PLT_CANVAS_TITLE = 'Visualized Predictions'


class PLT2DCanvas(BaseCanvasManager):

    def init_canvas(self, init_image: Any | None = None, *args, **kwargs):
        plt.close(PLT_CANVAS_TITLE)
        f = plt.figure(PLT_CANVAS_TITLE)
        return f

    def text(self, source: Figure,
             texts: list[str],
             *args, **kwargs):

        source.gca().set_title(', '.join(texts))
        return source

    def vis(self, source: Figure,
            obs: np.ndarray | None = None,
            gt: np.ndarray | None = None,
            pred: np.ndarray | None = None,
            neighbor: np.ndarray | None = None,
            *args, **kwargs):
        """
        Visualize trajectories with plt (2D coordinates only).
        """
        _p = '-' if self.vis_args.draw_lines else ''
        f = source.gca()

        # draw neighbors' trajectories
        if neighbor is not None:
            neighbor = neighbor[None] if neighbor.ndim == 2 else neighbor
            for nei in neighbor:
                f.plot(nei[-1, 0], nei[-1, 1], _p + 'o',
                       color='darkorange', markersize=13)

                f.plot(nei[:, 0], nei[:, 1], _p + 's', color='purple')

        # draw observations
        if obs is not None:
            obs = obs[None] if obs.ndim == 2 else obs
            for o in obs:
                f.plot(o[:, 0], o[:, 1], _p + 's', color='cornflowerblue')

        if gt is not None:
            gt = gt[None] if gt.ndim == 2 else gt
            for g in gt:
                f.plot(g[:, 0], g[:, 1], _p + 's', color='lightgreen')

        if pred is not None:
            pred = pred[None] if pred.ndim == 2 else pred

            # Draw as the trajectory distribution
            if False:
                # TODO
                pass

            # Draw as multiple trajectory points
            else:
                for p in pred:
                    f.plot(p[:, 0], p[:, 1], _p + 's')

        f.axis('equal')
        return source
