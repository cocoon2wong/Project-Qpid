"""
@Author: Conghao Wong
@Date: 2025-09-16 19:28:59
@LastEditors: Conghao Wong
@LastEditTime: 2025-09-16 20:29:45
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

from typing import Any

import numpy as np

from qpid.args import Args
from qpid.base import BaseManager

from ..__args import VisArgs


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
        raise NotImplementedError   # <- Define this method in subclasses

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
        raise NotImplementedError   # <- Define this method in subclasses

    def draw_dis(self, source,
                 pred: np.ndarray,
                 alpha: float,
                 *args, **kwargs):
        """
        Draw model predicted trajectories in the distribution way.

        :param source: The background image or the plt canvas.
        :param pred: Model predictions, shape = ((K), steps, dim).
        :param alpha: Transparency (from 0 to 1).
        """
        raise NotImplementedError   # <- Define this method in subclasses


class BaseCanvasManager(BaseManager):
    def __init__(self, args: Args | None = None,
                 manager: Any = None,
                 name: str | None = 'Visualization Helper'):

        super().__init__(args, manager, name)

        self.vis_args = self.args.register_subargs(
            VisArgs, 'Visualization Args')
        
    def init_canvas(self, init_image: Any | None = None, *args, **kwargs):
        raise NotImplementedError   # <- Define this method in subclasses

    def text(self, source: Any, texts: list[str], *args, **kwargs):
        raise NotImplementedError   # <- Define this method in subclasses

    def vis(self, source: Any,
            obs: np.ndarray | None = None,
            gt: np.ndarray | None = None,
            pred: np.ndarray | None = None,
            neighbor: np.ndarray | None = None,
            *args, **kwargs):
        raise NotImplementedError   # <- Define this method in subclasses
