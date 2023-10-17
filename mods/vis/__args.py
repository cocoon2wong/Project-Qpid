"""
@Author: Conghao Wong
@Date: 2023-10-17 09:52:24
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-17 15:57:20
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from ...args import EmptyArgs
from ...constant import ARG_TYPES

DYNAMIC = ARG_TYPES.DYNAMIC
STATIC = ARG_TYPES.STATIC
TEMPORARY = ARG_TYPES.TEMPORARY


class VisArgs(EmptyArgs):
    @property
    def draw_exclude_type(self) -> str:
        """
        Draw visualized results of agents except for user-assigned types.
        If the assigned types are `"Biker_Cart"` and the `draw_results`
        or `draw_videos` is not `"null"`, it will draw results of all
        types of agents except "Biker" and "Cart".
        It supports partial match, and it is case-sensitive.
        """
        return self._arg('draw_exclude_type', 'null', argtype=TEMPORARY)

    @property
    def draw_index(self) -> str:
        """
        Indexes of test agents to visualize.
        Numbers are split with `_`.
        For example, `'123_456_789'`.
        """
        return self._arg('draw_index', 'all', argtype=TEMPORARY)

    @property
    def draw_distribution(self) -> int:
        """
        Controls whether to draw distributions of predictions instead of points.
        If `draw_distribution == 0`, it will draw results as normal coordinates;
        If `draw_distribution == 1`, it will draw all results in the distribution
        way, and points from different time steps will be drawn with different colors.
        """
        return self._arg('draw_distribution', 0, argtype=TEMPORARY)

    @property
    def draw_extra_outputs(self) -> int:
        """
        Choose whether to draw (put text) extra model outputs
        on the visualized images.
        """
        return self._arg('draw_extra_outputs', 0, argtype=TEMPORARY)
