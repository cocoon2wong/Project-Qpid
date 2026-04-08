"""
@Author: Conghao Wong
@Date: 2023-10-17 09:52:24
@LastEditors: Conghao Wong
@LastEditTime: 2026-04-07 19:59:06
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
        Draw visualized results for all agents except those of user-assigned
        types. If the assigned types are `"Biker_Cart"` and `draw_results` or 
        `draw_videos` is not `"null"`, it draws results for all agent types
        except "Biker" and "Cart". It supports partial matching and is 
        case-sensitive.
        """
        return self._arg('draw_exclude_type', 'null', argtype=TEMPORARY)

    @property
    def draw_index(self) -> str:
        """
        Indices of test agents to visualize. Numbers are separated by `_`.
        For example, `'123_456_789'`.
        """
        return self._arg('draw_index', 'all', argtype=TEMPORARY)

    @property
    def draw_distribution(self) -> float:
        """
        Controls the bandwidth (smoothing) of the predicted trajectory
        distributions.

        - 0.0: Disable distribution drawing; draw as individual points.
        - > 0.0: Enable KDE distribution drawing, where this value acts as
          the `bw_adjust` parameter to control smoothing (e.g., 0.5).
        """
        return self._arg('draw_distribution', 0.0, argtype=TEMPORARY,
                         short_name='dd')

    @property
    def distribution_steps(self) -> str:
        """
        Controls which time step(s) to consider when visualizing the
        distribution of forecasted trajectories. Accepts one or more integers
        (starting from 0) separated by `'_'`. For example, `'4_8_11'`.
        Set to `'all'` to show the distribution of all predictions.
        """
        return self._arg('distribution_steps', 'all', argtype=TEMPORARY)

    @property
    def draw_on_empty_canvas(self) -> str:
        """
        Controls whether to draw on an empty (or a single-colored) canvas.
        Set to `'null'` to disable, or pass a 6-char RGB HEX string
        (e.g., `'EBEBEB'`).
        """
        return self._arg('draw_on_empty_canvas', 'null', argtype=TEMPORARY)

    @property
    def draw_extra_outputs(self) -> int:
        """
        (bool) Controls whether to draw (as text) extra model outputs on the
        visualized images.
        """
        return self._arg('draw_extra_outputs', 0, argtype=TEMPORARY)

    @property
    def draw_lines(self) -> int:
        """
        (bool) Controls whether to draw lines between consecutive 2D
        trajectory points.
        """
        return self._arg('draw_lines', 0, argtype=TEMPORARY)

    @property
    def draw_predictions(self) -> int:
        """
        (bool) Controls whether to draw prediction during visualization.
        """
        return self._arg('draw_predictions', 1, argtype=TEMPORARY)

    @property
    def draw_groundtruths(self) -> int:
        """
        (bool) Controls whether to draw ground-truth trajectories during
        visualization.
        """
        return self._arg('draw_groundtruths', 1, argtype=TEMPORARY)

    @property
    def draw_neighbor_IDs(self) -> int:
        """
        Controls whether to draw the index of neighbors during
        visualization. It accepts an integer value. Set it to `0` to disable
        this function. Set it to `1` to visualize all neighbors' IDs, while 
        set it to an integer larger that `1` will only display this limited
        number of neighbor IDs.
        """
        return self._arg('draw_neighbor_IDs', 0, argtype=TEMPORARY,
                         other_names=['draw_neighbor_ids'])

    @property
    def draw_full_neighbors(self) -> int:
        """
        (bool) Controls whether to draw the full observed trajectories of all
        neighbor agents, rather than only the last trajectory point at the
        current observation moment.
        """
        return self._arg('draw_full_neighbors', 0, argtype=TEMPORARY)

    @property
    def draw_with_plt(self) -> int:
        """
        (bool) Controls whether to use PLT (matplotlib) as the preferred
        method for visualizing trajectories on an empty canvas. If disabled,
        it attempts to visualize all points directly on the scene images.
        """
        return self._arg('draw_with_plt', 0, argtype=TEMPORARY)

    @property
    def text_scale(self) -> float:
        """
        A float value used to scale the legend (text, icons, etc.) during
        visualization. A larger value means text and icons occupy a larger
        relative proportion of the screen. Must be greater than `0.2`.
        """
        return self._arg('text_scale', -1.0, argtype=TEMPORARY)

    @property
    def pred_color_mode(self) -> int:
        """
        An integer indicating how stochastic predictions will be colored.
        It accepts the following values:

        - `0`: Assign a random color to each specific prediction of each
          agent to be forecasted (shaped with `t_f * dim`).
        - `1`: Assign the same random color to all predictions of a single
          agent.
        - `2`: Assign the same random color to the $k$th stochastic prediction
          of all agents to be forecasted.
        """
        return self._arg('pred_color_mode', 0, argtype=TEMPORARY)
