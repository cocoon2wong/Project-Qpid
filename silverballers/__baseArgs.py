"""
@Author: Conghao Wong
@Date: 2023-09-06 18:46:53
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-30 09:52:42
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from qpid.args import DYNAMIC, STATIC, TEMPORARY, EmptyArgs


class SilverballersArgs(EmptyArgs):

    def __init__(self, terminal_args: list[str] | None = None,
                 is_temporary=False) -> None:

        super().__init__(terminal_args, is_temporary)

    @property
    def loads(self) -> str:
        """
        Paths to load a stacked model with series of subnetworks.
        Paths of all subnetworks' weights should be splited with `,`
        (it does not contain any blanks).
        """
        return self._arg('loads', 'null', argtype=TEMPORARY)

    @property
    def down_sampling_rate(self) -> float:
        """
        Down sampling rate to sample trajectories from
        all N = K*Kc trajectories.
        """
        return self._arg('down_sampling_rate', 1.0, argtype=TEMPORARY)

    @property
    def pick_trajectories(self) -> float:
        """
        Calculates the sum of the context map values of the predicted trajectories
        and picks the top n (percentage) best predictions. This parameter is only
        valid when the model's input contains `MAPS` and `MAP_PARAS`.
        """
        return self._arg('pick_trajectories', 1.0, argtype=TEMPORARY, short_name='p')

    @property
    def channel(self) -> int:
        """
        Specify the k-th channel of the model output.
        If `channel == -1`, it outputs all channels' predictions.
        """
        return self._arg('channel', -1, argtype=TEMPORARY, short_name='c')
