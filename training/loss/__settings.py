"""
@Author: Conghao Wong
@Date: 2024-12-05 15:38:31
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-05 16:50:10
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

from typing import Callable

from ...args import Args
from ...constant import ANN_TYPES
from .__iou import AIOU, FIOU
from .__layers import ADE, FDE, l2


def default_metrics(args: Args):
    return {ADE: 1.0, FDE: 0.0}


def boundingbox_metric(args: Args):
    if args.anntype not in [ANN_TYPES.BB_2D,
                            ANN_TYPES.BB_3D]:
        return None

    return {ADE: 1.0,
            FDE: 0.0,
            AIOU: 0.0,
            FIOU: 0.0}


def nba_metrics(args: Args):
    if not ((args.dataset == 'NBA') and
            (args.obs_frames == 5) and
            (args.pred_frames == 10)):
        return None

    return [
        (ADE, (0.0, dict(points=5, name='ADE@2.0s'))),
        (FDE, (0.0, dict(index=4, name='FDE@2.0s'))),
        (ADE, (0.0, dict(points=10, name='ADE@4.0s'))),
        (FDE, (1.0, dict(index=9, name='FDE@4.0s'))),
    ]


def h36m_metrics(args: Args):
    if not ((args.dataset == 'Human3.6M') and
            (args.anntype in [ANN_TYPES.SKE_3D_17])):
        return None

    i = int(1000 * args.interval)  # Sample interval

    if args.pred_frames == 10:
        return [
            (FDE, (0.0, dict(index=1, name=f'FDE@{2*i}ms'))),
            (FDE, (0.0, dict(index=3, name=f'FDE@{4*i}ms'))),
            (FDE, (0.0, dict(index=7, name=f'FDE@{8*i}ms'))),
            (FDE, (1.0, dict(index=9, name=f'FDE@{10*i}ms'))),
        ]

    elif args.pred_frames == 25:
        return [
            (FDE, (0.0, dict(index=13, name=f'FDE@{14*i}ms'))),
            (FDE, (1.0, dict(index=24, name=f'FDE@{25*i}ms'))),
        ]


METRICS_RULES = [default_metrics,
                 boundingbox_metric,
                 h36m_metrics,
                 nba_metrics]


def add_metrics_rules(rule: Callable[[Args], None | list | dict]):
    METRICS_RULES.append(rule)


def default_loss(args: Args):
    return {l2: 1.0}


LOSS_RULES = [default_loss]


def add_loss_rules(rule: Callable[[Args], None | list | dict]):
    LOSS_RULES.append(rule)
