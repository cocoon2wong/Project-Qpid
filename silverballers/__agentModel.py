"""
@Author: Conghao Wong
@Date: 2023-09-06 18:49:17
@LastEditors: Conghao Wong
@LastEditTime: 2024-01-15 19:40:16
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from typing import Union

from ..args import DYNAMIC, STATIC, TEMPORARY
from ..constant import INPUT_TYPES
from ..training import loss
from .__baseArgs import BaseSilverballersArgs
from .__baseSubnetwork import BaseSubnetwork, BaseSubnetworkStructure
from .__loss import avgKey, keyl2


class AgentArgs(BaseSilverballersArgs):

    def __init__(self, terminal_args: list[str] | None = None,
                 is_temporary=False) -> None:

        super().__init__(terminal_args, is_temporary)

    @property
    def loss(self) -> str:
        """
        Loss used to train agent models.
        Canbe `'avgkey'` or `'keyl2'` (default).
        """
        return self._arg('loss', 'keyl2', argtype=DYNAMIC)


class BaseAgentModel(BaseSubnetwork):

    def __init__(self, Args: AgentArgs,
                 as_single_model: bool = True,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # Type hinting
        self.args: AgentArgs
        self.structure: BaseAgentStructure

        # Model input and label types
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ)
        self.set_labels(INPUT_TYPES.GROUNDTRUTH_TRAJ)

    def print_info(self, **kwargs):
        info = {'Keypoints and Transforms': None,
                '- Transform type': self.args.T,
                '- Index of keypoints': self.key_indices_future,
                '- Index of past keypoints': self.key_indices_past}
        return super().print_info(**kwargs, **info)


class BaseAgentStructure(BaseSubnetworkStructure):

    ARG_TYPE: type[AgentArgs] = AgentArgs
    MODEL_TYPE: type[BaseAgentModel] | None = None

    def __init__(self, terminal_args: Union[list[str], AgentArgs],
                 manager=None):

        super().__init__(terminal_args, manager)

        # For type hinting
        self.args: AgentArgs
        self.model: BaseAgentModel

        # Losses and metrics
        if self.args.loss == 'keyl2':
            self.loss.set({keyl2: 1.0})
        elif self.args.loss == 'avgkey':
            self.loss.set({avgKey: 1.0})
        else:
            raise ValueError(self.args.loss)

        self.metrics.set({avgKey: 1.0,
                          loss.FDE: 0.0})

    def print_test_results(self, loss_dict: dict[str, float], **kwargs):
        super().print_test_results(loss_dict, **kwargs)
        s = f'python main.py --model MKII --loada {self.args.load} --loadb l'
        self.log(f'You can run `{s}` to start the silverballers evaluation.')
