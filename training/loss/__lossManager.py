"""
@Author: Conghao Wong
@Date: 2022-10-12 11:13:46
@LastEditors: Conghao Wong
@LastEditTime: 2024-03-20 17:10:32
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from ...base import BaseManager
from ...dataset import Annotation, AnnotationManager
from ...utils import get_loss_mask
from .__layers import BaseLossLayer


class LossManager(BaseManager):

    def __init__(self, manager: BaseManager,
                 trajectory_scale: float = 1.0,
                 name='Loss Manager'):
        """
        Init a `LossManager` object.

        :param manager: The manager object, usually a `Structure` object.
        :param trajectory_scale: The global scale coefficient for trajectories.
        :param name: The name of the manager, which could appear in all dict
            keys in the final output `loss_dict`.
        """

        super().__init__(manager=manager, name=name)

        self.scale = trajectory_scale
        self.layers: list[BaseLossLayer] = []
        self.loss_weights: list[float] = []
        self.loss_paras: list[dict] = []

    @property
    def picker(self) -> Annotation:
        picker = self.manager.get_member(AnnotationManager)
        return picker.target

    def set(self, loss_dict: (
        dict[type[BaseLossLayer], float] |
        list[tuple[type[BaseLossLayer], tuple[float, dict]]]
    )):
        """
        Set loss functions and their weights.

        :param loss_dict: A dict of loss functions, where all dict keys
            are the callable loss function, and the dict values are the
            weights of the corresponding loss function.
            Accept other parameters of the loss function from a `dict`.

            For example, you can set multiple metrics by
            ```
            self.metrics.set([
                [loss.FDE, [1.0, {'index': 1, 'name': 'FDE@40ms'}]],
                [loss.FDE, [1.0, {'index': 2, 'name': 'FDE@80ms'}]],
            ])
            ```

            NOTE: The callable loss function must have the `**kwargs` in
            their definitions.
        """
        self.layers = []
        self.loss_weights = []
        self.loss_paras = []

        if type(loss_dict) is dict:
            items = loss_dict.items()
        elif isinstance(loss_dict, list | tuple):
            items = loss_dict
        else:
            raise TypeError(loss_dict)

        for k, vs in items:
            k: type[BaseLossLayer]

            if isinstance(vs, list | tuple):
                weights = vs[0]
                parameters = vs[1]
            else:
                weights = vs
                parameters = {}

            self.layers.append(k(coe=self.scale, manager=self))
            self.loss_weights.append(weights)
            self.loss_paras.append(parameters)

    def compute(self, outputs: list[torch.Tensor],
                labels: list[torch.Tensor],
                inputs: list[torch.Tensor],
                training=None):
        """
        Call all loss functions recorded in the `loss_list`.

        :param outputs: A list of the model's output tensors. \
            `outputs[0]` should be the predicted trajectories.
        :param labels: A list of groundtruth tensors. \
            `labels[0]` should be the groundtruth trajectories.
        :param inputs: A list of model inputs. \
            `inputs[0]` should be the observed trajectories.

        :return summary: The weighted sum of all loss functions.
        :return loss_dict: A dict of values of all loss functions.
        """

        loss_dict: dict[str, torch.Tensor] = {}
        mask = get_loss_mask(inputs[0], labels[0])
        for layer, paras in zip(self.layers, self.loss_paras):
            name = layer.name
            if len(paras):
                if 'name' in paras.keys():
                    name = paras['name']
                else:
                    name += f'@{paras}'

            value = layer(outputs=outputs, labels=labels,
                          inputs=inputs, mask=mask,
                          training=training, **paras)

            loss_dict[f'{name}({self.name})'] = value

        if (l := len(self.loss_weights)):
            if l != len(loss_dict):
                raise ValueError('Incorrect loss weights!')
            weights = torch.tensor(self.loss_weights, dtype=torch.float32)

        else:
            weights = torch.ones(len(loss_dict))

        summary = torch.matmul(
            torch.unsqueeze(torch.stack(list(loss_dict.values())), 0),
            torch.unsqueeze(weights.to(mask.device), 1)
        )
        summary = torch.reshape(summary, ())
        return summary, loss_dict

    def print_info(self, **kwargs):
        funcs = [type(f).__name__ for f in self.layers]
        return super().print_info(LossLayers=funcs,
                                  Weights=self.loss_weights,
                                  LossParameters=self.loss_paras,
                                  **kwargs)
