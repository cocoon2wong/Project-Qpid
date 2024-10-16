"""
@Author: Conghao Wong
@Date: 2022-10-12 11:13:46
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-16 19:27:19
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import torch

from ...base import BaseManager
from ...constant import INPUT_TYPES
from ...dataset import Annotation, AnnotationManager
from ...model import Model
from ...utils import decode_string, get_loss_mask
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

    @property
    def picker(self) -> Annotation:
        picker = self.manager.get_member(AnnotationManager)
        return picker.target

    @property
    def model(self) -> Model:
        return self.manager.model  # type: ignore

    def set(self, loss_dict: (
        dict[type[BaseLossLayer], float] |
        list[tuple[type[BaseLossLayer], tuple[float, dict]]]
    ), clear_before_setting: bool = True):
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
        :param clear_before_setting: Choose whether to clear original loss \
            layers before applying new layers.
        """
        if clear_before_setting:
            self.layers = []

        if type(loss_dict) is dict:
            items = loss_dict.items()
        elif isinstance(loss_dict, list | tuple):
            items = loss_dict
        else:
            raise TypeError(loss_dict)

        for k, vs in items:
            k: type[BaseLossLayer]

            if isinstance(vs, list | tuple):
                weight = vs[0]
                parameters = vs[1]
            else:
                weight = vs
                parameters = {}

            layer = k(coe=self.scale, manager=self)
            layer.loss_weight = weight
            layer.loss_paras = parameters
            self.layers.append(layer)

    def compute(self, outputs: list[torch.Tensor],
                labels: list[torch.Tensor],
                inputs: list[torch.Tensor],
                training=None):
        """
        Call all loss functions recorded in the `loss_list`.

        :param outputs: A list of the model's output tensors. \
            `outputs[0]` should be the predicted trajectories.
        :param labels: A list of groundtruth tensors.
        :param inputs: A list of model inputs.

        :return summary: The weighted sum of all loss functions.
        :return loss_dict: A dict of values of all loss functions.
        :return count_dict: A dict of weights of all loss functions.
        """

        loss_dict: dict[str, torch.Tensor] = {}
        count_dict: dict[str, int] = {}
        weight_dict: dict[str, float] = {}

        obs = self.model.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        label = self.model.get_label(labels, INPUT_TYPES.GROUNDTRUTH_TRAJ)
        mask = get_loss_mask(obs, label)

        if self.args.compute_metrics_with_types:
            types_code = self.model.get_input(inputs, INPUT_TYPES.AGENT_TYPES)
            types_decode = np.array([decode_string(s.cpu().numpy())
                                    for s in types_code])
            types_list = list(set(types_decode))

        for layer in self.layers:
            name = layer.name
            paras = layer.loss_paras
            weight = layer.loss_weight

            if len(paras):
                if 'name' in paras.keys():
                    name = paras['name']
                else:
                    name += f'@{paras}'

            # Compute losses or metrics on all classes
            if training or not self.args.compute_metrics_with_types:
                value = layer(outputs=outputs, labels=labels,
                              inputs=inputs, mask=mask,
                              training=training, **paras)

                loss_dict[f'{name}({self.name})'] = value
                count_dict[f'{name}({self.name})'] = len(outputs[0])
                weight_dict[f'{name}({self.name})'] = weight

            # Compute losses or metrics on each class separately
            else:
                for _type in types_list:
                    indices = torch.from_numpy(
                        np.where(types_decode == _type)[0])
                    _value = layer(outputs=gather_batch_inputs(outputs, indices),
                                   labels=gather_batch_inputs(labels, indices),
                                   inputs=gather_batch_inputs(inputs, indices),
                                   mask=mask[indices], training=training, **paras)

                    loss_dict[f'{name}({self.name}, {_type})'] = _value
                    count_dict[f'{name}({self.name}, {_type})'] = len(indices)
                    weight_dict[f'{name}({self.name}, {_type})'] = weight

        summary = torch.tensor(0.0)
        for loss_name, loss_value in loss_dict.items():
            summary = summary + loss_value * weight_dict[loss_name]

        if not training:
            loss_dict['__sum'] = summary
            count_dict['__sum'] = len(outputs[0])

        return summary, loss_dict, count_dict

    def print_info(self, **kwargs):
        funcs = [type(f).__name__ for f in self.layers]
        return super().print_info(LossLayers=funcs,
                                  **kwargs)


def gather_batch_inputs(inputs: list[torch.Tensor], indices: torch.Tensor):
    if not isinstance(inputs[0], torch.Tensor):
        return []
    else:
        return [_i[indices] for _i in inputs]
