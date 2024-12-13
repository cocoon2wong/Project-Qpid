"""
@Author: Conghao Wong
@Date: 2022-10-12 11:13:46
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-13 10:27:42
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from typing import Any, Callable, overload

import numpy as np
import torch

from ...args import Args
from ...base import BaseManager
from ...constant import INPUT_TYPES
from ...dataset import Annotation, AnnotationManager
from ...model import Model
from ...utils import decode_string, get_loss_mask
from .__layers import BaseLossLayer
from .__settings import LOSS_RULES, METRICS_RULES


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

            layer = k(manager=self,
                      traj_coe=self.scale,
                      extra_parameters=parameters,
                      value_weight=weight,
                      name_postfix=f'({self.name})')

            self.layers.append(layer)

    @overload
    def compute(self, outputs: list[torch.Tensor],
                labels: list[torch.Tensor],
                inputs: list[torch.Tensor]) -> tuple[None, None]: ...

    @overload
    def compute(self, outputs: list[torch.Tensor],
                labels: list[torch.Tensor],
                inputs: list[torch.Tensor],
                training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...

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

        :return weighted_sum: The weighted sum of all loss functions. \
            It returns a `None` when `training == False`.
        :return loss_dict: A dict of values of all loss functions. \
            It returns a `None` when `training == False`.
        """

        # Mask valid agents
        obs = self.model.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        label = self.model.get_label(labels, INPUT_TYPES.GROUNDTRUTH_TRAJ)
        mask = get_loss_mask(obs, label)

        # Decode agent types (if needed)
        # TODO: Train models with different agent-type-related loss functions
        if not training and self.args.compute_metrics_with_types:
            types_code = self.model.get_input(inputs, INPUT_TYPES.AGENT_TYPES)
            types = np.array([decode_string(s.cpu().numpy())
                              for s in types_code])
            types_list = list(set(types))
        else:
            types = None
            types_list = None

        for layer in self.layers:
            # Compute losses or metrics on all classes (types of agents)
            if (types is None) or (types_list is None):
                layer.compute(outputs, labels, inputs, mask, training)

            # Compute losses or metrics on each class separately
            else:
                for _type in types_list:
                    indices = torch.from_numpy(np.where(types == _type)[0])
                    layer.compute(gather_type(outputs, indices),
                                  gather_type(labels, indices),
                                  gather_type(inputs, indices),
                                  mask[indices], training, type=_type)

        if training:
            # Compute average loss
            value_dict, _, weighted_sum = self.pack_values()
            return weighted_sum, value_dict

        else:
            return None, None

    def pack_values(self) -> tuple[dict[str, torch.Tensor],
                                   dict[str, list],
                                   torch.Tensor]:
        """
        Gather computation results from all loss/metrics layers

        :return value_dict: All loss values in a dictionary. Keys are their names.
        :return info_dict: A dictionary to save all loss-related information.\
            Each value is a list with 3 members: `[count, weight, has_unit]`.\
            `count` is the number of agents that loss have computed on (it is a\
            `torch.Tensor` object with type `torch.float32`); \
            `weight` is the weight of the final loss value, and it is mainly \
            used during training; and `has_unit` indicates whether this loss \
            or metric is absolute and has a unit like `meters`.
        :return weighted_sum: The weighted sum loss (mainly used during training).  
        """
        value_dict: dict[str, torch.Tensor] = {}
        info_dict: dict[str, list] = {}
        for layer in self.layers:
            for _type in layer.value.keys():
                _key = layer.name + f'({_type})' if len(_type) else layer.name

                _loss = torch.stack(layer.value[_type])
                _weights = torch.tensor(layer.batch_size[_type],
                                        dtype=torch.float32).to(_loss.device)
                _loss = (torch.sum(_loss * _weights) /
                         (_count := torch.sum(_weights)))

                value_dict.update({_key: _loss})
                info_dict.update({_key: [_count,
                                         layer.loss_weight,
                                         layer.has_unit]})

        # Weighted-sum all single losses
        weighted_sum = torch.tensor(0.0)
        count = 0
        for _key in value_dict.keys():
            _count = info_dict[_key][0] * info_dict[_key][1]
            weighted_sum = weighted_sum + value_dict[_key] * _count
            count = count + _count

        weighted_sum = weighted_sum/count
        return value_dict, info_dict, weighted_sum

    def clear_memory(self):
        for layer in self.layers:
            layer.clear_memory()

    def print_info(self, **kwargs):
        funcs = [type(f).__name__ for f in self.layers]
        return super().print_info(LossLayers=funcs,
                                  **kwargs)

    def set_default_metrics(self):
        self._set_rule(METRICS_RULES)

    def set_default_loss(self):
        self._set_rule(LOSS_RULES)

    def _set_rule(self, rules: list[Callable[[Args], Any]]):
        for rule in rules[::-1]:
            v = rule(self.args)
            if v is not None:
                self.set(v)
                return


def gather_type(inputs: list[torch.Tensor], indices: torch.Tensor):
    if not isinstance(inputs[0], torch.Tensor):
        return []
    else:
        return [_i[indices] for _i in inputs]
