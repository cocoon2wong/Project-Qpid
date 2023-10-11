"""
@Author: Conghao Wong
@Date: 2022-09-01 10:38:49
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-11 10:35:26
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from ...__root import BaseObject
from ...dataset import Annotation


class BaseProcessLayer(torch.nn.Module, BaseObject):

    def __init__(self, anntype: str,
                 preprocess_input_types: list[str],
                 postprocess_input_types: list[str]):

        torch.nn.Module.__init__(self)
        BaseObject.__init__(self, f'{type(self).__name__}({hex(id(self))})')

        self.anntype = anntype
        self.preprocess_input_types = preprocess_input_types
        self.postprocess_input_types = postprocess_input_types

        self.picker = Annotation(anntype) if anntype else None

    def forward(self, inputs: dict[str, torch.Tensor],
                preprocess: bool,
                update_paras=False,
                training=None, *args, **kwargs) -> dict[str, torch.Tensor]:
        """
        Run preprocess or postprocess on the input dictionary.
        """
        if preprocess:
            if update_paras:
                self.update_paras(inputs)
            outputs = self.preprocess(inputs)
        else:
            outputs = self.postprocess(inputs)

        return outputs

    def preprocess(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError('Please rewrite this method')

    def postprocess(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError('Please rewrite this method')

    def update_paras(self, inputs: dict[str, torch.Tensor]) -> None:
        raise NotImplementedError('Please rewrite this method')
