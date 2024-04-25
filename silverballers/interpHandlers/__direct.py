"""
@Author: Conghao Wong
@Date: 2024-04-25 16:50:58
@LastEditors: Conghao Wong
@LastEditTime: 2024-04-25 20:06:35
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

from qpid.constant import INPUT_TYPES
from qpid.model import Model

from .__baseInterpHandler import _BaseInterpHandlerModel


class DirectHandlerModel(_BaseInterpHandlerModel):

    def __init__(self, structure=None, *args, **kwargs):
        Model.__init__(self, structure, *args, **kwargs)

        self.args._set('model_name', 'Direct Output Model')

        # Handler models do not use any preprocess methods
        self.set_preprocess()

        # Accept all points from the 1-st stage model's outputs
        self.args._set('input_pred_steps',
                       '_'.join([str(i) for i in range(self.args.pred_frames)]))

        # Set model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.GROUNDTRUTH_TRAJ)

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        return self.get_input(inputs, INPUT_TYPES.GROUNDTRUTH_KEYPOINTS)
