"""
@Author: Conghao Wong
@Date: 2022-06-20 16:14:03
@LastEditors: Conghao Wong
@LastEditTime: 2025-05-22 15:10:13
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import time

import numpy as np
import torch

from ..base import BaseManager
from ..constant import INPUT_TYPES, PROCESS_TYPES
from ..dataset import AgentManager, AnnotationManager
from ..utils import CHECKPOINT_FILENAME, WEIGHTS_FORMAT
from . import process

MAX_INFERENCE_TIME_STORGED = 100


class Model(torch.nn.Module, BaseManager):
    """
    Model (Model Manager)
    -----

    Usage
    -----
    When training or testing new models, please subclass this class, and clarify
    model layers used in your model.

    Public Methods
    --------------
    ```python
    # forward model with pre-process and post-process
    (method) forward: (self: Self@Model,
                       inputs: list[Tensor],
                       training: Any | None = None) -> list[Tensor]

    # Set model inputs
    (method) set_inputs: (self: Self@Model, *args: Any) -> None

    # Set pre/post-process methods
    (method) set_preprocess: (self: Self@Model, **kwargs: Any) -> None
    ```
    """

    def __init__(self, structure=None, *args, **kwargs):
        """
        Init a model object.

        :param structure: The training structure that used to manage this model, \
            which should has the type `Structure` (or its subclasses).
        """
        torch.nn.Module.__init__(self)
        BaseManager.__init__(self, manager=structure,
                             name=f'{type(self).__name__}({hex(id(self))})')

        # Pre/post-process model and settings
        preprocess = []
        for index, operation in enumerate([PROCESS_TYPES.MOVE,
                                           PROCESS_TYPES.SCALE,
                                           PROCESS_TYPES.ROTATE]):
            if self.args.preprocess[index] == '1':
                preprocess.append(operation)

        self.processor = process.ProcessModel(self.args)
        self.set_preprocess(*preprocess)

        # Init model inputs and labels
        self.input_types: list[str] = []
        self.label_types: list[str] = []

        if 'as_single_model' in kwargs.keys():
            self.as_single_model = kwargs['as_single_model']
        else:
            self.as_single_model = True

        self.as_final_stage_model = False

        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ)
        self.set_labels(INPUT_TYPES.GROUNDTRUTH_TRAJ)

        # Inference times
        self.inference_times: list[float] = []

        # Containers for extra times steps as input or output
        self.__input_pred_steps: torch.Tensor | None = None
        self.__output_pred_steps: torch.Tensor | None = None

        # Extra model outputs and their indexes
        self.ext_traj_wise_outputs: dict[int, str] = {}
        self.ext_agent_wise_outputs: dict[int, str] = {}

    @property
    def structure(self) -> BaseManager:
        return self.manager

    @structure.setter
    def structure(self, value):
        self.manager = value

    @property
    def picker(self):
        """
        Trajectory picker (from the top manager object).
        """
        return self.get_top_manager().get_member(AgentManager).picker

    @property
    def parameter_count(self) -> int:
        """
        The number of parameters in this model.
        """
        count = 0
        for param in self.parameters():
            count += torch.prod(torch.tensor(param.shape)).numpy()
        return count

    @property
    def average_inference_time(self) -> int | str:
        """
        Average inference time (ms).
        """
        if l := len(it := self.inference_times):
            if l > 3:
                it = it[1:-1]
            t = float(np.mean(it))
            return int(1000 * t)
        else:
            return '(Not Available)'

    @property
    def fastest_inference_time(self) -> int | str:
        """
        The fastest inference time (ms).
        """
        if l := len(it := self.inference_times):
            if l > 3:
                it = it[1:-1]
            t = min(it)
            return int(1000 * t)
        else:
            return '(Not Available)'

    @property
    def input_pred_steps(self) -> torch.Tensor | None:
        """
        Indices of future time steps that used as model inputs.
        It returns a tensor with type `torch.int32` or `None`.
        """
        if self.__input_pred_steps is None:
            if (i := self.args.input_pred_steps) == 'null':
                return None
            s = torch.tensor([int(_i) for _i in i.split('_')],
                             dtype=torch.int32)
            self.__input_pred_steps = s
        return self.__input_pred_steps

    @property
    def output_pred_steps(self) -> torch.Tensor:
        """
        Indices of future time steps to be predicted.
        It returns a tensor with type `torch.int32`.
        """
        if self.__output_pred_steps is None:
            if (i := self.args.output_pred_steps) != 'all':
                s = torch.tensor([int(_i) for _i in i.split('_')],
                                 dtype=torch.int32)
                self.__output_pred_steps = s
            else:
                s = torch.arange(0, self.args.pred_frames,
                                 dtype=torch.int32)
                self.__output_pred_steps = s
        return self.__output_pred_steps

    @property
    def d(self) -> int:
        """
        Feature dimension used in most layers.
        """
        return self.args.feature_dim

    @property
    def d_id(self) -> int:
        """
        Dimension of randomly sampled noise vectors.
        """
        return self.args.noise_depth

    @property
    def dim(self) -> int:
        """
        Dimension of the predicted trajectory.
        For example, `dim = 4` for 2D bounding boxes.
        """
        return self.structure.get_member(AnnotationManager).dim

    def get_input(self, inputs: list[torch.Tensor], dtype: str):
        """
        Get the input tensor with the given type from all model inputs.
        """
        if dtype == INPUT_TYPES.GROUNDTRUTH_KEYPOINTS:
            if self.input_pred_steps is None:
                raise ValueError

            if self.as_single_model:
                points = self.get_input(inputs, INPUT_TYPES.GROUNDTRUTH_TRAJ)
                points = points[..., self.input_pred_steps, :]
            else:
                points = inputs[-1]

            return self.processor([points], preprocess=True,
                                  update_paras=False)[0]

        index = self.input_types.index(dtype)
        return inputs[index]

    def get_label(self, labels: list[torch.Tensor], dtype: str):
        """
        Get the label tensor with the given type from all model labels.
        """
        index = self.label_types.index(dtype)
        return labels[index]

    def forward(self, inputs,
                training=None,
                mask=None,
                *args, **kwargs):

        raise NotImplementedError

    def implement(self, inputs: list[torch.Tensor],
                  training=None) -> list[torch.Tensor]:
        """
        Run a forward implementation.

        :param inputs: Input tensor (or a `list` of tensors).
        :param training: Config if running as training or test mode.
        :return outputs_p: Model's output. type=`list[torch.Tensor]`.
        """
        # Preprocess
        inputs_p = self.processor(inputs, preprocess=True, training=training)

        # Model inference
        time_start = time.time()
        outputs = self(inputs_p, training=training)
        time_end = time.time()

        # Compute time costs
        l = MAX_INFERENCE_TIME_STORGED
        if len(self.inference_times) > l:
            self.inference_times = self.inference_times[l//2:]

        time_cost = time_end - time_start
        self.inference_times.append(time_cost)

        # Postprocess
        return self.processor(outputs, preprocess=False, training=training)

    def down_sample_predictions(self, outputs: list[torch.Tensor],
                                inputs: list[torch.Tensor],
                                rate: float) -> list[torch.Tensor]:
        """
        Down-sampling the final predicted trajectories (if this model is a 
        multiple-generative model).
        NOTE: This method is typically used when the required number of
        trajectories is less than the actual model's generated ones.

        :param outputs: All this model's outputs.
        :param inputs: All this model's inputs. This is used to check if the \
            model is a multi-generation model.
        :param rate: The down-sampling rate, from 0.0 to 1.0.

        :return outputs: The list of outputs, containing the down-sampled \
            forecasted trajectories.
        """
        # Unpack outputs and inputs
        x = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        y = outputs[0]

        if x.ndim == y.ndim:
            self.log('It seems that you are trying to down-sample outputs ' +
                     'from this non-multi-generative model. ' +
                     'Please check your args.',
                     level='warning')
            return outputs

        # Shape of `x` is `(..., obs, dim)`
        # Shape of `y` is `(..., K, pred, dim)`
        k_current = y.shape[-3]
        k_new = int(k_current * rate)
        k_selected = torch.randperm(k_current)[:k_new]

        y_selected = y[..., k_selected, :, :]
        return [y_selected] + outputs[1:]

    def set_inputs(self, *args):
        """
        Set input types of the model.
        Accept keywords:
        ```python
        codes.constant.INPUT_TYPES.OBSERVED_TRAJ
        codes.constant.INPUT_TYPES.MAP
        codes.constant.INPUT_TYPES.DESTINATION_TRAJ
        codes.constant.INPUT_TYPES.GROUNDTRUTH_TRAJ
        codes.constant.INPUT_TYPES.GROUNDTRUTH_SPECTRUM
        codes.constant.INPUT_TYPES.ALL_SPECTRUM
        ```

        :param input_names: Type = `str`, accept several keywords.
        """
        self.input_types = [item for item in args]
        self.processor.set_preprocess_input_types(self.input_types)

    def set_labels(self, *args):
        """
        Set label types when calculating loss and metrics.
        Accept keywords:
        ```python
        codes.constant.INPUT_TYPES.OBSERVED_TRAJ
        codes.constant.INPUT_TYPES.MAP
        codes.constant.INPUT_TYPES.DESTINATION_TRAJ
        codes.constant.INPUT_TYPES.GROUNDTRUTH_TRAJ
        codes.constant.INPUT_TYPES.GROUNDTRUTH_SPECTRUM
        codes.constant.INPUT_TYPES.ALL_SPECTRUM
        ```

        :param input_names: Name of the inputs.\
            Type = `str`, accept several keywords.
        """
        self.label_types = [item for item in args]

    def set_preprocess(self, *args, **kwargs):
        """
        Set pre-process methods used before training.
        For example, enable all three preprocess methods by
        ```python
        self.set_preprocess(PROCESS_TYPES.MOVE,
                            PROCESS_TYPES.ROTATE,
                            PROCESS_TYPES.SCALE)
        ```

        *IMPORTANT NOTE:* You MUST call `self.set_inputs` after the
        `set_preprocess` to initialize all process layers.

        Extra args:
        :param builtin: Controls whether preprocess methods applied
            outside from the `call` method. If `builtin == True`, the
            preprocess layer will be called outside from the `call`.
        """
        return self.processor.set_process(*args, **kwargs)

    def load(self, dir: str, device: torch.device):
        """
        Load model weights from the given directory.
        """
        # Check all possible weights files in this dir
        all_files = os.listdir(dir)
        weights_files = [f for f in all_files if WEIGHTS_FORMAT in f]
        weights_files.sort()

        # Load according to the received arg
        if (e := self.args.load_epoch) >= 0:
            weights_files = [f for f in weights_files
                             if f'_epoch{e}{WEIGHTS_FORMAT}' in f]

        # Load according to the checkpoint file
        elif CHECKPOINT_FILENAME in all_files:
            p = os.path.join(dir, CHECKPOINT_FILENAME)
            epoch = int(np.loadtxt(p)[1])

            weights_files = [f for f in weights_files
                             if f'_epoch{epoch}{WEIGHTS_FORMAT}' in f]

        else:
            self.log('Neither the checkpoint file nor the arg `load_epoch` ' +
                     f'was found in `{dir}`. This may result in ' +
                     'incorrect weights being loaded.',
                     level='warning')

        if not len(weights_files):
            self.log(f'Failed to load model weights from `{dir}`!',
                     level='error', raiseError=ValueError)

        weights_name = weights_files[-1]
        p = os.path.join(dir, weights_name)
        dic = torch.load(p, map_location=device)

        # Fully or partially Load weights into models
        try:
            self.load_state_dict(
                dic, strict=False if self.args.load_part else True)
        except RuntimeError:
            self.log('The loaded weight does not match the code. Please ' +
                     'check the above outputs for detailed descriptions. ' +
                     'Consider adding the arg `--load_part` to fix it ' +
                     'if you insist on loading this weight.',
                     level='error', raiseError=ValueError)

        self.log(f'Model weights successfully loaded from `{p}`.')

    def print_info(self, **kwargs):
        try:
            p_layers = self.processor.get_layer_names()
        except:
            p_layers = None

        info = {'Model name': self.args.model_name,
                'Prediction type': self.args.anntype,
                'Prediction settings': f'{self.args.obs_frames} frames ' +
                                       f'-> {self.args.pred_frames} frames, ' +
                                       f'interval = {self.args.interval} seconds',
                'Pre/post-process layers': p_layers,
                'Parameters Amount': f'{self.parameter_count:,}'}

        if (s := self.input_pred_steps) is not None:
            info.update({'Index of input future steps': s})

        if len(s := self.output_pred_steps) < self.args.pred_frames:
            info.update({'Index of predicted future steps': s})

        # Print descriptions of important args
        sub_dicts: dict[str, dict] = {}
        for _arg in [self.args] + list(self.args._subargs_dict.values()):
            for name, desc in _arg._args_desc.items():
                value = _arg._get(name)

                # Add as a single line
                if isinstance(desc, str):
                    info[desc] = value

                # Add as an item of the arg-group
                elif (isinstance(desc, tuple) or isinstance(desc, list)):
                    _class, _desc = desc[:2]
                    _desc = ' ' * 4 + _desc

                    if not _class in sub_dicts.keys():
                        sub_dicts[_class] = {_class: None}

                    sub_dicts[_class][_desc] = value

                else:
                    raise ValueError(desc)

        for sub_dict in sub_dicts.values():
            info.update(sub_dict)

        return super().print_info(**info, **kwargs)
