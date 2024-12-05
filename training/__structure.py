"""
@Author: Conghao Wong
@Date: 2022-06-20 16:27:21
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-05 16:47:44
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
from typing import Any, overload

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from ..args import Args
from ..base import BaseManager
from ..constant import ANN_TYPES, STRUCTURE_STATUS
from ..dataset import AgentManager, Annotation, AnnotationManager, SplitManager
from ..model import Model
from ..utils import WEIGHTS_FORMAT, move_to_device
from . import loss
from .loss import LossManager


class Structure(BaseManager):
    """
    Structure
    ---------
    Training manager that manages all training/test-related methods.

    Member Managers
    ---------------
    - Model, `type = Model`;
    - Agent Manager, `type = AgentManager`;
    - Annotation Manager, `type = AnnotationManager`;
    - Loss Manager, `type = LossManager`;
    - Metrics Manager, `type = LossManager`.

    Public Methods
    --------------
    ```python
    # Start training or testing
    (method) train_or_test: (self: Self@Structure) -> None
    ```

    Other methods should be rewritten when subclassing.
    """

    is_trainable = True
    MODEL_TYPE: type[Model] | None = None

    def __init__(self, args: list[str] | Args | None = None,
                 manager: BaseManager | None = None,
                 name='Train Manager'):

        if isinstance(args, Args):
            init_args = args
        else:
            init_args = Args(args)

        super().__init__(init_args, manager, name)

        # Init device
        self.set_gpu()
        self._device: torch.device | None = None
        self._device_local: torch.device | None = None

        # Init managers
        self.agent_manager = AgentManager(self)
        self.split_manager = self.agent_manager.split_manager
        self.ann_manager = AnnotationManager(self, self.split_manager.anntype)
        self.loss = LossManager(self, name='Loss')
        self.metrics = LossManager(self, name='Metrics',
                                   trajectory_scale=self.split_manager.scale)

        # Init models and the optimizer (placeholders)
        self.model: Model
        self.optimizer: torch.optim.Optimizer

        # Set default loss functions and metrics
        self.loss.set_default_loss()
        self.metrics.set_default_metrics()

    @property
    def status(self) -> int:
        """
        Status of the training structure.
        Returns could be one of the
        ```python
        [STRUCTURE_STATUS.TEST, 
         STRUCTURE_STATUS.TEST_WITH_SAVED_WEIGHTS,
         STRUCTURE_STATUS.TRAIN, 
         STRUCTURE_STATUS.TRAIN_WITH_SAVED_WEIGHTS]
        ```
        """
        if not self.is_trainable:
            return STRUCTURE_STATUS.TEST
        elif self.args.load == 'null':
            if self.args.restore == 'null':
                return STRUCTURE_STATUS.TRAIN
            else:
                return STRUCTURE_STATUS.TRAIN_WITH_SAVED_WEIGHTS
        else:
            return STRUCTURE_STATUS.TEST_WITH_SAVED_WEIGHTS

    @property
    def is_prepared_for_training(self) -> bool:
        """
        Return `True` if it is now preparing for training.
        """
        return STRUCTURE_STATUS.is_training(self.status)

    @property
    def picker(self) -> Annotation:
        return self.ann_manager.annotations[self.args.anntype]

    @property
    def device(self):
        """
        Compute device (use GPU if available).
        """
        if self._device is None:
            if (torch.cuda.is_available() and
                    int(self.args.gpu.split('_')[0]) >= 0):
                d = torch.device("cuda")
            elif torch.backends.mps.is_available() and self.args.macos:
                d = torch.device("mps")
            else:
                d = torch.device("cpu")

            self._device = d
        return self._device

    @property
    def device_cpu(self):
        """
        Basic compute device (like CPU).
        """
        if self._device_local is None:
            self._device_local = torch.device("cpu")
        return self._device_local

    def set_optimizer(self):
        if self.is_prepared_for_training:
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.args.lr)

    def set_gpu(self):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu.replace('_', ',')
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if torch.cuda.is_available():
            gpu_id = int(self.args.gpu.split('_')[0])
            torch.cuda.set_device(gpu_id)

    def create_model(self, *args, **kwargs) -> None:
        """
        Create the trainable model objects `self.model` according to the
        default `MODEL_TYPE` setting.
        NOTE: The created model objects should be assign to `self.model`
        inner this method, and this method returns `None` only.
        """
        if not self.MODEL_TYPE:
            raise NotImplementedError('MODEL_TYPE is not defined!')
        self.model = self.MODEL_TYPE(structure=self, *args, **kwargs)

    def gradient_operations(self, inputs: list[torch.Tensor],
                            labels: list[torch.Tensor],
                            loss_move_average: torch.Tensor,
                            *args, **kwargs) -> tuple[torch.Tensor,
                                                      dict[str, torch.Tensor],
                                                      torch.Tensor]:
        """
        Run gradient descent once during training.

        :param inputs: Model inputs. It should be a list of tensors.
        :param labels: Ground truth. It should be a list of tensors.
        :param loss_move_average: Moving average loss.

        :return loss: The sum of all single loss functions.
        :return loss_dict: A dict of all loss functions.
        :return loss_move_average: Moving average loss.
        """
        # Compute predictions
        outputs = self.model.implement(inputs, training=True)
        loss, loss_dict, _ = self.loss.compute(outputs, labels,
                                               inputs, training=True)
        loss_move_average = 0.7 * loss + 0.3 * loss_move_average.item()

        # Compute gradients and run optimizer
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss, loss_dict, loss_move_average

    def train_or_test(self):
        """
        Load models and datasets, then start training or testing.
        """
        # Init model and dataset manager
        self.create_model()
        self.model.to(self.device)
        self.set_optimizer()
        self.agent_manager.set_types(input_types=self.model.input_types,
                                     label_types=self.model.label_types)

        # Start training or testing
        match self.status:
            case STRUCTURE_STATUS.TEST:
                self.test()
            case STRUCTURE_STATUS.TEST_WITH_SAVED_WEIGHTS:
                self.model.load_weights_from_logDir(self.args.load)
                self.test()
            case STRUCTURE_STATUS.TRAIN:
                self.train()
            case STRUCTURE_STATUS.TRAIN_WITH_SAVED_WEIGHTS:
                self.model.load_weights_from_logDir(self.args.restore)
                self.train()

    @torch.no_grad()
    def test(self):
        """
        Run a test on the given dataset.
        """
        self.log(f'Start test with args = {self.args._args_runnning}')
        test_sets = self.split_manager.test_sets
        r = None

        # test on a single sub-dataset
        if self.args.test_mode == 'one':
            clip = self.args.force_clip
            ds_test = self.agent_manager.make(clip, training=False)
            r = self.__test(ds_test)

        # test on all test datasets separately
        elif self.args.test_mode == 'all':
            metrics_dict = {}
            for clip in test_sets:
                ds_test = self.agent_manager.make(clip, training=False)
                _, m_dict, _ = self.__test(ds_test)
                metrics_dict[clip] = m_dict

            self.print_test_results(metrics_dict)

        # test on all test datasets together
        elif self.args.test_mode == 'mix':
            ds_test = self.agent_manager.make(test_sets, training=False)
            r = self.__test(ds_test)

        else:
            raise NotImplementedError(self.args.test_mode)

        # Write test results
        if r:
            metric, metrics_dict, outputs = r
            self.print_test_results(metrics_dict)
            self.write_test_results(outputs=outputs,
                                    clips=self.agent_manager.processed_clips['test'])

    def train(self):
        """
        Start training according to the args.
        """
        self.log(f'Start training with args = {self.args._args_runnning}')

        # Prepare dataset
        clips_train = self.split_manager.train_sets
        clips_val = self.split_manager.test_sets
        ds_train = self.agent_manager.make(clips_train, training=True)
        ds_val = self.agent_manager.clean().make(clips_val, training=False)

        # Print training infomation
        self.agent_manager.print_info()
        self.model.print_info()
        self.print_info()

        # Make the log directory and save current args
        self.args.save_args_as_json(self.args.log_dir)

        # Open tensorboard
        tb = SummaryWriter(self.args.log_dir)

        # Init variables for training
        loss_move = torch.tensor(0, dtype=torch.float32)
        loss_dict = {}
        metrics_dict = {}

        best_epoch = 0
        best_metric = 10000.0
        best_metrics_dict = {'-': best_metric}

        # Init paths for saving
        weights_path = os.path.join(self.args.log_dir,
                                    self.args.model_name + '_epoch{}' +
                                    WEIGHTS_FORMAT)
        checkpoint_path = os.path.join(self.args.log_dir, 'best_ade_epoch.txt')

        # start training
        for epoch in self.timebar(range(self.args.epochs), text='Training...'):

            # Split into batches
            for inputs, labels in ds_train:
                # Move data to GPU
                inputs = move_to_device(inputs, self.device)
                labels = move_to_device(labels, self.device)

                # Run training once
                loss, loss_dict, loss_move = self.gradient_operations(
                    inputs=inputs,
                    labels=labels,
                    loss_move_average=loss_move,
                    epoch=epoch,
                )

                # Check if `nan` in the loss dictionary
                if torch.isnan(loss):
                    self.log(f'Find `nan` values in the loss dictionary, ' +
                             f'stop training... ' +
                             f'Best metrics obtained from the last epoch: ' +
                             f'{best_metrics_dict}.',
                             level='error', raiseError=ValueError)

            # Training done (this epoch)
            # Run validation
            if ((epoch >= self.args.start_test_percent * self.args.epochs)
                    and ((epoch - 1) % self.args.test_step == 0)):

                _, metric, metrics_dict = self.__test_on_dataset(
                    ds=ds_val,
                    show_timebar=False,
                    is_during_training=True
                )

                # Save model
                if metric <= best_metric:
                    # Delete former saved weights (if needed)
                    if self.args.auto_clear:
                        if os.path.exists(p := weights_path.format(best_epoch)):
                            os.remove(p)

                    # Save new weights
                    best_metric = metric
                    best_metrics_dict = metrics_dict
                    best_epoch = epoch

                    torch.save(self.model.state_dict(),
                               weights_path.format(best_epoch))
                    np.savetxt(checkpoint_path,
                               np.array([best_metric, best_epoch]))

            # Save results into log files
            log_dict = dict(epoch=epoch,
                            best=list(best_metrics_dict.values()),
                            **loss_dict,
                            **metrics_dict)

            # Show to users
            self.update_timebar(log_dict, pos='end')

            # Write tensorboard
            for name, value in log_dict.items():
                if name == 'best':
                    value = best_metrics_dict
                    if '-' in value.keys():
                        continue

                    for k, v in value.items():
                        tb.add_scalar(k + ' (Best)', v, epoch)

                else:
                    tb.add_scalar(name, value, epoch)

        # Show summary information
        self.print_train_results(best_epoch=best_epoch,
                                 best_metric=best_metric)

    def __test(self, ds_test: DataLoader):
        """
        Test model on the given dataset.

        :param ds_test: The test dataset.

        :return metric:
        :return metrics_dict
        :return outputs: model outputs
        """
        # Print test information
        self.agent_manager.print_info()
        self.model.print_info()
        self.print_info()

        # Make the log directory and save current args
        if self.args.update_saved_args:
            if self.args.load != 'null':
                self.args.save_args_as_json(self.args.load)
            else:
                self.args.save_args_as_json(self.args.log_dir)

        # Run test
        outputs, metric, metrics_dict = self.__test_on_dataset(
            ds=ds_test,
            return_results=True,
            show_timebar=True,
        )
        return metric, metrics_dict, outputs

    @torch.no_grad()
    def __test_on_dataset(self, ds: DataLoader,
                          return_results=False,
                          show_timebar=False,
                          is_during_training=False):
        """
        Run a test on the given dataset.

        :param ds: The test `DataLoader` object.
        :param return_results: Controls items to return (the defaule value is `False`).
        :param show_timebar: Controls whether to show the process.
        :param is_during_training: Indicates whether it is test during training.

        :return outputs: A list of model outputs (or `None` when `return_results == False`).
        :return metric: The weighted sum of all metrics.
        :return metric_dict: A dict of all metrics.
        """
        # Init variables for test
        outputs_all = []

        all_metrics: dict[str, list[tuple[int, torch.Tensor]]] = {}
        all_count: dict[str, int] = {}

        # Hide the time bar when training
        timebar = self.timebar(ds, 'Test...') if show_timebar else ds

        for x, gt in timebar:
            # Move data to GPU
            x = move_to_device(x, self.device)
            gt = move_to_device(gt, self.device)

            # Run model, compute metrics and loss
            outputs = self.model.implement(x)
            _, _metrics, _count = self.metrics.compute(outputs, gt, x)

            if self.args.compute_loss:
                _, _loss, _ = self.loss.compute(outputs, gt, x, training=True)
                _metrics.update(_loss)

            # Add metrics and outputs to their dicts
            for _name, _value in _count.items():
                if not _name in all_count.keys():
                    all_count[_name] = 0
                all_count[_name] += _value

            for _name, _value in _metrics.items():
                if not _name in all_metrics.keys():
                    all_metrics[_name] = []
                all_metrics[_name].append((_count[_name], _value))

            if return_results:
                outputs_all.append(outputs)

        # Stack all model results
        if return_results:
            outputs_all = stack_batch_outputs(outputs_all)

        # Calculate average metrics
        avg_metrics = {}
        for _name, _value in all_metrics.items():
            _value = torch.tensor(_value)
            _sum = torch.sum(_value.T[0] * _value.T[1])
            _cnt = torch.sum(_value.T[0])
            avg_metrics[_name] = (_sum/_cnt)

        if not is_during_training:
            # Show metrics with units
            unit = self.get_member(AgentManager).get_member(SplitManager).type
            layers_with_units = [
                l.name for l in self.metrics.layers if l.has_unit]

            # Print number of agents in each class
            for layer_name in avg_metrics.keys():
                if layer_name.split('(')[0] in layers_with_units:
                    avg_metrics[layer_name] = f'{avg_metrics[layer_name]} ({unit})'

                if self.args.compute_metrics_with_types:
                    _cnt = all_count[layer_name]
                    avg_metrics[layer_name] = f'{avg_metrics[layer_name]} (on {_cnt} agents)'

            # Resort keys
            avg_metrics = dict(sorted(avg_metrics.items(),
                                      key=lambda item: item[0]))

            # Compute the inference time
            if len(self.model.inference_times) < 3:
                self.log('The "AverageInferenceTime" is for reference only and you can set a lower "batch_size" ' +
                         'or change a bigger dataset to obtain a more accurate result.')

            avg_metrics['Average Inference Time'] = f'{self.model.average_inference_time} ms'
            avg_metrics['Fastest Inference Time'] = f'{self.model.fastest_inference_time} ms'

        if not return_results:
            outputs_all = None

        summary = avg_metrics['__sum']
        if is_during_training:
            avg_metrics.pop('__sum')

        return outputs_all, summary, avg_metrics

    def print_info(self, **kwargs):
        info: dict = {'Batch size': self.args.batch_size}

        if self.device != self.device_cpu:
            info['Speed up device'] = self.device

        if self.is_prepared_for_training:
            info['Learning rate'] = self.args.lr
            info['Training epochs'] = self.args.epochs

        return super().print_info(**kwargs, **info)

    def print_train_results(self, best_epoch: int,
                            best_metric: float | np.ndarray):
        """
        Print train results on the screen.
        """
        self.log('Training done.')
        self.log('During training, the model reaches the best metric ' +
                 f'`{best_metric}` at epoch {best_epoch}.')

        self.log(f'Tensorboard file is saved at `{self.args.log_dir}`. ' +
                 'To open this log file, please use `tensorboard ' +
                 f'--logdir {self.args.log_dir}`')
        self.log(f'Trained model is saved at `{self.args.log_dir}`. ' +
                 'To re-test this model, please use ' +
                 f'`python main.py --load {self.args.log_dir}`.')

    def print_test_results(self, loss_dict: dict[str, Any], **kwargs):
        """
        Print test results on the screen.
        """
        self.print_parameters(title='Test Results',
                              **kwargs,
                              **loss_dict)
        self.log('Test done. ' +
                 f'Split: {self.args.split}, ' +
                 f'load: {self.args.load}, ' +
                 f'metrics: {loss_dict}.')

    def write_test_results(self, outputs: list[Any] | None, clips: list[str]):
        """
        Save visualized prediction results.
        """
        if outputs is None:
            return

        # Move data to cpu
        outputs = move_to_device(outputs, self.device_cpu)

        if (((self.args.draw_results != 'null') or
             (self.args.draw_videos != 'null'))
                and len(clips) == 1):

            if self.args.anntype not in ['coordinate', 'boundingbox']:
                self.log('Currently visualizing with annotation type ' +
                         f'`{self.args.anntype}` is not supported!',
                         level='error', raiseError=NotImplementedError)

            # Import vis package
            from qpid.mods import vis

            # draw results on video frames
            clip = clips[0]
            tv = vis.Visualization(self, self.args.dataset, clip)
            tv.run_commands(outputs)


def _get_item(item: list[torch.Tensor | list[torch.Tensor]],
              indices: list[int]) -> torch.Tensor:
    res = item
    for i in indices:
        res = res[i]

    if not isinstance(res, torch.Tensor):
        raise ValueError(indices)
    return res


def stack_batch_outputs(outputs: list[list[(
    torch.Tensor |
    list[torch.Tensor]
)]]) -> list[torch.Tensor | list[torch.Tensor]]:
    """
    Stack several batches' model outputs.
    Input of this function should be a list of model outputs,
    where each list member is a batch's output.
    """
    # Check output shapes
    indices: list[list[int]] = []
    for index, item in enumerate(outputs[0]):
        if type(item) in [list, tuple]:
            indices += [[index, i] for i in range(len(item))]
        else:
            indices.append([index])

    # Concat all output tensors
    o = [torch.concat([_get_item(_o, _i) for _o in outputs], dim=0)
         for _i in indices]

    final_outputs = []
    for tensor, index in zip(o, indices):
        if (l := len(index)) == 1:
            final_outputs.append(tensor)
        elif l == 2:
            if len(final_outputs) <= index[0]:
                final_outputs.append([])
            final_outputs[index[0]].append(tensor)
        else:
            raise NotImplementedError

    return final_outputs


@overload
def weighted_average(inputs: list[torch.Tensor],
                     weights: list[int]) -> torch.Tensor: ...


@overload
def weighted_average(inputs: list[list[torch.Tensor]],
                     weights: list[int], return_numpy: bool) -> \
    list[np.ndarray]: ...


@overload
def weighted_average(inputs: list[torch.Tensor],
                     weights: list[int], return_numpy: bool) -> \
    np.ndarray: ...


def weighted_average(inputs: list[torch.Tensor] | list[list[torch.Tensor]],
                     weights: list[int], return_numpy=False):
    """
    Weighted sum all the inputs.
    NOTE: `inputs` and `weights` should have the same length.
    """
    x = torch.tensor(inputs, dtype=torch.float32)
    w = torch.tensor(weights, dtype=torch.float32)
    count = torch.sum(w)

    while w.ndim < x.ndim:
        w = w[..., None]

    res = torch.sum(x * w / count, dim=0)

    if return_numpy:
        res = res.numpy()

    return list(res) if res.ndim else res
