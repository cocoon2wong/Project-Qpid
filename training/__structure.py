"""
@Author: Conghao Wong
@Date: 2022-06-20 16:27:21
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-11 10:45:03
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
from typing import Union, overload

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..args import Args
from ..base import BaseManager
from ..constant import ANN_TYPES, INPUT_TYPES
from ..dataset import AgentManager, Annotation, AnnotationManager, SplitManager
from ..model import Model
from ..utils import WEIGHTS_FORMAT, dir_check, get_loss_mask
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

    def __init__(self, args: Union[list[str], Args] = None,
                 manager: BaseManager = None,
                 name='Train Manager'):

        if issubclass(type(args), Args):
            init_args = args
        else:
            init_args = Args(args)

        super().__init__(init_args, manager, name)

        # check args (such as wrong spellings)
        if not self.manager:
            self.args._check_terminal_args()

        # init managers
        self._am = AgentManager(self)
        self.annmanager = AnnotationManager(self, self.split_manager.anntype)
        self.loss = LossManager(self, name='Loss')
        self.metrics = LossManager(self, name='Metrics',
                                   trajectory_scale=self.split_manager.scale)

        # init model options
        self.model: Model = None
        self.optimizer: torch.optim.Optimizer = None
        self.noTraining = False

        # init compute devices
        self._device: torch.device = None
        self._device_local: torch.device = None

        # Set labels, loss functions, and metrics
        self.label_types: list[str] = []
        self.set_labels(INPUT_TYPES.GROUNDTRUTH_TRAJ)
        self.loss.set({loss.ADE: 1.0})

        if self.args.anntype in [ANN_TYPES.BB_2D,
                                 ANN_TYPES.BB_3D]:

            self.metrics.set({loss.ADE: 1.0,
                              loss.FDE: 0.0,
                              loss.AIOU: 0.0,
                              loss.FIOU: 0.0})

        elif self.args.anntype in [ANN_TYPES.SKE_3D_17]:
            # These configs are only used on `h36m` dataset
            i = int(1000 * self.args.interval)  # Sample interval

            if self.args.pred_frames == 10:
                self.metrics.set([
                    [loss.FDE, [0.0, {'index': 1, 'name': f'FDE@{2*i}ms'}]],
                    [loss.FDE, [0.0, {'index': 3, 'name': f'FDE@{4*i}ms'}]],
                    [loss.FDE, [0.0, {'index': 7, 'name': f'FDE@{8*i}ms'}]],
                    [loss.FDE, [1.0, {'index': 9, 'name': f'FDE@{10*i}ms'}]],
                ])

            elif self.args.pred_frames == 25:
                self.metrics.set([
                    [loss.FDE, [0.0, {'index': 13, 'name': f'FDE@{14*i}ms'}]],
                    [loss.FDE, [1.0, {'index': 24, 'name': f'FDE@{25*i}ms'}]],
                ])

        else:
            self.metrics.set({loss.ADE: 1.0,
                              loss.FDE: 0.0})

    @property
    def agent_manager(self) -> AgentManager:
        return self._am

    @property
    def picker(self) -> Annotation:
        return self.annmanager.annotations[self.args.anntype]

    @property
    def split_manager(self) -> SplitManager:
        """
        The Split Manager is managed by the `AgentManager`.
        """
        return self.agent_manager.split_manager

    @property
    def device(self):
        """
        Compute device (use GPU if available).
        """
        if self._device is None:
            self._device = torch.device("cpu")
        return self._device

    @property
    def device_local(self):
        """
        Basic compute device (like CPU).
        """
        if self._device_local is None:
            self._device_local = torch.device("cpu")
        return self._device_local

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

    def set_optimizer(self, epoch: int = None) -> torch.optim.Optimizer:
        if self.noTraining:
            return None

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.lr)
        return self.optimizer

    def set_gpu(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu.replace('_', ',')
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        # FIXME: Set GPU

    def create_model(self) -> Model:
        """
        Create models.
        Please *rewrite* this when training new models.

        :return model: created model
        """
        raise NotImplementedError('MODEL is not defined!')

    def gradient_operations(self, inputs: list[torch.Tensor],
                            labels: list[torch.Tensor],
                            loss_move_average: torch.Tensor,
                            *args, **kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """
        Run gradient descent once during training.

        :param inputs: Model inputs.
        :param labels: Ground truth.
        :param loss_move_average: Moving average loss.

        :return loss: The sum of all single loss functions.
        :return loss_dict: A dict of all loss functions.
        :return loss_move_average: Moving average loss.
        """
        # Compute predictions
        outputs = self.model.implement(inputs, training=True)
        loss, loss_dict = self.loss.forward(
            outputs, labels, inputs=inputs, training=True)
        loss_move_average = 0.7 * loss + 0.3 * loss_move_average

        # Compute gradients
        loss.backward()
        self.optimizer.step()

        return loss, loss_dict, loss_move_average

    def model_validate(self, inputs: list[torch.Tensor],
                       labels: torch.Tensor,
                       training=None) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        """
        Run one step forward and calculate metrics.

        :param inputs: Model inputs.
        :param labels: Ground truth.

        :return model_output: Model output.
        :return metrics: The weighted sum of all loss.
        :return loss_dict: A dict contains all loss.
        """
        outputs = self.model.implement(inputs, training)
        metrics, metrics_dict = self.metrics.forward(
            outputs, labels, inputs=inputs, training=None)

        if self.args.compute_loss:
            _, loss_dict = self.loss.forward(
                outputs, labels, inputs=inputs, training=None)

            metrics_dict.update(loss_dict)

        return outputs, metrics, metrics_dict

    def train_or_test(self):
        """
        Load models and datasets, then start training or testing.
        """
        # init model and dataset manager
        self.model = self.create_model().to(self.device)
        self.optimizer = self.set_optimizer()
        self.set_gpu()
        self.agent_manager.set_types(inputs_type=self.model.input_types,
                                     labels_type=self.label_types)

        # start training or testing
        if self.noTraining:
            self.test()

        elif self.args.load == 'null':
            # restore weights before training (optional)
            if self.args.restore != 'null':
                self.model.load_weights_from_logDir(self.args.restore)
            self.train()

        else:
            self.model.load_weights_from_logDir(self.args.load)
            self.test()

    def train(self):
        """
        Start training according to the args.
        """
        self.log(f'Start training with args = {self.args._args_runnning}')

        clips_train = self.split_manager.train_sets
        clips_val = self.split_manager.test_sets
        ds_train = self.agent_manager.make(clips_train, training=True)
        ds_val = self.agent_manager.clean().make(clips_val, training=False)

        # train on all test/train clips
        _, _, best_metric, best_epoch = self.__train(ds_train, ds_val)
        self.print_train_results(best_epoch=best_epoch,
                                 best_metric=best_metric)

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

    def __train(self, ds_train: DataLoader, ds_val: DataLoader):
        """
        Train the model on the given dataset.

        :param ds_train: The train dataset.
        :param ds_val: The val dataset.

        :return loss_dict:
        :return metrics_dict:
        :return best_metric:
        :return best_epoch:
        """
        # print training infomation
        self.split_manager.print_info()
        self.agent_manager.print_info()
        self.model.print_info()
        self.print_info()

        # make a log directory and save current args
        self.args._save_as_json(self.args.log_dir)

        # open tensorboard
        tb = SummaryWriter(self.args.log_dir)

        # init variables for training
        loss_move = torch.autograd.Variable(
            torch.tensor(0, dtype=torch.float32))
        loss_dict = {}
        metrics_dict = {}

        best_epoch = 0
        best_metric = 10000.0
        best_metrics_dict = {'-': best_metric}
        test_epochs = []
        train_number = len(ds_train)

        # start training
        for epoch in self.timebar(range(self.args.epochs), text='Training...'):

            # Update learning rate and optimizer
            self.set_optimizer(epoch)

            # Split into batches
            for inputs, labels in ds_train:
                # Move data to GPU
                inputs = [i.to(self.device) for i in inputs]
                labels = [i.to(self.device) for i in labels]

                # Run training once
                len_labels = len(self.label_types)
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

                metric, metrics_dict = self.__test_on_dataset(
                    ds=ds_val,
                    show_timebar=False,
                    test_during_training=True
                )

                # Save model
                if metric <= best_metric:
                    best_metric = metric
                    best_metrics_dict = metrics_dict
                    best_epoch = epoch

                    torch.save(self.model.state_dict(),
                               os.path.join(self.args.log_dir,
                                            f'{self.args.model_name}_epoch{epoch}' +
                                            WEIGHTS_FORMAT))

                    np.savetxt(os.path.join(self.args.log_dir, 'best_ade_epoch.txt'),
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

        return log_dict, metrics_dict, best_metric, best_epoch

    def __test(self, ds_test: DataLoader) -> \
            tuple[float, dict[str, float], list[torch.Tensor]]:
        """
        Test model on the given dataset.

        :param ds_test: The test dataset.

        :return metric:
        :return metrics_dict
        :return outputs: model outputs
        """
        # Print test information
        self.split_manager.print_info()
        self.agent_manager.print_info()
        self.model.print_info()
        self.print_info()

        # make a log directory and save current args
        if self.args.update_saved_args:
            if self.args.load != 'null':
                self.args._save_as_json(self.args.load)
            else:
                self.args._save_as_json(self.args.log_dir)

        # Run test
        outputs, metric, metrics_dict = self.__test_on_dataset(
            ds=ds_test,
            return_results=True,
            show_timebar=True,
        )

        return metric, metrics_dict, outputs

    @overload
    def __test_on_dataset(self, ds: DataLoader,
                          show_timebar=False,
                          test_during_training=False) \
        -> tuple[float, dict[str, float]]: ...

    @overload
    def __test_on_dataset(self, ds: DataLoader,
                          return_results=False,
                          show_timebar=False,
                          test_during_training=False) \
        -> tuple[list[torch.Tensor], float, dict[str, float]]: ...

    @torch.no_grad()
    def __test_on_dataset(self, ds: DataLoader,
                          return_results=False,
                          show_timebar=False,
                          test_during_training=False):
        """
        Run a test on the given dataset.

        :param ds: The test `DataLoader` object.
        :param return_results: Controls items to return (the defaule value is `False`).
        :param show_timebar: Controls whether to show the process.
        :param test_during_training: Indicates whether to test during training.

        Returns if `return_results == False`:
        :return metric: The weighted sum of all metrics.
        :return metric_dict: A dict of all metrics.

        Returns if `return_results == True`:
        :return outputs: A list of model outputs.
        :return metric: The weighted sum of all metrics.
        :return metric_dict: A dict of all metrics.
        """
        # init variables for test
        outputs_all = []
        batch_all_metrics = []
        batch_weightedsum_metrics = []
        metrics_names: list[str] = None

        # hide time bar when training
        timebar = self.timebar(ds, 'Test...') if show_timebar else ds

        count = []
        for x, gt in timebar:
            # Move data to GPU
            x = [i.to(self.device) for i in x]
            gt = [i.to(self.device) for i in gt]

            mask = get_loss_mask(x[0], gt[0])
            valid_count = torch.sum(mask)

            outputs, metrics, metrics_dict = self.model_validate(
                inputs=x, labels=gt, training=False)

            # Check if there are valid trajectories in this batch
            if valid_count == 0:
                outputs[0] = torch.zeros_like(outputs[0]) / 0.0

            # Add metrics and outputs to their dicts
            else:
                count.append(outputs[0].shape[0])
                metrics_names = list(metrics_dict.keys())
                batch_all_metrics.append(list(metrics_dict.values()))
                batch_weightedsum_metrics.append(metrics)

            if return_results:
                outputs_all.append(outputs)

        # Stack all model results
        if return_results:
            outputs_all = stack_batch_outputs(outputs_all)

        # Calculate average metrics
        all_metrics = weighted_average(batch_all_metrics, count,
                                       return_numpy=True)
        weightedsum_metrics = weighted_average(batch_weightedsum_metrics,
                                               count, return_numpy=True)

        # Make the metric dict
        mdict_avg = dict(zip(metrics_names, all_metrics))

        # Compute the inference time
        if not test_during_training:
            if len(self.model.inference_times) < 3:
                self.log('The "AverageInferenceTime" is for reference only and you can set a lower "batch_size" ' +
                         'or change a bigger dataset to obtain a more accurate result.')

            mdict_avg['Average Inference Time'] = f'{self.model.average_inference_time} ms'
            mdict_avg['Fastest Inference Time'] = f'{self.model.fastest_inference_time} ms'

        if return_results:
            return outputs_all, weightedsum_metrics, mdict_avg
        else:
            return weightedsum_metrics, mdict_avg

    def print_info(self, **kwargs):
        info = {'Batch size': self.args.batch_size,
                'GPU index': self.args.gpu,
                'Train epochs': self.args.epochs,
                'Learning rate': self.args.lr}

        kwargs.update(**info)
        return super().print_info(**kwargs)

    def print_train_results(self, best_epoch: int, best_metric: float):
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

    def print_test_results(self, loss_dict: dict[str, float], **kwargs):
        """
        Print test results on the screen.
        """
        self.print_parameters(title='Test Results',
                              **kwargs,
                              **loss_dict)
        self.log(f'split: {self.args.split}, ' +
                 f'load: {self.args.load}, ' +
                 f'metrics: {loss_dict}')

    def write_test_results(self, outputs: list[torch.Tensor], clips: list[str]):
        """
        Save visualized prediction results.
        """

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

            save_base_path = dir_check(self.args.log_dir) \
                if self.args.load == 'null' \
                else self.args.load

            img_dir = dir_check(os.path.join(save_base_path, 'VisualTrajs'))
            save_format = os.path.join(img_dir, clip + '_{}')
            tv.log(f'Start saving images into `{img_dir}`...')

            pred_all = outputs[0].numpy()
            traj_wise_outputs = dict([
                (key, outputs[i].numpy())
                for i, key in self.model.ext_traj_wise_outputs.items()])

            agent_wise_outputs = dict([
                (key, outputs[i].numpy())
                for i, key in self.model.ext_agent_wise_outputs.items()])

            if self.args.draw_index == 'all':
                agent_indexes = list(range(len(pred_all)))
            else:
                _indexes = self.args.draw_index.split('_')
                agent_indexes = [int(i) for i in _indexes]

            ex_types: list[str] = []
            if self.args.draw_exclude_type != 'null':
                ex_types = self.args.draw_exclude_type.split("_")

            for index in self.timebar(agent_indexes, 'Saving...'):
                # write traj
                agent = self.agent_manager.agents[index]
                agent.write_pred(pred_all[index])

                # extra outputs
                to = dict([(k, v[index])
                          for (k, v) in traj_wise_outputs.items()])
                ao = dict([(k, v[index])
                          for (k, v) in agent_wise_outputs.items()])

                # choose to draw as a video or a single image
                if self.args.draw_videos != 'null':
                    save_image = False
                    frames = agent.frames
                else:
                    save_image = True
                    frames = [agent.frames[self.args.obs_frames-1]]

                skip = False
                for extype in ex_types:
                    if extype in agent.type:
                        skip = True
                        break
                if skip:
                    continue

                tv.draw(agent=agent,
                        frames=frames,
                        save_name=save_format.format(index),
                        draw_dis=self.args.draw_distribution,
                        save_as_images=save_image,
                        traj_wise_outputs=to,
                        agent_wise_outputs=ao)

            self.log(f'Prediction result images are saved at {img_dir}')


def _get_item(item, indices: list):
    res = item
    for i in indices:
        res = res[i]
    return res


def stack_batch_outputs(outputs: list[list[torch.Tensor]]):
    """
    Stack several batches' model outputs.
    Input of this function should be a list of model outputs,
    where each list member is a batch's output.
    """
    # Check output shapes
    indices = []
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


def weighted_average(inputs: list, weights: list, return_numpy=False) -> list:
    """
    Weighted sum all the inputs.
    NOTE: The length of `inputs` and `weights` should be the same value.
    """
    inputs = torch.tensor(inputs, dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32)
    count = torch.sum(weights)

    while weights.ndim < inputs.ndim:
        weights = weights[..., None]

    res = torch.sum(inputs * weights / count, dim=0)

    if return_numpy:
        res = res.numpy()

    return list(res) if res.ndim else res
