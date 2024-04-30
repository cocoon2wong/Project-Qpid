"""
@Author: Conghao Wong
@Date: 2022-06-20 10:53:48
@LastEditors: Conghao Wong
@LastEditTime: 2024-04-30 10:23:02
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

from ..__root import ArgsManager
from ..constant import ARG_TYPES
from ..utils import DATASET_CONFIG_DIR, DEFAULT_LOG_DIR, TIME, dir_check

DYNAMIC = ARG_TYPES.DYNAMIC
STATIC = ARG_TYPES.STATIC
TEMPORARY = ARG_TYPES.TEMPORARY
NA = 'Unavailable'

ARG_ALIAS: dict[str, list[str]] = {}


def add_arg_alias(alias: str | list[str], command: list[str]):
    """
    Add a new alias for running args.

    :param alias: The alias string(s).
    :param command: Commands, should be a list of strings.
    """
    if isinstance(alias, str):
        alias = [alias]

    for a in alias:
        ARG_ALIAS[a] = command


def parse_arg_alias(terminal_args: list[str] | None):
    """
    Parse arg alias from the terminal inputs.
    """
    if terminal_args is None:
        return None

    index = 0
    while index != len(terminal_args):
        item = terminal_args[index]
        if item in ARG_ALIAS.keys():
            terminal_args = (terminal_args[:index] +
                             ARG_ALIAS[item] +
                             terminal_args[index+1:])
        index += 1
    return terminal_args


class Args(ArgsManager):
    """
    A set of args used for training or evaluating prediction models.
    """

    def __init__(self, terminal_args: list[str] | None = None,
                 is_temporary=False) -> None:

        super().__init__(parse_arg_alias(terminal_args), is_temporary)

    def _init_all_args(self):
        super()._init_all_args()

        # Init split
        if self.force_split != 'null':
            if self.update_saved_args:
                self.log('Parameters cannot be saved when forced ' +
                         'parameters (`force_dataset`, `force_split`) ' +
                         'are set. Please remove them and run again.',
                         level='error', raiseError=ValueError)

            self._set('split', self.force_split)
        self._args_need_initialize.remove('split')

        # Init dataset
        if self.force_dataset != 'null':
            if self.update_saved_args:
                self.log('Parameters cannot be saved when forced ' +
                         'parameters (`force_dataset`, `force_split`) ' +
                         'are set. Please remove them and run again.',
                         level='error', raiseError=ValueError)

            self._set('dataset', self.force_dataset)

        # This argument can only be set manually by codes
        # or read from the saved JSON file
        elif 'dataset' not in self._args_load.keys():

            dirs = os.listdir(DATASET_CONFIG_DIR)

            split_list: list[tuple[str, str]] = []
            for d in dirs:
                try:
                    _path = os.path.join(DATASET_CONFIG_DIR, d)
                    for p in os.listdir(_path):
                        if p.endswith('.plist'):
                            split_list.append((d, p.split('.plist')[0]))
                except:
                    pass

            dataset = None
            for _dataset, _split in split_list:
                if self.split == _split:
                    dataset = _dataset
                    break

            if not dataset:
                self.log(f'Dataset of the specified split `{self.split}`' +
                         ' not found. Please check your spell.',
                         level='error', raiseError=ValueError)

            self._set('dataset', dataset)
        self._args_need_initialize.remove('dataset')

        # Init clip
        if self.draw_results != 'null':
            self._set('force_clip', self.draw_results)

        if self.draw_videos != 'null':
            self._set('force_clip', self.draw_videos)

        self._args_need_initialize.remove('force_clip')

        # Init test mode
        if self.draw_results != 'null' or self.draw_videos != 'null':
            self._set('test_mode', 'one')
        self._args_need_initialize.remove('test_mode')

        # Init logs paths
        if self._is_temporary:
            pass

        # This argument can only be set manually by codes
        # or read from the saved JSON file
        elif 'log_dir' not in self._args_load.keys():
            log_dir_current = (TIME +
                               ('' if self.model_name.startswith('_') else '_') +
                               self.model_name +
                               ('' if self.model_name.endswith('_') else '_') +
                               self.model +
                               self.split)

            default_log_dir = os.path.join(dir_check(self.save_base_dir),
                                           log_dir_current)

            self._set('log_dir', dir_check(default_log_dir))

        self._args_need_initialize.remove('log_dir')

        # Import visualization package
        if ((self.draw_results != 'null') or
                (self.draw_videos != 'null')):
            from qpid.mods import vis

        if self._verbose_mode:
            self.log('Training args initialized.')

    @property
    def help(self) -> str:
        """
        Print help information on the screen.
        """
        return self._arg('help', 'null', argtype=TEMPORARY, short_name='h')

    @property
    def experimental(self) -> bool:
        """
        NOTE: It is only used for code test.
        """
        return self._arg('experimental', False, argtype=TEMPORARY)

    @property
    def model_type(self) -> str:
        """
        Model type, canbe `'agent-based'` or `'frame-based'`.
        """
        return self._arg('model_type', 'agent-based', argtype=STATIC)

    @property
    def max_agents(self) -> int:
        """
        Max number of agents to predict in one frame.
        It only works when `model_type == 'frame-based'`.
        """
        return self._arg('max_agents', 50, argtype=STATIC)

    @property
    def batch_size(self) -> int:
        """
        Batch size when implementation.
        """
        return self._arg('batch_size', 5000, argtype=DYNAMIC, short_name='bs')

    @property
    def dataset(self) -> str:
        """
        Name of the video dataset to train or evaluate.
        For example, `'ETH-UCY'` or `'SDD'`.
        NOTE: DO NOT set this argument manually.
        """
        return self._arg('dataset', NA, argtype=STATIC,
                         need_initialize=True)

    @property
    def force_dataset(self) -> str:
        """
        Force test dataset (ignore the train/test split).
        It only works when `test_mode` has been set to `one`.
        """
        return self._arg('force_dataset', 'null', argtype=TEMPORARY)

    @property
    def split(self) -> str:
        """
        The dataset split that used to train and evaluate.
        """
        return self._arg('split', 'zara1', argtype=STATIC,
                         short_name='s',
                         need_initialize=True)

    @property
    def force_split(self) -> str:
        """
        Force test dataset (ignore the train/test split). 
        It only works when `test_mode` has been set to `one`.
        """
        return self._arg('force_split', 'null', argtype=TEMPORARY)

    @property
    def epochs(self) -> int:
        """
        Maximum training epochs.
        """
        return self._arg('epochs', 500, argtype=STATIC)

    @property
    def force_clip(self) -> str:
        """
        Force test video clip (ignore the train/test split).
        It only works when `test_mode` has been set to `one`. 
        """
        return self._arg('force_clip', 'null', argtype=TEMPORARY,
                         need_initialize=True)

    @property
    def gpu(self) -> str:
        """
        Speed up training or test if you have at least one NVidia GPU. 
        If you have no GPUs or want to run the code on your CPU, 
        please set it to `-1`.
        NOTE: It only supports training or testing on one GPU.
        """
        return self._arg('gpu', '0', argtype=TEMPORARY)

    @property
    def save_base_dir(self) -> str:
        """
        Base folder to save all running logs.
        """
        return self._arg('save_base_dir', DEFAULT_LOG_DIR, argtype=STATIC)

    @property
    def start_test_percent(self) -> float:
        """
        Set when (at which epoch) to start validation during training.
        The range of this arg should be `0 <= x <= 1`. 
        Validation may start at epoch
        `args.epochs * args.start_test_percent`.
        """
        return self._arg('start_test_percent', 0.0, argtype=TEMPORARY)

    @property
    def log_dir(self) -> str:
        """
        Folder to save training logs and model weights.
        Logs will save at `args.save_base_dir/current_model`.
        DO NOT change this arg manually. (You can still change
        the path by passing the `save_base_dir` arg.)
        """
        return self._arg('log_dir', NA, argtype=STATIC,
                         need_initialize=True)

    @property
    def compute_loss(self) -> int:
        """
        Controls whether compute losses when testing.
        """
        return self._arg('compute_loss', 0, argtype=TEMPORARY)

    @property
    def model(self) -> str:
        """
        The model type used to train or test.
        """
        return self._arg('model', 'none', argtype=STATIC)

    @property
    def model_name(self) -> str:
        """
        Customized model name.
        """
        return self._arg('model_name', 'model', argtype=STATIC)

    @property
    def restore(self) -> str:
        """
        Path to restore the pre-trained weights before training.
        It will not restore any weights if `args.restore == 'null'`.
        """
        return self._arg('restore', 'null', argtype=TEMPORARY)

    @property
    def test_step(self) -> int:
        """
        Epoch interval to run validation during training.
        """
        return self._arg('test_step', 1, argtype=TEMPORARY)

    """
    Trajectory Prediction Args
    """
    @property
    def obs_frames(self) -> int:
        """
        Observation frames for prediction.
        """
        return self._arg('obs_frames', 8, argtype=STATIC, short_name='obs')

    @property
    def pred_frames(self) -> int:
        """
        Prediction frames.
        """
        return self._arg('pred_frames', 12, argtype=STATIC, short_name='pred')

    @property
    def draw_results(self) -> str:
        """
        Controls whether to draw visualized results on video frames.
        Accept the name of one video clip. The codes will first try to
        load the video file according to the path saved in the `plist`
        file (saved in `dataset_configs` folder), and if it loads successfully
        it will draw the results on that video, otherwise it will draw results
        on a blank canvas. Note that `test_mode` will be set to `'one'` and
        `force_split` will be set to `draw_results` if `draw_results != 'null'`.
        """
        return self._arg('draw_results', 'null', argtype=TEMPORARY, short_name='dr')

    @property
    def draw_videos(self) -> str:
        """
        Controls whether draw visualized results on video frames and save as images.
        Accept the name of one video clip.
        The codes will first try to load the video according to the path
        saved in the `plist` file, and if successful it will draw the
        visualization on the video, otherwise it will draw on a blank canvas.
        Note that `test_mode` will be set to `'one'` and `force_split`
        will be set to `draw_videos` if `draw_videos != 'null'`.
        """
        return self._arg('draw_videos', 'null', argtype=TEMPORARY)

    @property
    def step(self) -> float:
        """
        Frame interval for sampling training data.
        """
        return self._arg('step', 1.0, argtype=DYNAMIC)

    @property
    def test_mode(self) -> str:
        """
        Test settings. It can be `'one'`, `'all'`, or `'mix'`.
        When setting it to `one`, it will test the model on the `args.force_split` only;
        When setting it to `all`, it will test on each of the test datasets in `args.split`;
        When setting it to `mix`, it will test on all test datasets in `args.split` together.
        """
        return self._arg('test_mode', 'mix', argtype=TEMPORARY,
                         need_initialize=True)

    @property
    def lr(self) -> float:
        """
        Learning rate.
        """
        return self._arg('lr', 0.001, argtype=STATIC, short_name='lr')

    @property
    def K(self) -> int:
        """
        The number of multiple generations when testing.
        This arg only works for multiple-generation models.
        """
        return self._arg('K', 20, argtype=DYNAMIC)

    @property
    def K_train(self) -> int:
        """
        The number of multiple generations when training.
        This arg only works for multiple-generation models.
        """
        return self._arg('K_train', 10, argtype=STATIC)

    @property
    def anntype(self) -> str:
        """
        Model's predicted annotation type.
        Can be `'coordinate'` or `'boundingbox'`.
        """
        return self._arg('anntype', 'coordinate', argtype=STATIC)

    @property
    def interval(self) -> float:
        """
        Time interval of each sampled trajectory point.
        """
        return self._arg('interval', 0.4, argtype=STATIC)

    @property
    def pmove(self) -> int:
        """
        (Pre/post-process Arg)
        Index of the reference point when moving trajectories.
        """
        return self._arg('pmove', -1, argtype=STATIC)

    @property
    def update_saved_args(self) -> int:
        """
        Choose whether to update (overwrite) the saved arg files or not.
        """
        return self._arg('update_saved_args', 0, argtype=TEMPORARY)

    @property
    def force_anntype(self) -> str:
        """
        Assign the prediction type.
        It is now only used for silverballers models that are trained
        with annotation type `coordinate` but want to test on datasets
        with annotation type `boundingbox`.
        """
        return self._arg('force_anntype', 'null', argtype=TEMPORARY)

    @property
    def macos(self) -> int:
        """
        (Experimental) Choose whether enable the `MPS (Metal Performance Shaders)`
        on Apple platforms (instead of running on CPUs).
        """
        return self._arg('macos', 0, argtype=TEMPORARY)

    @property
    def input_pred_steps(self) -> str:
        """
        Indices of future time steps that used as extra model inputs.
        It accepts a string that contains several integer numbers separated
        with `'_'`. For example, `'3_6_9'`.
        It will take the corresponding ground truth points as the input when 
        training the model, and take the first output of the former network
        as this input when testing the model.
        Set it to `'null'` to disable this extra model inputs.
        """
        return self._arg('input_pred_steps', 'null', argtype=STATIC)

    @property
    def output_pred_steps(self) -> str:
        """
        Indices of future time steps to be predicted.
        It accepts a string that contains several integer numbers separated
        with `'_'`. For example, `'3_6_9'`.
        Set it to `'all'` to predict points among all future steps.
        """
        return self._arg('output_pred_steps', 'all', argtype=STATIC,
                         other_names=['key_points'])

    @property
    def noise_depth(self) -> int:
        """
        Depth of the random noise vector.
        """
        return self._arg('noise_depth', 16, argtype=STATIC,
                         other_names=['depth'])

    @property
    def feature_dim(self) -> int:
        """
        Feature dimensions that are used in most layers.
        """
        return self._arg('feature_dim', 128, argtype=STATIC)

    @property
    def preprocess(self) -> str:
        """
        Controls whether to run any pre-process before the model inference.
        It accepts a 3-bit-like string value (like `'111'`):
        - The first bit: `MOVE` trajectories to (0, 0);
        - The second bit: re-`SCALE` trajectories;
        - The third bit: `ROTATE` trajectories.
        """
        return self._arg('preprocess', '100', argtype=STATIC)

    @property
    def auto_clear(self) -> int:
        """
        Controls whether to clear all other saved weights except for the best one.
        It performs similar as running `python scripts/clear.py --logs logs`.
        """
        return self._arg('auto_clear', 1, argtype=TEMPORARY)
