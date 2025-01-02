"""
@Author: Conghao Wong
@Date: 2022-06-22 09:58:48
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-09 19:15:02
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

import torch

import qpid
from qpid.args import Args
from qpid.base import BaseManager
from qpid.constant import INPUT_TYPES
from qpid.model import Model
from qpid.training import Structure

from .__baseArgs import SilverballersArgs


class SilverballersModel(Model):
    """
    SilverballersModel
    ---
    The container of multi-stage trajectory prediction models.
    NOTE: This model is typically used for testing, not training.
    The SilverballersModel itself is not trainable. It only serves 
    as a container that provides a running environment for different 
    subnetworks with different prediction stages.
    """

    def __init__(self, substructures: list[Structure],
                 structure=None, *args, **kwargs):
        """
        :param substructures: A list of training structures of subnetworks.
        """
        super().__init__(structure, *args, **kwargs)

        # Init containers
        input_types = []
        self._subnetwork_count = 0

        # Args
        self.s_args = self.args.register_subargs(SilverballersArgs, 's_args')

        # This model does not need any preprocess layers
        self.set_preprocess()

        # Assign each subnetwork
        for s in substructures:
            net = s.model
            net.as_single_model = False

            # Remove the type groundtruth from model inputs
            if (gt := INPUT_TYPES.GROUNDTRUTH_TRAJ) in net.input_types:
                net.input_types.remove(gt)
            input_types += net.input_types

            self.add_subnetwork(net)

        # Set input types
        input_types = list(set(input_types))
        self.set_inputs(*input_types)

    @property
    def subnetworks(self) -> dict[int, Model]:
        res = {}
        for i in range(self._subnetwork_count):
            try:
                net = self.get_submodule(f'subnetwork_{i}')
                res[i] = net
            except:
                break
        return res

    def add_subnetwork(self, net: Model):
        self.add_module(f'subnetwork_{self._subnetwork_count}', net)
        self._subnetwork_count += 1

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        model_outputs: dict[int, list] = {}
        for index, net in self.subnetworks.items():
            _inputs = [self.get_input(inputs, d) for d in net.input_types]
            if (index_last := index - 1) in model_outputs.keys():
                keypoints = model_outputs[index_last][0]
                _inputs = _inputs + [keypoints]

            _outputs = net.implement(_inputs)

            # Down sampling from K*Kc generations (if needed)
            if self.s_args.down_sampling_rate < 1.0:
                _pred = _outputs[0]
                K_current = _pred.shape[-3]
                K_new = K_current * self.s_args.down_sampling_rate
                new_index = torch.randperm(K_current)[:int(K_new)]
                _pred = _pred[..., new_index, :, :]
                _outputs[0] = _pred

            # Save outputs
            model_outputs[index] = _outputs

        return [_outputs[0]] + list(model_outputs.values())

    def print_info(self, **kwargs):
        info = {}
        for index, net in self.subnetworks.items():
            info[f'Network {index}'] = net.name
        super().print_info(**info, **kwargs)
        for net in self.subnetworks.values():
            net.print_info(**kwargs)


class SilverballersMKII(Structure):
    """
    SilverballersMKII Structure
    ---
    Basic structure to run the multi-stage `SilverballersModel` model.
    NOTE: It is only used for TESTING silverballers models, not training.
    """

    is_trainable = False

    def __init__(self, args: list[str],
                 manager: BaseManager | None = None,
                 name='Train Manager'):

        # Args
        main_args = Args(args)
        mkii_args = main_args.register_subargs(SilverballersArgs, 'mkii_args')

        # Init log-related functions
        BaseManager.__init__(self)

        # Structures of all subnetworks
        self.substructures: list[Structure] = []

        # Check args
        if 'null' in mkii_args.loads:
            self.log('Model not found!',
                     level='error', raiseError=KeyError)

        for path in mkii_args.loads.split(','):
            # Init model and training structure of the subnetwork
            if not os.path.exists(path):
                _args = Args(args)
                _load = False
                try:
                    s_type = qpid.get_structure(path)
                except NotImplementedError as e:
                    self.log(f'Weights `{path}` does not exist.' +
                             ' Please check your spell.',
                             level='error', raiseError=type(e))
            else:
                _args = Args(['--load', path] + args)
                _load = True
                s_type = qpid.get_structure(_args.model)

            if not s_type.MODEL_TYPE:
                s_type.MODEL_TYPE = qpid.get_model(path)

            # Set force args
            if len(self.substructures) and (s_last := self.substructures[-1]):
                _args_last = s_last.args
            else:
                _args_last = _args

            self.set_args(_args, ref_args=_args_last)

            if len(self.substructures):
                _args._set('input_pred_steps', _args_last.output_pred_steps)

            # Create model (subnetwork)
            s = s_type(_args, manager=self)
            s.create_model()
            if _load:
                s.model.load(path, device=self.device_cpu)

            self.substructures.append(s)

            if ((len(s.model.output_pred_steps) == s.args.pred_frames) or
                (s.model.as_final_stage_model)):
                break

        self.set_args(main_args, ref_args=self.substructures[0].args)
        super().__init__(main_args)

    def set_args(self, args: Args, ref_args: Args):
        args._set('dataset', ref_args.dataset)
        args._set('split', ref_args.split)
        args._set('anntype', ref_args.anntype)
        args._set('obs_frames', ref_args.obs_frames)
        args._set('pred_frames', ref_args.pred_frames)
        args._set('interval', ref_args.interval)
        args._set('batch_size', ref_args.batch_size)
        args._set('model_type', ref_args.model_type)

    def create_model(self, *args, **kwargs):
        self.model = SilverballersModel(self.substructures, self)

    def print_test_results(self, loss_dict: dict[str, float], **kwargs):
        super().print_test_results(loss_dict, **kwargs)
        self.log(f'Test with subnetworks ' +
                 f'{[s.args.load for s in self.substructures]} done.')
