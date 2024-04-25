"""
@Author: Conghao Wong
@Date: 2022-06-22 09:58:48
@LastEditors: Conghao Wong
@LastEditTime: 2024-04-25 20:44:35
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.__root import BaseObject
from qpid.args import Args
from qpid.base import BaseManager
from qpid.constant import INPUT_TYPES, INTERPOLATION_TYPES
from qpid.model import Model
from qpid.training import Structure

from .__baseArgs import SilverballersArgs
from .__MKII_utils import SILVERBALLERS_DICT as SDICT
from .interpHandlers import DirectHandlerModel
from .interpHandlers.__baseInterpHandler import _BaseInterpHandlerModel


class SilverballersModel(Model):
    """
    SilverballersModel
    ---
    The two-stage silverballers model.
    NOTE: This model is typically used for testing, not training.
    The SilverballersModel itself is not trainable. It only serves 
    as a container that provides a running environment for different 
    agent and handler sub-networks.

    Member Managers
    ---
    - (Soft member) Stage-1 Subnetwork, type is `BaseAgentModel`
        or a subclass of it;
    - (Soft member) Stage-2 Subnetwork, type is `BaseHandlerModel`
        or a subclass of it.
    """

    def __init__(self, agent_model: Model,
                 handler_model: Model,
                 structure=None,
                 *args, **kwargs):

        super().__init__(structure, *args, **kwargs)

        # This model does not need any preprocess layers
        self.set_preprocess()

        # Args
        self.s_args = self.args.register_subargs(SilverballersArgs, 's_args')

        # Layers
        self.agent = agent_model
        self.handler = handler_model

        # Set model inputs
        self.a_types = self.agent.input_types
        self.h_types = self.handler.input_types
        self.h_types.remove(INPUT_TYPES.GROUNDTRUTH_TRAJ)

        # Make sure that observed trajectories are the first input in the list
        input_types = list(set(self.a_types + self.h_types))
        input_types.remove(INPUT_TYPES.OBSERVED_TRAJ)
        input_types = [INPUT_TYPES.OBSERVED_TRAJ] + input_types
        self.set_inputs(*input_types)

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        ######################
        # Stage-1 Subnetwork #
        ######################
        a_inputs = [self.get_input(inputs, d) for d in self.a_types]
        a_outputs = self.agent.implement(a_inputs)

        # Down sampling from K*Kc generations (if needed)
        if self.s_args.down_sampling_rate < 1.0:
            a_pred = a_outputs[0]
            K_current = a_pred.shape[-3]
            K_new = K_current * self.s_args.down_sampling_rate
            new_index = torch.randperm(K_current)[:int(K_new)]
            a_pred = a_pred[..., new_index, :, :]
            a_outputs[0] = a_pred

        ######################
        # Stage-2 Subnetwork #
        ######################
        h_inputs = [self.get_input(inputs, d) for d in self.h_types]
        h_outputs = self.handler.implement(h_inputs + [a_outputs[0]])

        if not training and (c := self.s_args.channel) != -1:
            h_outputs[0] = h_outputs[0][..., c, None, :, :]

        return [h_outputs[0]] + [a_outputs] + [h_outputs]

    def print_info(self, **kwargs):
        info = {'Stage-1 Subnetwork': f"'{self.agent.name}' from '{self.s_args.loada}'",
                'Stage-2 Subnetwork': f"'{self.handler.name}' from '{self.s_args.loadb}'"}

        self.print_parameters(**info)
        self.agent.print_info(**kwargs)
        self.handler.print_info(**kwargs)


class SilverballersMKII(Structure):
    """
    SilverballersMKII Structure
    ---
    Basic structure to run the `agent-handler` based silverballers model.
    NOTE: It is only used for TESTING silverballers models, not training.

    Member Managers
    ---
    - Stage-1 Subnetwork Manager, type is `BaseAgentStructure` or its subclass;
    - Stage-2 Subnetwork Manager, type is `BaseHandlerStructure` or its subclass;
    - All members from the `Structure`.
    """

    AGENT_STRUCTURE_TYPE = Structure
    HANDLER_STRUCTURE_TYPE = Structure

    is_trainable = False

    def __init__(self, args: list[str], manager: BaseManager | None = None, name='Train Manager'):

        temp_args = Args(args, is_temporary=True)
        min_args = temp_args.register_subargs(SilverballersArgs, 's_min_args')
        a_model_path = min_args.loada
        b_model_path = min_args.loadb

        # Init log-related functions
        BaseObject.__init__(self)

        # Check args
        if 'null' in [a_model_path, b_model_path]:
            self.log('`Agent` or `Handler` model not found!' +
                     ' Please specific their paths via `--loada` (`-la`)' +
                     ' or `--loadb` (`-lb`).',
                     level='error', raiseError=KeyError)

        # Assign the model type of the first-stage subnetwork
        _args_a = Args(is_temporary=True).load_args_from_json(a_model_path)
        a_m_type: type[Model] = SDICT.get_model(_args_a.model)
        a_s_type: type[Structure] = SDICT.get_structure(_args_a.model)

        # Assign the model type of the second-stage subnetwork
        interp_model = INTERPOLATION_TYPES.get_type(b_model_path)
        if interp_model is None:
            _args_b = Args(is_temporary=True).load_args_from_json(b_model_path)
            interp_model = _args_b.model

        h_m_type: type[Model] = SDICT.get_model(interp_model)
        h_s_type: type[Structure] = SDICT.get_structure(interp_model)

        # Assign types of all subnetworks
        self.agent_model_type = a_m_type
        self.handler_model_type = h_m_type
        if a_s_type:
            self.AGENT_STRUCTURE_TYPE = a_s_type

        if h_s_type:
            self.HANDLER_STRUCTURE_TYPE = h_s_type

        # Load basic args from the saved agent model
        agent_args = Args(args + ['--load', min_args.loada], is_temporary=True)

        manual_args = ['--split', str(agent_args.split),
                       '--anntype', str(agent_args.anntype),
                       '--obs_frames', str(agent_args.obs_frames),
                       '--pred_frames', str(agent_args.pred_frames),
                       '--interval', str(agent_args.interval),
                       '--model_type', str(agent_args.model_type)]

        # Assign args from the saved Agent-Model's args
        if temp_args.batch_size > agent_args.batch_size:
            manual_args += ['--batch_size', str(agent_args.batch_size)]

        # Init the structure
        # The above `temp_args` has become the `self.args`
        super().__init__(args + manual_args)
        self.s_args = self.args.register_subargs(SilverballersArgs, 's_args')

        # Config the second-stage handler model
        if issubclass(self.handler_model_type, _BaseInterpHandlerModel):
            handler_args = None
            need_load = False
        else:
            handler_args = args + ['--load', min_args.loadb]
            need_load = True

        # Create the first-stage agent model
        self.agent = self.AGENT_STRUCTURE_TYPE(agent_args, manager=self)
        self.agent.MODEL_TYPE = self.agent_model_type
        self.agent.create_model()
        self.agent.model.load_weights_from_logDir(min_args.loada)

        # Create the second-stage handler model
        min_args_h = Args(handler_args, is_temporary=True)
        self.handler = self.HANDLER_STRUCTURE_TYPE(min_args_h, manager=self)

        if agent_args.output_pred_steps == 'all':
            self.handler.MODEL_TYPE = DirectHandlerModel
            need_load = False
        else:
            self.handler.args._set(
                'input_pred_steps', agent_args.output_pred_steps)
            self.handler.MODEL_TYPE = self.handler_model_type

        self.handler.create_model(as_single_model=False)
        if need_load:
            self.handler.model.load_weights_from_logDir(min_args.loadb)

    def create_model(self, *args, **kwargs):
        self.model = SilverballersModel(agent_model=self.agent.model,
                                        handler_model=self.handler.model,
                                        structure=self,
                                        *args, **kwargs)

    def print_test_results(self, loss_dict: dict[str, float], **kwargs):
        super().print_test_results(loss_dict, **kwargs)
        self.log(f'Test with 1st sub-network `{self.s_args.loada}` ' +
                 f'and 2nd seb-network `{self.s_args.loadb}` done.')


SDICT.register(MKII=[SilverballersMKII, None])
