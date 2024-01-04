"""
@Author: Conghao Wong
@Date: 2022-06-22 09:58:48
@LastEditors: Conghao Wong
@LastEditTime: 2024-01-04 16:42:47
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from ..__root import BaseObject
from ..args import Args
from ..constant import ANN_TYPES, INPUT_TYPES, INTERPOLATION_TYPES
from ..dataset import AnnotationManager
from ..model import Model
from ..training import Structure
from .__agentModel import AgentArgs, BaseAgentModel, BaseAgentStructure
from .__baseArgs import SilverballersArgs
from .__handlerModel import BaseHandlerModel, BaseHandlerStructure
from .__MKII_utils import SILVERBALLERS_DICT as SDICT


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

    def __init__(self, Args: SilverballersArgs,
                 agentModel: BaseAgentModel,
                 handlerModel: BaseHandlerModel,
                 structure,
                 *args, **kwargs):

        super().__init__(Args, structure, *args, **kwargs)

        # Type hints
        self.args: SilverballersArgs
        self.manager: SilverballersMKII
        self.structure: SilverballersMKII

        # Processes are applied in AgentModels and HandlerModels
        self.set_preprocess()

        # Layers
        self.agent = agentModel
        self.handler = handlerModel

        # Set model inputs
        a_type = self.agent.input_types
        h_type = self.handler.input_types[:-1]
        self.input_types = list(set(a_type + h_type))

        # Make sure that obs trajs are the first model input
        if ((t := INPUT_TYPES.OBSERVED_TRAJ) in self.input_types and
                self.input_types[0] != t):
            self.input_types.remove(t)
            self.input_types = [t] + self.input_types

        self.agent_input_index = self.get_input_index(a_type)
        self.handler_input_index = self.get_input_index(h_type)

        # set labels
        self.set_labels(INPUT_TYPES.GROUNDTRUTH_TRAJ)

        # Extra model outputs
        self.ext_traj_wise_outputs = self.handler.ext_traj_wise_outputs
        self.ext_agent_wise_outputs = self.handler.ext_agent_wise_outputs

    def get_input_index(self, input_type: list[str]):
        return [self.input_types.index(t) for t in input_type]

    def forward(self, inputs: list[torch.Tensor],
                training=None, mask=None,
                *args, **kwargs):

        # Prepare model inputs
        traj_index = self.agent_input_index[0]

        # Predict with `co2bb` (Coordinates to 2D bounding boxes)
        if self.args.force_anntype == ANN_TYPES.BB_2D and \
           self.agent.args.anntype == ANN_TYPES.CO_2D and \
           self.manager.split_manager.anntype == ANN_TYPES.BB_2D:

            # Flatten into a series of 2D points
            all_trajs = self.manager.get_member(AnnotationManager) \
                .target.get_coordinate_series(inputs[traj_index])

        else:
            all_trajs = [inputs[traj_index]]

        ######################
        # Stage-1 Subnetwork #
        ######################
        y_all_agent = []
        y_agent = []
        for traj in all_trajs:
            # Call the first stage model multiple times
            x_agent = [traj] + [inputs[i] for i in self.agent_input_index[1:]]
            y_all_agent.append(o := self.agent.implement(x_agent))
            y_agent.append(o[0])

        y_agent = torch.concat(y_agent, dim=-1)

        # Down sampling from K*Kc generations (if needed)
        if self.args.down_sampling_rate < 1.0:
            K_current = y_agent.shape[-3]
            K_new = K_current * self.args.down_sampling_rate
            new_index = torch.randperm(K_current)[:int(K_new)]
            y_agent = y_agent[..., new_index, :, :]

        ######################
        # Stage-2 Subnetwork #
        ######################
        x_handler = [inputs[i] for i in self.handler_input_index]
        x_handler.append(y_agent)
        y_handler = self.handler.implement(x_handler)

        if not training and (c := self.args.channel) != -1:
            y_handler[0] = y_handler[0][..., c, None, :, :]

        return [y_handler[0]] + y_all_agent + [y_handler]

    def print_info(self, **kwargs):
        info = {'Stage-1 Subnetwork': f"'{self.agent.name}' from '{self.structure.args.loada}'",
                'Stage-2 Subnetwork': f"'{self.handler.name}' from '{self.structure.args.loadb}'"}

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

    AGENT_STRUCTURE_TYPE = BaseAgentStructure
    HANDLER_STRUCTURE_TYPE = BaseHandlerStructure

    is_trainable = False

    def __init__(self, terminal_args: list[str]):

        min_args = SilverballersArgs(terminal_args, is_temporary=True)
        a_model_path = min_args.loada
        b_model_path = min_args.loadb

        # Assign the model type of the first-stage subnetwork
        _args_a = Args(is_temporary=True).load_args_from_json(a_model_path)
        a_m_type = SDICT.get_model(_args_a.model, BaseAgentModel)
        a_s_type = SDICT.get_structure(_args_a.model, BaseAgentStructure)

        # Assign the model type of the second-stage subnetwork
        interp_model = INTERPOLATION_TYPES.get_type(b_model_path)
        if interp_model is None:
            _args_b = Args(is_temporary=True).load_args_from_json(b_model_path)
            interp_model = _args_b.model

        h_m_type = SDICT.get_model(interp_model, BaseHandlerModel)
        h_s_type = SDICT.get_structure(interp_model, BaseHandlerStructure)

        # Assign types of all subnetworks
        self.agent_model_type = a_m_type
        self.handler_model_type = h_m_type
        if a_s_type:
            self.AGENT_STRUCTURE_TYPE = a_s_type

        if h_s_type:
            self.HANDLER_STRUCTURE_TYPE = h_s_type

        # Init log-related functions
        BaseObject.__init__(self)

        # Check structures' types
        if not issubclass(a_s_type, BaseAgentStructure):
            self.log(f'Structure `{a_s_type}` dose not support two-stage ' +
                     'MKII models. Try re-run using `python main.py --load ' +
                     f'{a_model_path}`.',
                     level='error', raiseError=ValueError)

        # Load minimal args
        min_args = SilverballersArgs(terminal_args, is_temporary=True)

        # Check args
        if 'null' in [min_args.loada, min_args.loadb]:
            self.log('`Agent` or `Handler` model not found!' +
                     ' Please specific their paths via `--loada` (`-la`)' +
                     ' or `--loadb` (`-lb`).',
                     level='error', raiseError=KeyError)

        # Load basic args from the saved agent model
        min_args_a = AgentArgs(terminal_args + ['--load', min_args.loada],
                               is_temporary=True)

        # Assign args from the saved Agent-Model's args
        extra_args: list[str] = []
        if min_args.batch_size > min_args_a.batch_size:
            extra_args += ['--batch_size', str(min_args_a.batch_size)]

        extra_args += ['--split', str(min_args_a.split),
                       '--anntype', str(min_args_a.anntype),
                       '--obs_frames', str(min_args_a.obs_frames),
                       '--pred_frames', str(min_args_a.pred_frames),
                       '--interval', str(min_args_a.interval),
                       '--model_type', str(min_args_a.model_type)]

        self.args: SilverballersArgs = SilverballersArgs(
            terminal_args + extra_args)

        if self.args.force_anntype != 'null':
            self.args._set('anntype', self.args.force_anntype)

        # init the structure
        super().__init__(self.args)

        if (k := '--force_anntype') in terminal_args:
            terminal_args.remove(k)

        # config second-stage model
        if self.handler_model_type.is_interp_handler:
            handler_args = None
        else:
            handler_args = terminal_args + ['--load', self.args.loadb]

        # First-stage subnetwork
        agent_args = self.AGENT_STRUCTURE_TYPE.ARG_TYPE(
            terminal_args + ['--load', self.args.loada],
            is_temporary=True)
        self.agent = self.AGENT_STRUCTURE_TYPE(agent_args, manager=self)
        self.agent.set_model_type(self.agent_model_type)
        self.agent.create_model()
        self.agent.model.load_weights_from_logDir(self.args.loada)

        # Second-stage subnetwork
        handler_args = self.HANDLER_STRUCTURE_TYPE.ARG_TYPE(handler_args,
                                                            is_temporary=True)
        handler_args._set('key_points', self.agent.args.key_points)
        self.handler = self.HANDLER_STRUCTURE_TYPE(handler_args, self)
        self.handler.set_model_type(self.handler_model_type)
        self.handler.create_model(as_single_model=False)
        if not self.handler_model_type.is_interp_handler:
            self.handler.model.load_weights_from_logDir(self.args.loadb)

    def create_model(self, *args, **kwargs):
        self.model = SilverballersModel(
            self.args,
            agentModel=self.agent.model,
            handlerModel=self.handler.model,
            structure=self,
            *args, **kwargs)

    def print_test_results(self, loss_dict: dict[str, float], **kwargs):
        super().print_test_results(loss_dict, **kwargs)
        self.log(f'Test with 1st sub-network `{self.args.loada}` ' +
                 f'and 2nd seb-network `{self.args.loadb}` done.')


SDICT.register(MKII=[SilverballersMKII, None])
