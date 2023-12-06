"""
@Author: Conghao Wong
@Date: 2023-09-06 18:52:26
@LastEditors: Conghao Wong
@LastEditTime: 2023-12-06 16:04:54
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.args import Args
from qpid.training import Structure

from ..args import DYNAMIC, STATIC, TEMPORARY
from ..base import SecondaryBar
from ..constant import INPUT_TYPES
from ..dataset import AgentManager
from ..training import Structure, loss
from .__baseArgs import BaseSilverballersArgs, SilverballersArgs
from .__baseSubnetwork import BaseSubnetwork, BaseSubnetworkStructure
from .__loss import avgKey


class HandlerArgs(BaseSilverballersArgs):

    def __init__(self, terminal_args: list[str] | None = None,
                 is_temporary=False) -> None:

        super().__init__(terminal_args, is_temporary)

        self._set_default('key_points', 'null', overwrite=False)

    @property
    def points(self) -> int:
        """
        The number of keypoints accepted in the handler model.
        """
        return self._arg('points', 1, argtype=STATIC)


class BaseHandlerModel(BaseSubnetwork):

    is_interp_handler = False

    def __init__(self, Args: HandlerArgs,
                 as_single_model: bool = True,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # For type hinting
        self.args: HandlerArgs
        self.structure: BaseHandlerStructure

        # Configs
        # GT in the inputs is only used when training
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.MAP,
                        INPUT_TYPES.MAP_PARAS,
                        INPUT_TYPES.GROUNDTRUTH_TRAJ)

        # Keypoints and their indices
        self.points = self.args.points
        self.key_points = self.args.key_points
        self.accept_batchK_inputs = False

        self.ext_traj_wise_outputs[1] = 'Interaction Scores'

    def forward(self, inputs: list[torch.Tensor],
                keypoints: torch.Tensor,
                keypoints_index: torch.Tensor,
                training=None, mask=None):

        raise NotImplementedError

    def forward_as_handler(self, inputs: list[torch.Tensor],
                           keypoints: torch.Tensor,
                           keypoints_index: torch.Tensor,
                           training=None, mask=None):
        """
        Call as the second stage handler model.
        Do NOT call this method when training.

        :param inputs: a list of trajs and context maps
        :param keypoints: predicted keypoints, shape is `(batch, K, n_k, 2)`
        :param keypoints_index: index of predicted keypoints, shape is `(n_k)`
        """

        if not self.accept_batchK_inputs:
            p_all = []
            for k in SecondaryBar(range(keypoints.shape[1]),
                                  manager=self.structure.manager,
                                  desc='Running Stage-2 Sub-Network...',
                                  update_main_bar=True):

                # Run stage-2 network on a batch of inputs
                pred = self(inputs=inputs,
                            keypoints=keypoints[:, k, :, :],
                            keypoints_index=keypoints_index)

                if type(pred) not in [list, tuple]:
                    pred = [pred]

                # A single output shape is (batch, pred, dim).
                p_all.append(pred[0])

            return torch.permute(torch.stack(p_all), [1, 0, 2, 3])

        else:
            return self(inputs=inputs,
                        keypoints=keypoints,
                        keypoints_index=keypoints_index)

    def implement(self, inputs: list[torch.Tensor],
                  training=None,
                  *args, **kwargs):

        # Preprocess (include keypoints)
        keypoints = [inputs[-1]]
        inputs_p = self.processor(inputs, preprocess=True, training=training)
        keypoints_p = self.processor(keypoints, preprocess=True,
                                     update_paras=False,
                                     training=training)

        # only when training the single model
        device = inputs[0].device
        if self.as_single_model:
            gt_processed = keypoints_p[0]

            if self.key_points == 'null':
                index = torch.randperm(self.args.pred_frames-1, device=device)
                index = index[:self.points-1].sort()[0]
                index = torch.concat(
                    [index, torch.tensor([self.args.pred_frames-1])])

            else:
                index = self.key_indices_future

            points = gt_processed[:, index]
            outputs = self(inputs_p,
                           keypoints=points,
                           keypoints_index=index.to(torch.float32),
                           training=training)

        # Or use it as the second stage model
        else:
            outputs = self.forward_as_handler(
                inputs_p,
                keypoints=keypoints_p[0],
                keypoints_index=self.key_indices_future.to(
                    torch.float32).to(device),
                training=None)

        # Run post-process
        outputs_p = self.processor(outputs, preprocess=False,
                                   training=training)
        pred_o = outputs_p[0]

        # Calculate scores
        if (not training) and (
            (INPUT_TYPES.MAP in self.input_types)
            and (INPUT_TYPES.MAP_PARAS in self.input_types)
        ):

            from qpid.mods import contextMaps as maps
            map_mgr = self.get_top_manager().get_member(
                AgentManager).get_member(maps.MapParasManager)
            scores = map_mgr.score(trajs=outputs_p[0],
                                   maps=inputs[1],
                                   map_paras=inputs[2],
                                   centers=inputs[0][..., -1, :])

            # Pick trajectories
            # Only work when it play as the subnetwork
            if not self.as_single_model:
                run_args: SilverballersArgs = self.get_top_manager().args
                if (p := run_args.pick_trajectories) < 1.0:
                    pred_o = map_mgr.pick_trajectories(pred_o, scores, p)

            return [pred_o, scores] + outputs_p[1:]

        else:
            return outputs_p

    def print_info(self, **kwargs):
        info = {'Keypoints and Transforms': None,
                '- Transform type': self.args.T,
                '- Number of keypoints': self.args.points}

        return super().print_info(**kwargs, **info)


class BaseHandlerStructure(BaseSubnetworkStructure):

    ARG_TYPE: type[HandlerArgs] = HandlerArgs
    MODEL_TYPE: type[BaseHandlerModel] | None = None

    def __init__(self, terminal_args: list[str] | Args,
                 manager: Structure | None = None):

        super().__init__(terminal_args, manager)

        # For type hinting
        self.args: HandlerArgs
        self.model: BaseHandlerModel

        # Configs, losses, and metrics
        self.set_labels(INPUT_TYPES.GROUNDTRUTH_TRAJ)
        self.loss.set({loss.l2: 1.0})

        if self.args.key_points == 'null':
            self.metrics.set({loss.ADE: 1.0,
                              loss.FDE: 0.0})
        else:
            self.metrics.set({loss.ADE: 1.0,
                              loss.FDE: 0.0,
                              avgKey: 0.0})
