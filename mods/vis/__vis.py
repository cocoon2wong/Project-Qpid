"""
@Author: Conghao Wong
@Date: 2022-06-21 20:36:21
@LastEditors: Conghao Wong
@LastEditTime: 2024-11-07 21:21:56
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from ...base import BaseManager, SecondaryBar
from ...dataset import AnnotationManager, Clip, SplitManager
from ...dataset.__base import BaseInputObject
from ...dataset.agent_based import Agent
from ...model import Model
from ...training import Structure
from ...utils import dir_check, get_relative_path
from .__args import VisArgs
from .__helper import ADD, get_helper
from .settings import IF_PUT_TEXT_IN_VIDEOS

OBS_IMAGE = 'obs_small.png'
NEI_OBS_IMAGE = 'neighbor_small.png'
CURRENT_IMAGE = 'neighbor_current.png'
GT_IMAGE = 'gt_small.png'
PRED_IMAGE = 'pred_small.png'

OBS_IMAGE = get_relative_path(__file__, OBS_IMAGE)
NEI_OBS_IMAGE = get_relative_path(__file__, NEI_OBS_IMAGE)
CURRENT_IMAGE = get_relative_path(__file__, CURRENT_IMAGE)
GT_IMAGE = get_relative_path(__file__, GT_IMAGE)
PRED_IMAGE = get_relative_path(__file__, PRED_IMAGE)

DRAW_ON_VIDEO = 0
DRAW_ON_IMAGE = 1
DRAW_ON_EMPTY = 2


class Visualization(BaseManager):

    def __init__(self, manager: BaseManager,
                 dataset: str, clip: str,
                 name='Visualization Manager'):

        super().__init__(manager=manager, name=name)

        # For type hinting
        self.manager: Structure

        # Init vis-related args
        self.vis_args = self.args.register_subargs(VisArgs, __package__)

        # Get information of the video clip
        self.info: Clip = self.manager.split_manager.clips_dict[clip]

        # Try to open the video
        video_path = self.info.video_path
        vc = cv2.VideoCapture(video_path)
        self._vc = vc if vc.open(video_path) else None

        # Try to read the scene image
        try:
            img_path = self.info.other_files['rgb_image']
            self.scene_image = cv2.imread(img_path)
        except:
            self.scene_image = None

        # annotation helper
        self.helper = get_helper(self.args.anntype)

        # Read png files
        self.obs_file = cv2.imread(OBS_IMAGE, -1)
        self.neighbor_file = cv2.imread(NEI_OBS_IMAGE, -1)
        self.current_file = cv2.imread(CURRENT_IMAGE, -1)
        self.pred_file = cv2.imread(PRED_IMAGE, -1)
        self.gt_file = cv2.imread(GT_IMAGE, -1)

    @property
    def video_capture(self) -> cv2.VideoCapture | None:
        return self._vc

    @property
    def picker(self) -> AnnotationManager:
        return self.manager.get_member(AnnotationManager)

    @property
    def model(self) -> Model:
        return self.manager.model

    def run_commands(self, outputs: list[torch.Tensor]):
        # Make the dir to save images
        save_base_path = dir_check(self.args.log_dir) \
            if self.args.load == 'null' \
            else self.args.load

        img_dir = dir_check(os.path.join(save_base_path, 'VisualTrajs'))
        save_format = os.path.join(img_dir, self.info.clip_name + '_{}')

        # Unpack model outputs
        pred_all = outputs[0].numpy()
        traj_wise_outputs = dict([
            (key, outputs[i].numpy())
            for i, key in self.model.ext_traj_wise_outputs.items()])

        agent_wise_outputs = dict([
            (key, outputs[i].numpy())
            for i, key in self.model.ext_agent_wise_outputs.items()])

        if self.vis_args.draw_index == 'all':
            agent_indexes = list(range(len(pred_all)))
        else:
            _indexes = self.vis_args.draw_index.split('_')
            agent_indexes = [int(i) for i in _indexes]

        ex_types: list[str] = []
        if self.vis_args.draw_exclude_type != 'null':
            ex_types = self.vis_args.draw_exclude_type.split("_")

        self.log(f'Start saving images into `{img_dir}`...')
        for index in self.timebar(agent_indexes, 'Saving...'):
            # Write traj into the agent
            agent = self.manager.agent_manager.agents[index]
            agent.write_pred(pred_all[index])

            # Get extra model outputs
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

            self.draw(agent=agent,
                      frames=frames,
                      save_name=save_format.format(index),
                      save_as_images=save_image,
                      traj_wise_outputs=to,
                      agent_wise_outputs=ao)

        self.log(f'Prediction result images are saved at {img_dir}')

    def get_image(self, frame: int) -> np.ndarray:
        """
        Get a frame from the video

        :param frame: The frame number of the image.
        """
        if not self.video_capture:
            raise ValueError

        time = 1000 * frame / self.info.paras[1]
        self.video_capture.set(cv2.CAP_PROP_POS_MSEC, time - 1)
        _, f = self.video_capture.read()
        f = self.rescale(f)
        return f

    def get_text(self, frame: int, agent: BaseInputObject) -> list[str]:
        return [self.info.clip_name,
                f'frame: {str(frame).zfill(6)}',
                f'agent: {agent.id}',
                f'type: {agent.type}']

    def get_trajectories(self, agent: BaseInputObject, real2pixel=True):
        obs = agent.traj_masked
        pred = agent.pred_masked
        gt = agent.groundtruth_masked
        if isinstance(agent, Agent):
            ref = agent.traj[np.newaxis, -1:, :]
            nei = agent.traj_neighbor[:agent.neighbor_number] + ref
        else:
            nei = None

        if real2pixel:
            integer = True
            obs = self.real2pixel(obs, integer)
            pred = self.real2pixel(pred, integer)
            gt = self.real2pixel(gt, integer)
            if isinstance(agent, Agent) and nei is not None:
                nei = self.real2pixel(nei, integer=True)

        if pred is not None and pred.ndim == 2:
            pred = pred[np.newaxis]

        return obs, pred, gt, nei

    def pixel2real(self, pixel_pos: np.ndarray):
        """
        Transfer coordinates from pixel scale to the real scale.

        :param pixel_pos: Coordinates in pixels, shape = (..., 2).
        :return r: Coordinates in meters, shape = (..., 2).
        """
        scale = self.info.get_manager(SplitManager).scale / \
            self.info.get_manager(SplitManager).scale_vis
        weights = self.info.matrix

        w = [weights[0], weights[2]]
        b = [weights[1], weights[3]]

        real = pixel_pos / scale

        r = np.stack([
            (real[..., 0] - b[0]) / w[0],
            (real[..., 1] - b[1]) / w[1]
        ], axis=-1)
        return r

    def real2pixel(self, real_pos, integer=True):
        """
        Transfer coordinates from real scale to pixels.

        :param real_pos: Coordinates, shape = (n, 2) or (k, n, 2).
        :return pixel_pos: Coordinates in pixels.
        """
        scale = self.info.get_manager(SplitManager).scale / \
            self.info.get_manager(SplitManager).scale_vis
        weights = self.info.matrix

        if type(real_pos) == list:
            real_pos = np.array(real_pos)

        w = [weights[0], weights[2]]
        b = [weights[1], weights[3]]

        real = scale * real_pos
        real_2d = self.manager.get_member(AnnotationManager) \
            .target.get_coordinate_series(real)

        pixel = []
        for p in real_2d:
            pixel += [w[0] * p[..., 0] + b[0],
                      w[1] * p[..., 1] + b[1]]

        pixel = np.stack(pixel, axis=-1)

        if integer:
            pixel = pixel.astype(np.int32)
        return pixel

    def rescale(self, f: np.ndarray):
        if (s := self.info.get_manager(SplitManager).scale_vis) > 1:
            x, y = f.shape[:2]
            f = cv2.resize(f, (int(y/s), int(x/s)))
        return f

    def draw(self, agent: BaseInputObject,
             frames: list[int] | np.ndarray,
             save_name: str,
             save_name_with_frame=True,
             draw_with_plt=False,
             interp=True,
             save_as_images=False,
             traj_wise_outputs: dict = {},
             agent_wise_outputs: dict = {}):
        """
        Draw trajectories on the video.

        :param agent: The agent object (`Agent`) to visualize.
        :param frames: A list frames of current observation steps.
        :param save_name: The name to save the output video, which does not contain \
            the file format.
        :param save_name_with_frame: Choose whether to add the `frame_index` after \
            the file name. (For example, `'zara1_0.jpg'` -> `'zara1_0_70.jpg'`)
        :param draw_with_plt: Choose whether to draw with plt by default.
        :param interp: Choose whether to draw the full video or only
            draw on the sampled time steps.
        :param save_as_images: Choose if to save as an image or a video clip.
        :param traj_wise_outputs: Extra trajectory-wise model outputs.
        :param agent_wise_outputs: Extra agent-wise model outputs.
        """

        video_writer = None
        status: int = -1
        f_empty = None

        # Try obtaining the RGB image
        if self.video_capture is not None:
            f = self.get_image(frames[0])
            status = DRAW_ON_VIDEO
        elif self.scene_image is not None:
            f = np.array(self.scene_image).copy()
            f = self.rescale(f)
            status = DRAW_ON_IMAGE
        else:
            f = None
            status = DRAW_ON_EMPTY

        # Decide to draw on RGB images or plt canvas
        if (f is not None) and (not draw_with_plt):
            vis_func = self.vis
            text_func = self.text
            real2pixel = True
            f_empty = np.zeros((f.shape[0], f.shape[1], 4))
            if self.vis_args.draw_on_empty_canvas:
                f = 255 * np.ones_like(f)
        else:
            vis_func = self._visualization_plt
            text_func = self._put_text_plt
            real2pixel = False
            plt.figure()

        # Prepare trajectories
        sampled_frames: list[int] = list(agent.frames)
        obs_len = agent.obs_length
        obs, pred, gt, nei = self.get_trajectories(agent, real2pixel)

        # Interpolate frames
        if interp:
            frames = np.arange(frames[0], frames[-1]+1)

        # Vis on a single frame of image
        if save_as_images:
            start_frame = frames[0]

            # Draw trajectories
            f = vis_func(f, obs, gt, pred, neighbor=nei)

            # Put text (top-left)
            f = text_func(f, self.get_text(start_frame, agent))

            # Put trajectory-wise outputs
            if self.vis_args.draw_extra_outputs:
                # Put text (trajectory-wise)
                for index in range(len(pred)):
                    pos = pred[index, -1]
                    text = [f'{v[index]:.2f}' for (
                        k, v) in traj_wise_outputs.items()]
                    f = text_func(f, texts=text, x=pos[1], y=pos[0],
                                  font=cv2.FONT_HERSHEY_SIMPLEX,
                                  size=0.5, width=2, line_height=30,
                                  shadow_bias=1)

                # TODO: draw agent-wise outputs on images

            if save_name_with_frame:
                _name = save_name + f'_{start_frame}.jpg'
            else:
                _name = save_name

            dir_check(os.path.dirname(_name))
            if ((status in [DRAW_ON_IMAGE, DRAW_ON_VIDEO])
                    and (not draw_with_plt)):
                cv2.imwrite(_name, f)
            else:
                plt.savefig(_name)
                plt.close()
            return

        dir_check(os.path.dirname(save_name))
        video_shape = (f.shape[1], f.shape[0])
        video_writer = cv2.VideoWriter(save_name + '.mp4',
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       self.info.paras[1],
                                       video_shape)

        # Continue vis as a short video clip
        f_pred = vis_func(f_empty, pred=pred)
        f_others = np.zeros_like(f_pred)

        for frame in SecondaryBar(frames,
                                  manager=self,
                                  desc='Processing frames...'):
            # Get a new scene image
            if status == DRAW_ON_VIDEO:
                f = self.get_image(frame)
            elif status == DRAW_ON_IMAGE:
                f = self.scene_image.copy()
            elif status == DRAW_ON_EMPTY:
                f = None
            else:
                raise ValueError(status)

            if frame in sampled_frames:
                step = sampled_frames.index(frame)
                if step < obs_len:
                    start, end = [max(0, step-1), step+1]
                    f_others = vis_func(
                        f_others, obs=obs[start:end],
                        neighbor=nei[:, start:end] if nei is not None else None)
                else:
                    step -= obs_len
                    start, end = [max(0, step-1), step+1]
                    f_others = vis_func(f_others, gt=gt[start:end])

            # Draw predictions
            if frame > sampled_frames[obs_len-1]:
                f = vis_func(f, background=f_pred)

            # Draw observations and groundtruths
            f = vis_func(f, background=f_others)

            if IF_PUT_TEXT_IN_VIDEOS:
                f = text_func(f, self.get_text(frame, agent))

            video_writer.write(f)

    def vis(self, source: np.ndarray,
            obs=None, gt=None, pred=None,
            neighbor=None,
            background: np.ndarray | None = None):
        """
        Draw one agent's observations, predictions, and groundtruths.

        :param source: The image file.
        :param obs: (optional) The observations in *pixel* scale.
        :param gt: (optional) The ground truth in *pixel* scale.
        :param pred: (optional) The predictions in *pixel* scale,\
            shape = `(K, steps, dim)`.
        :param neighbor: (optional) The observed neighbors' positions\
             in *pixel* scale, shape = `(batch, dim)`.
        :param draw_distribution: Controls whether to draw as a distribution.
        :param alpha: The alpha channel coefficient.
        """
        f = np.zeros([source.shape[0], source.shape[1], 4])

        # draw neighbors' observed trajectories
        if neighbor is not None:
            f = self.helper.draw_traj(f, neighbor[..., -1:, :],
                                      self.current_file,
                                      draw_lines=False)

            neighbor = neighbor if self.vis_args.draw_full_neighbors \
                else neighbor[..., -1:, :]
            for nei in neighbor:
                f = self.helper.draw_traj(
                    f, nei, self.neighbor_file,
                    draw_lines=self.vis_args.draw_lines)

        # draw predicted trajectories
        if pred is not None:
            if self.vis_args.draw_distribution:
                alpha = 0.8 if not self.vis_args.draw_on_empty_canvas else 1.0
                f = self.helper.draw_dis(f, pred, alpha=alpha,
                                         steps=self.vis_args.distribution_steps)
            else:
                for pred_k in pred:
                    f = self.helper.draw_traj(
                        f, pred_k, self.pred_file,
                        draw_lines=self.vis_args.draw_lines,
                        color=255 * np.random.rand(3))

        # draw observed and groundtruth trajectories
        if obs is not None:
            if obs.ndim == 2:
                obs = obs[np.newaxis]

            for _obs in obs:
                f = self.helper.draw_traj(f, _obs, self.obs_file,
                                          draw_lines=self.vis_args.draw_lines)

        if gt is not None:
            if gt.ndim == 2:
                gt = gt[np.newaxis]

            for _gt in gt:
                f = self.helper.draw_traj(f, _gt, self.gt_file,
                                          draw_lines=self.vis_args.draw_lines)

        # draw the background image
        if background is not None:
            f = ADD(background, f, [f.shape[1]//2, f.shape[0]//2])

        # add the original image
        f = ADD(source, f, [f.shape[1]//2, f.shape[0]//2])
        return f

    def text(self, f: np.ndarray,
             texts: list[str],
             x: int = 10,
             y: int = 40,
             font: int = cv2.FONT_HERSHEY_COMPLEX,
             size: float = 0.9,
             width: int = 2,
             line_height: int = 30,
             shadow_bias: int = 3) -> np.ndarray:
        """
        Put text on one image
        """
        for index, text in enumerate(texts):
            f = cv2.putText(f, text,
                            org=(x + shadow_bias, y + index *
                                 line_height + shadow_bias),
                            fontFace=font,
                            fontScale=size,
                            color=(0, 0, 0),
                            thickness=width)

            f = cv2.putText(f, text,
                            org=(x, y + index * line_height),
                            fontFace=font,
                            fontScale=size,
                            color=(255, 255, 255),
                            thickness=width)

        return f

    def _visualization_plt(self, f,
                           obs=None, gt=None, pred=None,
                           neighbor=None,
                           **kwargs):
        """
        Vis with plt (It only support 2D coordinates now).

        :param f: (Useless in this method.)
        :param obs: Observations, shape = (..., 2)
        """

        _p = '-' if self.vis_args.draw_lines else ''

        # draw neighbors' trajectories
        if neighbor is not None:
            neighbor = neighbor[None] if neighbor.ndim == 2 else neighbor
            for nei in neighbor:
                plt.plot(nei[-1, 0], nei[-1, 1], _p + 'o',
                         color='darkorange', markersize=13)

                plt.plot(nei[:, 0], nei[:, 1], _p + 's', color='purple')

        # draw observations
        if obs is not None:
            obs = obs[None] if obs.ndim == 2 else obs
            for o in obs:
                plt.plot(o[:, 0], o[:, 1], _p + 's', color='cornflowerblue')

        if gt is not None:
            gt = gt[None] if gt.ndim == 2 else gt
            for g in gt:
                plt.plot(g[:, 0], g[:, 1], _p + 's', color='lightgreen')

        if pred is not None:
            pred = pred[None] if pred.ndim == 2 else pred

            # Draw as the trajectory distribution
            if self.vis_args.draw_distribution:
                f = self.helper.draw_dis(f, pred, alpha=1.0,
                                         steps=self.vis_args.distribution_steps,
                                         plt_mode=True)

            # Draw as multiple trajectories
            else:
                for p in pred:
                    plt.plot(p[:, 0], p[:, 1], _p + 's')

        plt.axis('equal')

    def _put_text_plt(self, f: np.ndarray, texts: list[str], *args, **kwargs):
        plt.title(', '.join(texts))


def __draw_single_boundingbox(source, box: np.ndarray, png_file,
                              color, width, alpha):
    """
    The shape of `box` is `(4)`.
    """
    (y1, x1, y2, x2) = box[:4]
    cv2.rectangle(img=source,
                  pt1=(x1, y1),
                  pt2=(x2, y2),
                  color=color,
                  thickness=width)
    return source


def __draw_traj_boundingboxes(source, trajs, png_file,
                              color, width, alpha):
    """
    The shape of `trajs` is `(steps, 4)`.
    """
    for box in trajs:
        source = __draw_single_boundingbox(
            source, box, png_file, color, width, alpha)

    # draw center point
    source[:, :, 3] = alpha * 255 * source[:, :, 0]/color[0]
    for box in trajs:
        source = ADD(source, png_file,
                     ((box[1]+box[3])//2, (box[0]+box[2])//2))

    return source
