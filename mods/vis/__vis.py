"""
@Author: Conghao Wong
@Date: 2022-06-21 20:36:21
@LastEditors: Conghao Wong
@LastEditTime: 2025-09-16 21:10:52
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from qpid.base import BaseManager, SecondaryBar
from qpid.dataset import AnnotationManager, Clip, SplitManager
from qpid.dataset.__base import BaseInputObject
from qpid.dataset.agent_based import Agent
from qpid.training import Structure
from qpid.utils import dir_check

from . import helpers
from .__args import VisArgs
from .settings import *


class Visualization(BaseManager):

    def __init__(self, manager: BaseManager,
                 dataset: str, clip: str,
                 name='Visualization Manager'):

        super().__init__(manager=manager, name=name)

        # For type hinting
        self.manager: Structure

        # Init vis-related args
        self.vis_args = self.args.register_subargs(
            VisArgs, 'Visualization Args')

        # Get information of the video clip
        self.info: Clip = self.manager.split_manager.clips_dict[clip]

        # Try to open the video
        video_path = self.info.video_path
        vc = cv2.VideoCapture(video_path)
        self.video_capture = vc if vc.open(video_path) else None

        # Try to read the scene image
        try:
            img_path = self.info.other_files['rgb_image']
            self.scene_image = cv2.imread(img_path)
        except:
            self.scene_image = None

        # Init the canvas manager and visualization helpers
        self.canvas_helper = helpers.get(
            anntype=self.args.anntype,
            video_status=self.video_capture,
            image_status=self.scene_image,
            force_using_plt=self.vis_args.draw_with_plt
        )(manager=self)

    def run_commands(self, outputs: list[torch.Tensor]):
        # Make the dir to save images
        save_base_path = dir_check(self.args.log_dir) \
            if self.args.load == 'null' \
            else self.args.load

        img_dir = dir_check(os.path.join(save_base_path, 'VisualTrajs'))
        save_format = os.path.join(img_dir, self.info.clip_name + '_{}')

        # Unpack model outputs
        pred_all = outputs[0].numpy()

        # Filter agents by their indices and types
        if self.vis_args.draw_index == 'all':
            agent_indexes = list(range(len(pred_all)))
        else:
            _indexes = self.vis_args.draw_index.split('_')
            agent_indexes = [int(i) for i in _indexes]

        ex_types: list[str] = []
        if self.vis_args.draw_exclude_type != 'null':
            ex_types = self.vis_args.draw_exclude_type.split("_")

        # Start visualizing
        self.log(f'Start saving images into `{img_dir}`...')
        for index in self.timebar(agent_indexes, 'Saving...'):
            # Write traj into the agent
            agent = self.manager.agent_manager.agents[index]
            agent.write_pred(pred_all[index])

            # choose to draw as a video or a single image
            if self.args.draw_videos != 'null':
                frames = agent.frames
            else:
                frames = agent.frames[self.args.obs_frames - 1]

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
                      draw_with_plt=self.vis_args.draw_with_plt)

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

    def get_static_image(self) -> np.ndarray:
        if self.scene_image is None:
            raise ValueError

        return self.rescale(self.scene_image.copy())

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
             frames: int | list[int] | np.ndarray,
             save_name: str,
             save_name_postfix=True,
             draw_with_plt: int | bool = False,
             interp=True) -> str:
        """
        Visualize forecasted trajectories.

        :param agent: The agent object (`Agent`) to visualize.
        :param frames: A list frames of current observation steps to be \
            visualized. Pass a list with only one element or a single int \
            value to visualize forecasted trajectories as an image file.
        :param save_name: The name to save the output video, which does not \
            contain the file format.
        :param save_name_postfix: Choose whether to add the `frame_index` \
            after the file name. (Like `'zara1_0.jpg'` -> `'zara1_0_70.jpg'`)
        :param draw_with_plt: Choose whether to draw with plt by default.
        :param interp: Choose whether to draw the full video or only
            draw on the sampled time steps.

        :return file_name: Name of the saved image (at the current moment).
        """

        # Prepare folder
        dir_check(os.path.dirname(save_name))

        # Try obtaining the RGB image
        # Decide to draw on RGB images or plt canvas
        status = -1
        f = None

        if not draw_with_plt:
            if self.video_capture is not None:
                status = DRAW_ON_VIDEO
                fps = self.info.paras[1]
            elif self.scene_image is not None:
                status = DRAW_ON_IMAGE
                fps = self.info.paras[1]
            else:
                draw_with_plt = True

            if not draw_with_plt:
                real2pixel = True

        if draw_with_plt:
            real2pixel = False
            status = DRAW_ON_PLTCANVAS
            fps = 1 / self.args.interval

        # Update the canvas helper
        canvas_type = helpers.get(
            anntype=self.args.anntype,
            video_status=self.video_capture,
            image_status=self.scene_image,
            force_using_plt=draw_with_plt,
        )

        if not isinstance(self.canvas_helper, canvas_type):
            self.canvas_helper = canvas_type(manager=self)

        # Init the canvas
        f = self.canvas_helper.init_canvas(init_image=self.get_static_image())

        # Get shape of the target video
        if isinstance(f, np.ndarray):
            video_shape = (f.shape[1], f.shape[0])
        elif isinstance(f, Figure):
            _shape = f.get_size_inches() * 100
            video_shape = (int(_shape[0]), int(_shape[1]))
        else:
            raise ValueError(f)

        # Prepare trajectories
        sampled_frames: list[int] = list(agent.frames)
        obs, pred, gt, nei = self.get_trajectories(agent, real2pixel)
        obs_len = agent.obs_length

        # Visualize as a single image
        if (isinstance(frames, (int, float)) or
                (getattr(frames, 'size', 0) == 1)):
            frames = [int(frames)]
            video_writer = None
            pred_colors = None

        # Open the video writer
        else:
            if status == DRAW_ON_PLTCANVAS:
                frames = frames[:obs_len]

            if pred is not None:
                pred_colors = 255 * np.random.rand(pred.shape[0], 3)

            video_writer = cv2.VideoWriter(
                save_name + '.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps=fps,
                frameSize=video_shape,
            )

        # Interpolate frames
        if interp and (status != DRAW_ON_PLTCANVAS):
            frames = np.arange(frames[0], frames[-1]+1)

        # Config the timebar
        desc = 'Processing frames...'
        if not self.bar:
            timebar = self.timebar(frames, desc)
        else:
            timebar = SecondaryBar(frames, manager=self, desc=desc)

        # Start visualizing
        vis_func = self.canvas_helper.vis
        text_func = self.canvas_helper.text

        for frame in timebar:
            # Get a new scene image
            if status == DRAW_ON_VIDEO:
                init_image = self.get_image(frame)
            elif status == DRAW_ON_IMAGE:
                init_image = self.get_static_image()
            elif status == DRAW_ON_PLTCANVAS:
                init_image = None
            else:
                raise ValueError(status)

            # Init canvas with the above image
            f = self.canvas_helper.init_canvas(init_image)

            # Draw trajectories step-by-step
            if frame in sampled_frames:
                step = sampled_frames.index(frame)
                obs_end = min(step, obs_len) + 1
                nei_end = min(step, obs_len + 1) + 1
                if nei_end > obs_len + 1:
                    nei_end = 0

                if step > obs_len and (gt is not None):
                    gt_end = step - obs_len + 1
                else:
                    gt_end = 0

            if nei_end > 0 and (nei is not None):
                f = vis_func(source=f, neighbor=nei[:, :nei_end])

            f = vis_func(source=f, obs=obs[:obs_end])

            # Draw predictions
            if frame >= sampled_frames[obs_len-1]:
                f = vis_func(source=f, pred=pred, pred_colors=pred_colors)

            # Draw ground truths
            # On a video
            if (gt_end > 0) and (gt is not None):
                f = vis_func(source=f, neighbor=gt[None, :step - obs_len + 1])
                f = vis_func(source=f, gt=gt[:step - obs_len + 1])

            # On a single image
            elif (len(frames) == 1) and (gt is not None):
                f = vis_func(source=f, gt=gt)

            if IF_PUT_TEXT_IN_VIDEOS:
                f = text_func(f, self.get_text(frame, agent))

            # Save as images (on sampled points)
            if frame in sampled_frames:
                name_postfix = f'_f{frame}.jpg' if save_name_postfix else ''
                saved_img_name = save_name + name_postfix
                if isinstance(f, np.ndarray):
                    cv2.imwrite(saved_img_name, f)

                elif isinstance(f, Figure):
                    f.savefig(saved_img_name)
                    plt.close(f)

            if video_writer is not None:
                if isinstance(f, Figure):
                    f = cv2.imread(saved_img_name)

                if f is not None:
                    video_writer.write(f)

        return saved_img_name
