"""
@Author: Conghao Wong
@Date: 2022-07-19 11:19:58
@LastEditors: Conghao Wong
@LastEditTime: 2024-03-07 15:47:41
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

from ..base import BaseManager
from ..utils import DATASET_CONFIG_DIR, ROOT_TEMP_DIR, load_from_plist


class Clip(BaseManager):
    """
    Clip
    ---
    Base structure for controlling each video dataset.
    It is managed by the `SplitManager` object.

    Properties
    -----------------
    ```python
    >>> self.dataset        # name of the dataset
    >>> self.clip_name      # video clip name
    >>> self.annpath        # dataset annotation file
    >>> self.order          # X-Y order in the annotation file
    >>> self.paras          # [sample_step, frame_rate]
    >>> self.video_path     # video path   
    >>> self.matrix         # transfer matrix from real scales to pixels
    >>> self.other_files    # paths of all other dataset files
    ```
    """

    # Saving paths
    CONFIG_FILE = os.path.join(DATASET_CONFIG_DIR, '{}',
                               'subsets', '{}.plist')

    def __init__(self, manager: BaseManager, clip_name: str):

        super().__init__(manager=manager,
                         name=f'Video Clip Manager ({clip_name})')

        # init paths and folders
        dataset = self.get_manager(SplitManager).dataset_name
        self.CONFIG_FILE = self.CONFIG_FILE.format(dataset, clip_name)

        try:
            dic = load_from_plist(self.CONFIG_FILE)
        except:
            self.log(f'Clip `{self.CONFIG_FILE}` NOT FOUND.', level='error')
            raise FileNotFoundError

        self.__clip_name = clip_name
        self.__annpath = dic['annpath']
        self.__order = dic['order']
        self.__paras = dic['paras']
        self.__video_path = dic['video_path']
        self.__matrix = dic['matrix']
        self.__dataset = dic['dataset']

        if (k := 'other_files') in dic.keys():
            self.__other_files = dic[k]
        else:
            self.__other_files = {}

        # init temp path
        self.root_dir = os.path.dirname(self.annpath)
        self.temp_dir = os.path.join(
            ROOT_TEMP_DIR, self.dataset, self.clip_name)

    @property
    def dataset(self) -> str:
        """
        Name of the dataset.
        """
        return self.__dataset

    @property
    def clip_name(self):
        """
        Name of the video clip.
        """
        return self.__clip_name

    @property
    def annpath(self) -> str:
        """
        Path of the annotation file. 
        """
        return self.__annpath

    @property
    def other_files(self) -> dict[str, str]:
        """
        Paths of all other dataset files.
        Available keys:
          - `'rgb_image'`: Path of the scene RGB image;
          - `'segmentation_image'`: Path of the scene seg image.
        """
        return self.__other_files

    @property
    def order(self) -> list[int]:
        """
        X-Y order in the original annotation file.
        This property is deprecated now, and will be removed in the future. 
        """
        return self.__order

    @property
    def paras(self) -> tuple[int, int]:
        """
        [sample_step, frame_rate]
        """
        return self.__paras

    @property
    def video_path(self) -> str:
        """
        Path of the video file.
        """
        return self.__video_path

    @property
    def matrix(self) -> list[float]:
        """
        transfer weights from real scales to pixels.
        """
        return self.__matrix


class SplitManager(BaseManager):
    """
    SplitManager
    ---
    Manage a split of a full trajectory prediction dataset.
    A dataset split may contain several video clips.
    The `SplitManager` is managed by the `AgentManager` object.

    Properties
    ---
    ```python
    # Training clips
    (property) train_clips: (self: Self@SplitManager) -> list[Clip]

    # Test clips
    (property) test_clips: (self: Self@SplitManager) -> list[Clip]

    # Val clips
    (property) val_clips: (self: Self@SplitManager) -> list[Clip]

    # Name of the video dataset
    (property) dataset_name: (self: Self@SplitManager) -> str

    # Annotation type of the dataset
    (property) type: (self: Self@SplitManager) -> str

    # Global data scaling scale
    (property) scale: (self: Self@SplitManager) -> float

    # Video scaling when saving visualized results
    (property) scale_vis: (self: Self@SplitManager) -> float

    # Maximum dimension of trajectories recorded in this dataset
    (property) dimension: (self: Self@SplitManager) -> int

    # Type of annotations
    (property) anntype: (self: Self@SplitManager) -> str
    ```
    """

    # Saving paths
    CONFIG_FILE = os.path.join(DATASET_CONFIG_DIR, '{}', '{}.plist')

    def __init__(self, manager: BaseManager,
                 dataset: str, split: str,
                 name='Split Manager'):
        """
        :param manager: Manager object of this `SplitManager` object.
        :param dataset: The name of the image dataset.
        :param split: The split name of the dataset.
        :param name: Name of this manager object.
        """
        super().__init__(manager=manager, name=name)

        # init paths and folders
        self.CONFIG_FILE = self.CONFIG_FILE.format(dataset, split)

        try:
            dic = load_from_plist(self.CONFIG_FILE)
        except:
            self.log(f'Dataset `{self.CONFIG_FILE}` NOT FOUND.', level='error')
            raise FileNotFoundError

        self.__ds_name = dic['dataset']
        self.__type = dic['type']
        self.__scale = dic['scale']
        self.__scale_vis = dic['scale_vis']
        self.__dimension = dic['dimension']
        self.__anntype = dic['anntype']

        self.split: str = split
        self.train_sets: list[str] = dic['train']
        self.test_sets: list[str] = dic['test']
        self.val_sets: list[str] = dic['val']

        # init clips
        self.clips_dict: dict[str, Clip] = {}
        all_sets = self.train_sets + self.test_sets + self.val_sets
        for _set in set(all_sets):
            self.clips_dict[_set] = Clip(self, _set)

    @property
    def train_clips(self) -> list[Clip]:
        return [self.clips_dict[s] for s in self.train_sets]

    @property
    def test_clips(self) -> list[Clip]:
        return [self.clips_dict[s] for s in self.test_sets]

    @property
    def val_clips(self) -> list[Clip]:
        return [self.clips_dict[s] for s in self.val_sets]

    @property
    def dataset_name(self) -> str:
        """
        Name of the video dataset.
        For example, `ETH-UCY` or `SDD`.
        """
        return self.__ds_name

    @property
    def type(self) -> str:
        """
        Annotation type of the dataset.
        For example, `'pixel'` or `'meter'`.
        """
        return self.__type

    @property
    def scale(self) -> float:
        """
        Global data scaling scale.
        """
        return self.__scale

    @property
    def scale_vis(self) -> float:
        """
        Video scaling when saving visualized results.
        """
        return self.__scale_vis

    @property
    def dimension(self) -> int:
        """
        Maximum dimension of trajectories recorded in this dataset.
        For example, `(x, y)` -> `dimension = 2`.
        """
        return self.__dimension

    @property
    def anntype(self) -> str:
        """
        Type of annotations.
        For example, `'coordinate'` or `'boundingbox'`.
        """
        return self.__anntype
