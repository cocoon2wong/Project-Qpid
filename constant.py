"""
@Author: Conghao Wong
@Date: 2022-11-23 18:01:16
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-08 10:53:06
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""


class ARG_TYPES():
    """
    Type names of all running args.
    """
    STATIC = 'static'
    DYNAMIC = 'dynamic'
    TEMPORARY = 'temporary'


class ANN_TYPES():
    """
    Type names of heterogeneous trajectories.
    """
    CO_2D = 'coordinate'
    CO_3D = '3Dcoordinate'
    BB_2D = 'boundingbox'
    BB_2D_R = 'boundingbox-rotate'
    BB_3D = '3Dboundingbox'
    BB_3D_R = '3Dboundingbox-rotate'
    SKE_3D_17 = '3Dskeleton-17'

    _CO_SERIES_2D = 'coordinate-series'


class INPUT_TYPES():
    """
    Type names of all kinds of model inputs.
    """
    OBSERVED_TRAJ = 'TRAJ'
    NEIGHBOR_TRAJ = 'TRAJ_NEIGHBOR'
    MAP = 'MAP'
    MAP_PARAS = 'MAP_PARAS'
    DESTINATION_TRAJ = 'DEST'
    GROUNDTRUTH_TRAJ = 'GT'
    SEG_MAP = 'SEG_MAP'
    SEG_MAP_PARAS = 'SEG_MAP_PARAS'


class OUTPUT_TYPES():
    """
    Type names of all kinds of model outputs.
    """
    PREDICTED_TRAJ = 'TRAJ_PREDICTED'


class PROCESS_TYPES():
    """
    Names of all pre-process and post-process methods.
    """
    MOVE = 'MOVE'
    ROTATE = 'ROTATE'
    SCALE = 'SCALE'
    UPSAMPLING = 'UPSAMPLING'


class INTERPOLATION_TYPES():
    """
    Names of all interpolation methods.
    """

    LINEAR = 'l'
    LINEAR_SPEED = 'speed'
    LINEAR_ACC = 'acc'
    NEWTON = 'newton'

    @classmethod
    def get_type(cls, s: str):
        for _s in [cls.LINEAR, cls.LINEAR_ACC,
                   cls.LINEAR_SPEED, cls.NEWTON]:
            if s.startswith(_s):
                return _s
        return None


class DATASET_CONFIGS():
    """
    Names of configs used in all dataset-config's plist files.
    """
    SEG_IMG = 'segmentation_image'
    RGB_IMG = 'rgb_image'


class STRUCTURE_STATUS():
    """
    Status of training structures.
    """
    TRAIN = 0
    TRAIN_WITH_SAVED_WEIGHTS = 1

    TEST = 10
    TEST_WITH_SAVED_WEIGHTS = 11

    @staticmethod
    def is_training(s: int):
        return True if s < 10 else False
