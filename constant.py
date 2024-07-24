"""
@Author: Conghao Wong
@Date: 2022-11-23 18:01:16
@LastEditors: Conghao Wong
@LastEditTime: 2024-07-24 09:23:13
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
    NEIGHBOR_GROUNDTRUTH_TRAJ = 'NEIGHBOR_GT_TRAJ'
    DESTINATION_TRAJ = 'DEST'
    GROUNDTRUTH_TRAJ = 'GT'
    GROUNDTRUTH_KEYPOINTS = 'GT_KEYPOINTS'
    LOSS_WEIGHT = 'LOSS_WEIGHT'


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
