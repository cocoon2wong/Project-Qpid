---
layout: page
title: Dataset and Split Notes
table-of-contents: true
---
<!--
 * @Author: Conghao Wong
 * @Date: 2023-04-11 20:48:08
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2025-05-19 16:49:57
 * @Description: file content
 * @Github: https://cocoon2wong.github.io
 * Copyright 2023 Conghao Wong, All Rights Reserved.
-->

This page describes how the `qpid` package split datasets, like the interval of frames, lengths of predictions, etc.

> [!NOTE]
> Similar to other trajectory prediction approaches, the validation sets (those sets that should be tested during training) are actually equals to the final test sets.
> Thus, clips listed in the `val sets` are not used during the whole training process.

## ETH-UCY and Stanford Drone Dataset

When making dataset files, we process the ETH-UCY dataset files by the `leave-one-out` strategy.
For the SDD, we split video clips with the dataset split method from [SimAug](https://github.com/JunweiLiang/Multiverse) (`36 train sets` + `12 test sets` + `12 val sets`).
Especially, following other researchers, the `univ13` split is not split exactly according to the literal *leave-one-out* approach.
The `univ13` split (ETH-UCY) takes `univ` and `univ3` as test sets, and other sets {`eth`, `hotel`, `unive`, `zara1`, `zara2`, `zara3`} as training sets.
Differently, the `univ` split only includes `univ` for testing models.

> [!NOTE]
> Please note that we do not use dataset split files like TrajNet on ETH-UCY and SDD for several reasons.
> For example, the frame rate problem in `ETH-eth` sub-dataset, and some of these splits only consider the `pedestrians` in the SDD dataset.
> We process the original full-dataset files from these datasets with observations = 3.2 seconds (or 8 frames) and predictions = 4.8 seconds (or 12 frames) to train and test the model.
> This means that our used `eth` subset in `ETU-UCY` is sampled with an interval of `6` observation frames, similar to the SR-LSTM (CVPR2019).
> See details in [this issue](https://github.com/cocoon2wong/Vertical/issues/1).

## NBA SportVU

Following most previous works, we only sample about 50K samples on this dataset.
Thus, the corresponding split name has become `nba50k`.
Among the 50K samples,

- `65% Train sets`: About 32.5K training samples;
- `25% Test sets`: About 12.5K test samples;
- `10% Val sets`: About 0.5K samples.

> [!NOTE]
> This dataset is labeled in `foot`.
> Please manually scale them to `meter` values when comparing results.

## nuScenes

Since the official test set in the nuScenes dataset has no labels, we use the following method to split the training set:

- `550 Train sets`: Randomly sampling 550 video clips from the official 850 training sets;
- `150 Val sets`: Randomly sampling the other 150 video clips from the official 850 training sets;
- `150 Test sets`: We treat the official 150 val sets as our test sets.

> [!NOTE]
> Results reported in our papers are trained/tested with the only-vehicle split `nuScenes_ov_v1.0`.
> Please uncomment the corresponding lines in `main_nuscenes.py` to make them manually, or just download them from our [provided page](../create-processed-dataset-files).

## Human3.6M

Following previous settings, we use all data from subjects `[1, 6, 7, 8, 9]` to train the model, subjects `[11]` to validate, and subjects `[5]` to test.

When making the training samples, we sample observations with the frequency of `25Hz` (*i.e.*, the sample interval is `40ms`) and use `t_h = 10` frames (`400ms`) of observations from all subjects to predict their possible trajectories for the next `t_f = 10` frames (`400ms`).
