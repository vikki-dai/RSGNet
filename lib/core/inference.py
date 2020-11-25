# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch

from utils.transforms import transform_preds
from numpy.linalg import LinAlgError
import torch.nn.functional as F
import cv2

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]  # batch_heatmaps: (batch_size, 17, 64, 48)
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))  # (64, 17, 3072)
    idx = np.argmax(heatmaps_reshaped, 2)  # return the index (64, 17)
    maxvals = np.amax(heatmaps_reshaped, 2)  # return the max values (64, 17)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))  # (64, 17, 1)
    idx = idx.reshape((batch_size, num_joints, 1))  # (64, 17, 1)

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals  # preds:(64, 17, 2)  maxvals (64, 17, 1)


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)  # coords:(64, 17, 2)  maxvals (64, 17, 1) batch_heatmaps: (batch_size, 17, 64, 48)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]  # (64,48)
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()  # preds:(64, 17, 2), the coordinate of heatmaps

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals # preds:(64, 17, 2), the coordinate of intial images


def softmax(logits):
    # logits :C,H, W
    max_logits = np.max(logits,axis=0)[np.newaxis]
    logits = logits - np.repeat(max_logits, logits.shape[0], axis=0)
    logits = np.exp(logits)
    sum_logits = np.sum(logits,axis=0)[np.newaxis]
    probs = logits / (np.repeat(sum_logits, logits.shape[0], axis=0))
    return probs