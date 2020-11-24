# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii
from .coco import COCOCrowdDataset as coco
from .coco import COCOCPDataset as cp_coco

from .crowdpose import CrowdPoseSkeletonDataset as crowd_sk_pose
