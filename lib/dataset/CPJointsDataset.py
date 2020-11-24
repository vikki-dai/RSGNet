# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
import pycocotools._mask as _mask


logger = logging.getLogger(__name__)


class CPJointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''


        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('ori_img.jpg', data_numpy[:, :, ::-1])
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']
        if 'interference' in db_rec.keys():
            interference_joints = db_rec['interference']
            interference_joints_vis = db_rec['interference_vis']
        else:
            interference_joints = [joints]
            interference_joints_vis = [joints_vis]

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1
                for i in range(len(interference_joints)):
                    interference_joints[i], interference_joints_vis[i] = fliplr_joints(
                    interference_joints[i], interference_joints_vis[i], data_numpy.shape[1], self.flip_pairs)

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        # cv2.imwrite('img.jpg',input[:,:,::-1])
        if self.transform:
            input = self.transform(input)


        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)
        # interference joints heatmaps
        inter_target = np.zeros_like(target)
        inter_target_weight = np.zeros_like(target_weight)
        for i in range(len(interference_joints)):
            inter_joints = interference_joints[i]
            inter_joints_vis = interference_joints_vis[i]
            for j in range(self.num_joints):
                if inter_joints_vis[j, 0] > 0.0:
                    inter_joints[j, 0:2] = affine_transform(inter_joints[j, 0:2], trans)
            _inter_target, _inter_target_weight = self.generate_target(inter_joints, inter_joints_vis)
            inter_target = np.maximum(inter_target, _inter_target)
            inter_target_weight = np.maximum(inter_target_weight, _inter_target_weight)
        inter_target = np.maximum(inter_target, target)
        inter_target_weight = np.maximum(inter_target_weight, target_weight)
        # cv2.imwrite('heatmap.jpg',np.max(target,axis=0)*255)
        # cv2.imwrite('inter_heatmap.jpg', np.max(inter_target, axis=0) * 255)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)
        inter_target = torch.from_numpy(inter_target)
        inter_target_weight = torch.from_numpy(inter_target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }
        return input, target, target_weight, meta
        # return input, target, target_weight, inter_target, inter_target_weight, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight


class CPRelationJointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        # print('test_image_file', image_file)
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )

        if self.color_rgb:
            # print('test_image', data_numpy.shape)
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('/home/daiyan/code/deep-high-resolution-net/ori_img.jpg', data_numpy[:, :, ::-1])
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']
        if 'interference' in db_rec.keys():
            interference_joints = db_rec['interference']
            interference_joints_vis = db_rec['interference_vis']
        else:
            interference_joints = [joints]
            interference_joints_vis = [joints_vis]

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1


        # daiyan target person mask
        # self.MASK_ON = False
        # height, width = data_numpy.shape[:2]
        # if self.MASK_ON:
        #     tmp_mask = db_rec['mask']
        #     rle_mask = _mask.frPyObjects(tmp_mask, height, width)
        #     if type(rle_mask) == list:
        #         mask_numpy = _mask.decode(rle_mask)
        #     else:
        #         mask_numpy = _mask.decode([rle_mask])[:, :, 0]
        #     mask_numpy = np.array(mask_numpy, dtype=np.float32)
        #     mask_numpy = np.sum(mask_numpy, axis=2)  # connect the divided part
        #     mask_numpy = np.array(mask_numpy > 0, dtype=np.float32)
        # else:
        #     mask_numpy = np.zeros((height, width), dtype=np.float32)


        # size = db_rec['obj_size']
        # size = db_rec['obj_size']
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0
            # c_mask = c
            # s_mask = s
            # r_mask = r

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                # mask_numpy = mask_numpy[:, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1
                # c_mask[0] = mask_numpy.shape[1]-c_mask[0]-1
                for i in range(len(interference_joints)):
                    interference_joints[i], interference_joints_vis[i] = fliplr_joints(
                    interference_joints[i], interference_joints_vis[i], data_numpy.shape[1], self.flip_pairs)

        trans = get_affine_transform(c, s, r, self.image_size)
        # trans_mask = get_affine_transform(c_mask, s_mask, r_mask, self.image_size)


        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        # input_copy = np.copy(input)
        # cv2.imwrite('/home/daiyan/code/deep-high-resolution-net/img.jpg',input[:,:,::-1])

        # target_mask = cv2.warpAffine(
        #     mask_numpy,
        #     trans,
        #     (int(self.image_size[0]), int(self.image_size[1])),
        #     flags=cv2.INTER_LINEAR)
        # cv2.imwrite('/home/daiyan/code/deep-high-resolution-net/mask.jpg', 255*target_mask)
        # target_mask = cv2.resize(target_mask, (self.heatmap_size[0], self.heatmap_size[1]))
        # target_mask = (target_mask > 0.5).astype(np.float32)

        if self.transform:
            input = self.transform(input)
            # target_mask = self.transform(target_mask)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)


       # daiyan tight target bbox
       #  target_joints = np.floor(joints)  # (17,3) ,each (x,y,k)
        # loc = np.logical_and(target_joints[:, 0] > 0, target_joints[:, 1] > 0)
        # if loc.sum() > 3:
        #     tmp_target_joints = target_joints[loc]
        #     xmin = tmp_target_joints[:, 0].min()-3
        #     xmax = tmp_target_joints[:, 0].max()+3
        #     ymin = tmp_target_joints[:, 1].min()-3
        #     ymax = tmp_target_joints[:, 1].max()+3
        #     xmin = xmin if xmin > 0 else 0
        #     xmax = xmax if xmax < self.image_size[0] else self.image_size[0] - 1
        #     ymin = ymin if ymin > 0 else 0
        #     ymax = ymax if ymax < self.image_size[1] else self.image_size[1] - 1
        #     # cv2.rectangle(input_copy, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 0, 255), thickness=1)
        #     # cv2.imwrite('/home/daiyan/code/deep-high-resolution-net/rectangle.jpg', input_copy)
        # else:
        #     xmin = 0
        #     xmax = self.image_size[0] -1
        #     ymin = 0
        #     ymax = self.image_size[1] -1
        #
        # tight_bbox_regression = self.generate_tight_bbox(xmin, xmax, ymin, ymax)

        target, target_weight = self.generate_target(joints, joints_vis)

        # to generate the target limbs
        limbs_target, limbs_vis = self.generate_limbs_target(joints, (target_weight > 0).astype(np.float32))
        for conn_id, conn in enumerate(self.connection_rules):
            kpt1_hm, kpt2_hm = target[conn[0]], target[conn[1]]
            limbs_target[conn_id] = np.maximum(limbs_target[conn_id], np.maximum(kpt1_hm, kpt2_hm))
        limbs_target = (limbs_target > 0.9).astype(np.float32)

        inter_target = np.zeros_like(target)
        inter_target_weight = np.zeros_like(target_weight)
        for i in range(len(interference_joints)):
            inter_joints = interference_joints[i]
            inter_joints_vis = interference_joints_vis[i]
            for j in range(self.num_joints):
                if inter_joints_vis[j, 0] > 0.0:
                    inter_joints[j, 0:2] = affine_transform(inter_joints[j, 0:2], trans)

            _inter_target, _inter_target_weight = self.generate_target(inter_joints, inter_joints_vis)

            inter_target = np.maximum(inter_target, _inter_target)
            inter_target_weight = np.maximum(inter_target_weight, _inter_target_weight)

        # all_ins_target = np.maximum(0.5*inter_target, target)
        all_ins_target = np.maximum(0 * inter_target, target)
        # points = self.generate_candidate_points_from_heatmaps(inter_target)
        # all_ins_target_weight = np.maximum(inter_target_weight, target_weight)
        all_ins_target_weight = np.maximum(0 * inter_target_weight, target_weight)
        # cv2.imwrite('heatmap.jpg',np.max(target,axis=0)*255)
        # cv2.imwrite('inter_heatmap.jpg', np.max(inter_target, axis=0) * 255)

        kpts_onehots = self.heatmap2onehot(target)


        # heatmap labels
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)
        all_ins_target = torch.from_numpy(all_ins_target)
        all_ins_target_weight = torch.from_numpy(all_ins_target_weight)


        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            # 'tight_bbox': tight_bbox_regression,
            'interference_maps': inter_target,
            'kpt_cat_maps': kpts_onehots,
            # 'target_mask': target_mask,
            'limbs_target': limbs_target,
        }
        return input, target, target_weight, meta
        # return input, target, target_weight, all_ins_target, all_ins_target_weight, limbs_target, meta


    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                # size = 2 * tmp_size + 3
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def generate_tight_bbox(self, xmin, xmax, ymin, ymax):
        """ Get box regression deltas (dx, dy, dw, dh) that (dx, dy) is the Offset in the upper left corner, (dw, dh) is the width and height"""

        target_width = np.array([xmax - xmin + 1]).astype(np.float32)
        target_width = torch.from_numpy(target_width)
        target_height = np.array([ymax - ymin + 1]).astype(np.float32)
        target_height = torch.from_numpy(target_height)

        # normalized
        dx = np.array([(xmin - 0) / self.image_size[0]]).astype(np.float32)
        dx = torch.from_numpy(dx)
        dy = np.array([(ymin - 0) / self.image_size[1]]).astype(np.float32)
        dy = torch.from_numpy(dy)
        dw = target_width / self.image_size[0]
        dh = target_height / self.image_size[1]
        # dw = torch.log(target_width / self.image_size[0])
        # dh = torch.log(target_height / self.image_size[1])
        delta = torch.cat((dx, dy, dw, dh,), dim=0)
        # delta = torch.stack((dx, dy, dw, dh), dim=1)

        return delta

    def generate_limb_from_two_point(self, pointA, pointB, hm_x, hm_y, thre=1):
        limb_maps = np.zeros((hm_y, hm_x))
        centerA = pointA.astype(float)
        centerB = pointB.astype(float)
        epis = 1e-10
        limb_vec = centerB - centerA
        norm = np.linalg.norm(limb_vec)
        limb_vec_unit = limb_vec / (norm + epis)

        # To make sure not beyond the border of this two points
        min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
        max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), hm_x)
        min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
        max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), hm_y)

        range_x = list(range(int(min_x), int(max_x), 1))
        range_y = list(range(int(min_y), int(max_y), 1))

        xx, yy = np.meshgrid(range_x, range_y)

        ba_x = xx - centerA[0]  # the vector from (x,y) to centerA

        ba_y = yy - centerA[1]

        limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])

        mask = limb_width < thre  # mask is 2D

        xx = xx.reshape((-1, 1))
        yy = yy.reshape((-1, 1))
        mask = mask.reshape(-1)
        limb_points = np.hstack([xx[mask], yy[mask]])
        limb_points = limb_points.astype(np.int32)
        limb_maps[limb_points[:, 1], limb_points[:, 0]] = 1

        return limb_maps

    def generate_limbs_target(self, joints, joints_vis):
        num_limbs = len(self.connection_rules)
        limbs_target = np.zeros((num_limbs, self.heatmap_size[1], self.heatmap_size[0]))
        feat_stride = self.image_size / self.heatmap_size
        limbs_vis = np.zeros((num_limbs,))
        for conn_id, conn in enumerate(self.connection_rules):
            kpt1, kpt2 = joints[conn[0]], joints[conn[1]]
            vis1, vis2 = joints_vis[conn[0], 0], joints_vis[conn[1], 0]

            if vis1 > 0 and vis2 > 0:
                kpt1 = np.asarray([int(kpt1[0] / feat_stride[0] + 0.5), int(kpt1[1] / feat_stride[1] + 0.5)])
                kpt2 = np.asarray([int(kpt2[0] / feat_stride[0] + 0.5), int(kpt2[1] / feat_stride[1] + 0.5)])
                limbs_target[conn_id] = self.generate_limb_from_two_point(kpt1,
                                                                          kpt2,
                                                                          self.heatmap_size[0],
                                                                          self.heatmap_size[1]
                                                                          )
                limbs_vis[conn_id] = 1
        return limbs_target, limbs_vis



    def heatmap2onehot(self, heatmaps):
        bg_map = (np.max(heatmaps, axis=0) < 0.99).astype(np.float32)
        fg_map = (heatmaps >= 0.99).astype(np.float32)
        onehot = np.concatenate([bg_map[None], fg_map], axis=0)
        return onehot

    def generate_association_map_from_gt_heatmaps(self, targets, all_targets):
        heatmaps = (np.max(targets, axis=0) == 1.).astype(np.float32)
        heatmaps = heatmaps.reshape((-1, 1))
        AMaps = heatmaps * heatmaps.transpose()
        weights = (np.max(all_targets, axis=0) == 1.).astype(np.float32)
        weights = weights.reshape((-1, 1))
        weights = weights * weights.transpose()
        return AMaps, weights

    def generate_association_map_from_labels(self, points):
        num_joints = len(points)
        AMaps = np.zeros((num_joints, num_joints))
        num_target_joints = np.sum(points[:, -1])
        if num_target_joints > 0:
            ind = np.where(points[:,-1] == 1)[0]
            ind_x, ind_y = np.meshgrid(ind, ind)
            AMaps[ind_y.flatten(),ind_x.flatten()] = 1
        return AMaps

    def generate_candidate_points_from_heatmaps(self, heatmaps, thresh=1.0):
        points = []
        for i in range(len(heatmaps)):

            locs = np.where(heatmaps[i]>=thresh)
            scores = heatmaps[i][locs]
            locs = np.asarray(locs).transpose()
            ind = np.argsort(scores)[::-1]
            scores = scores[ind]
            locs = locs[ind]
            dist = self.compute_points_dist(locs)
            remove_tags = np.zeros((len(scores)))
            target_points = []
            for j in range(len(scores)):
                if remove_tags[j] == 1:
                    continue
                target_points += [locs[j,0], locs[j,1], i]
                current_score = scores[j]
                remove_cands = dist[j].flatten()
                remove_targets = np.logical_and(remove_cands<3, scores<current_score)
                remove_targets[j] = 0
                remove_tags[remove_targets] = 1
            points += target_points
        return points


    def compute_points_dist(self,points):
        loc_x = points[:,0].reshape((-1,1))
        loc_y = points[:,1].reshape((-1,1))
        dx = (loc_x[:,None] - loc_x)**2
        dy = (loc_y[:,None] - loc_y)**2
        dist = np.sqrt(dx + dy)
        return dist


class CPAEJointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB
        self.max_num_people = cfg.DATASET.MAX_PEOPLE_PER_BBOX

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']
        if 'interference' in db_rec.keys():
            interference_joints = db_rec['interference']
            interference_joints_vis = db_rec['interference_vis']
        else:
            interference_joints = [joints]
            interference_joints_vis = [joints_vis]

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1
                for i in range(len(interference_joints)):
                    interference_joints[i], interference_joints_vis[i] = fliplr_joints(
                    interference_joints[i], interference_joints_vis[i], data_numpy.shape[1], self.flip_pairs)

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        # cv2.imwrite('img.jpg',input[:,:,::-1])
        if self.transform:
            input = self.transform(input)


        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        target, target_weight = self.generate_target(joints, joints_vis)

        # interference joints heatmaps
        inter_target = np.zeros_like(target)
        inter_target_weight = np.zeros_like(target_weight)
        for i in range(len(interference_joints)):
            inter_joints = interference_joints[i]
            inter_joints_vis = interference_joints_vis[i]
            for j in range(self.num_joints):
                if inter_joints_vis[j, 0] > 0.0:
                    inter_joints[j, 0:2] = affine_transform(inter_joints[j, 0:2], trans)
            _inter_target, _inter_target_weight = self.generate_target(inter_joints, inter_joints_vis)
            inter_target = np.maximum(inter_target, _inter_target)
            inter_target_weight = np.maximum(inter_target_weight, _inter_target_weight)
        all_ins_target = np.maximum(inter_target, target)
        all_ins_target_weight = np.maximum(inter_target_weight, target_weight)

        # AE labels
        All_joints = [joints] + interference_joints
        ae_targets = self.generate_joints_ae_targets(All_joints)

        # GPU formate
        all_ins_target = torch.from_numpy(all_ins_target)
        all_ins_target_weight = torch.from_numpy(all_ins_target_weight)
        ae_targets = torch.from_numpy(ae_targets)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'interference_maps': inter_target,
        }
        return input, all_ins_target, all_ins_target_weight, ae_targets, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected


    def generate_joints_ae_targets(self, joints):
        visible_nodes = np.zeros((self.max_num_people, self.num_joints, 2))

        for i in range(len(joints)):
            if i > self.max_num_people:
                break
            tot = 0
            for idx, pt in enumerate(joints[i]):
                x, y = int(pt[0]), int(pt[1])
                if pt[2] > 0 and x >= 0 and y >= 0 \
                        and x < self.output_res[0] and y < self.output_res[1]:
                    visible_nodes[i][tot] = (y * self.output_res[0] + x, 1)
                    tot += 1
        return visible_nodes

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight


class CPSkelotonJointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']
        crowd_index = db_rec['crowd_index']
        crowd_index = np.ones((self.num_joints, 1)) * crowd_index
        if 'interference' in db_rec.keys():
            interference_joints = db_rec['interference']
            interference_joints_vis = db_rec['interference_vis']
        else:
            interference_joints = [joints]
            interference_joints_vis = [joints_vis]

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0


            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                # mask_numpy = mask_numpy[:, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1
                for i in range(len(interference_joints)):
                    interference_joints[i], interference_joints_vis[i] = fliplr_joints(
                    interference_joints[i], interference_joints_vis[i], data_numpy.shape[1], self.flip_pairs)
        joints_heatmap = joints.copy()
        trans = get_affine_transform(c, s, r, self.image_size)
        trans_heatmap = get_affine_transform(c, s, r, self.heatmap_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)
        # print('the size after transform', input.size())

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                joints_heatmap[i, 0:2] = affine_transform(joints_heatmap[i, 0:2], trans_heatmap)


        # target, target_weight = self.generate_target(joints, joints_vis)
        target, target_weight = self.generate_unbiased_target(joints_heatmap, joints_vis)

        # to generate the target limbs
        limbs_target, limbs_vis = self.generate_limbs_target(joints, (target_weight>0).astype(np.float32))
        for conn_id, conn in enumerate(self.connection_rules):
            kpt1_hm, kpt2_hm = target[conn[0]], target[conn[1]]
            limbs_target[conn_id] = np.maximum(limbs_target[conn_id], np.maximum(kpt1_hm, kpt2_hm))
        limbs_target = (limbs_target > 0.9).astype(np.float32)

        inter_target = np.zeros_like(target)
        inter_target_weight = np.zeros_like(target_weight)
        for i in range(len(interference_joints)):
            inter_joints = interference_joints[i]
            inter_joints_vis = interference_joints_vis[i]
            for j in range(self.num_joints):
                if inter_joints_vis[j, 0] > 0.0:
                    # inter_joints[j, 0:2] = affine_transform(inter_joints[j, 0:2], trans)
                    inter_joints[j, 0:2] = affine_transform(inter_joints[j, 0:2], trans_heatmap)

            # _inter_target, _inter_target_weight = self.generate_target(inter_joints, inter_joints_vis)
            _inter_target, _inter_target_weight = self.generate_unbiased_target(inter_joints, inter_joints_vis)

            inter_target = np.maximum(inter_target, _inter_target)
            inter_target_weight = np.maximum(inter_target_weight, _inter_target_weight)

        all_ins_target = np.maximum(0.5*inter_target, target)
        all_ins_target_weight = np.maximum(inter_target_weight, target_weight)

        # kpts_onehots = self.heatmap2onehot(target)

        # heatmap labels
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)
        all_ins_target = torch.from_numpy(all_ins_target)
        all_ins_target_weight = torch.from_numpy(all_ins_target_weight)


        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'interference_maps': inter_target,
            'crowd_index': crowd_index,
            # 'kpt_cat_maps': kpts_onehots,
            'limbs_target': limbs_target,
        }
        return input, target, target_weight, all_ins_target, all_ins_target_weight, limbs_target, meta


    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def generate_unbiased_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                target_weight[joint_id] = \
                    self.adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)

                if target_weight[joint_id] == 0:
                    continue

                mu_x = joints[joint_id][0]
                mu_y = joints[joint_id][1]

                x = np.arange(0, self.heatmap_size[0], 1, np.float32)
                y = np.arange(0, self.heatmap_size[1], 1, np.float32)
                y = y[:, np.newaxis]

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2))

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def adjust_target_weight(self, joint, target_weight, tmp_size):
        feat_stride = self.image_size / self.heatmap_size
        mu_x = joint[0]
        mu_y = joint[1]
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight

    def generate_tight_bbox(self, xmin, xmax, ymin, ymax):
        """ Get box regression deltas (dx, dy, dw, dh) that (dx, dy) is the Offset in the upper left corner, (dw, dh) is the width and height"""

        target_width = np.array([xmax - xmin + 1]).astype(np.float32)
        target_width = torch.from_numpy(target_width)
        target_height = np.array([ymax - ymin + 1]).astype(np.float32)
        target_height = torch.from_numpy(target_height)

        # normalized
        dx = np.array([(xmin - 0) / self.image_size[0]]).astype(np.float32)
        dx = torch.from_numpy(dx)
        dy = np.array([(ymin - 0) / self.image_size[1]]).astype(np.float32)
        dy = torch.from_numpy(dy)
        dw = target_width / self.image_size[0]
        dh = target_height / self.image_size[1]
        # dw = torch.log(target_width / self.image_size[0])
        # dh = torch.log(target_height / self.image_size[1])
        delta = torch.cat((dx, dy, dw, dh,), dim=0)
        # delta = torch.stack((dx, dy, dw, dh), dim=1)

        return delta

    def generate_limb_from_two_point(self, pointA, pointB, hm_x, hm_y, thre=1):
        limb_maps = np.zeros((hm_y, hm_x))
        centerA = pointA.astype(float)
        centerB = pointB.astype(float)
        epis = 1e-10
        limb_vec = centerB - centerA
        norm = np.linalg.norm(limb_vec)
        limb_vec_unit = limb_vec / (norm + epis)

        # To make sure not beyond the border of this two points
        min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
        max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), hm_x)
        min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
        max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), hm_y)

        range_x = list(range(int(min_x), int(max_x), 1))
        range_y = list(range(int(min_y), int(max_y), 1))

        xx, yy = np.meshgrid(range_x, range_y)

        ba_x = xx - centerA[0]  # the vector from (x,y) to centerA

        ba_y = yy - centerA[1]

        limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])

        mask = limb_width < thre  # mask is 2D

        xx = xx.reshape((-1, 1))
        yy = yy.reshape((-1, 1))
        mask = mask.reshape(-1)
        limb_points = np.hstack([xx[mask], yy[mask]])
        limb_points = limb_points.astype(np.int32)
        limb_maps[limb_points[:, 1], limb_points[:, 0]] = 1

        return limb_maps

    def generate_limbs_target(self, joints, joints_vis):
        num_limbs = len(self.connection_rules)
        limbs_target = np.zeros((num_limbs, self.heatmap_size[1], self.heatmap_size[0]))
        feat_stride = self.image_size / self.heatmap_size
        limbs_vis = np.zeros((num_limbs,))
        for conn_id, conn in enumerate(self.connection_rules):
            kpt1, kpt2 = joints[conn[0]], joints[conn[1]]
            vis1, vis2 = joints_vis[conn[0], 0], joints_vis[conn[1], 0]

            if vis1 > 0 and vis2 > 0:
                kpt1 = np.asarray([int(kpt1[0] / feat_stride[0] + 0.5), int(kpt1[1] / feat_stride[1] + 0.5)])
                kpt2 = np.asarray([int(kpt2[0] / feat_stride[0] + 0.5), int(kpt2[1] / feat_stride[1] + 0.5)])
                limbs_target[conn_id] = self.generate_limb_from_two_point(kpt1,
                                                                          kpt2,
                                                                          self.heatmap_size[0],
                                                                          self.heatmap_size[1]
                                                                          )
                limbs_vis[conn_id] = 1
        return limbs_target, limbs_vis

    def heatmap2onehot(self, heatmaps):
        bg_map = (np.max(heatmaps, axis=0) < 0.99).astype(np.float32)
        fg_map = (heatmaps >= 0.99).astype(np.float32)
        onehot = np.concatenate([bg_map[None], fg_map], axis=0)
        return onehot

