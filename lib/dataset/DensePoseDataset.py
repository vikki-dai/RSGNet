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
from pycocotools import mask as mask_util
from scipy.io  import loadmat
import scipy.spatial.distance
import os
logger = logging.getLogger(__name__)
# color_list = np.array(
#         [
#             0.000, 0.447, 0.741,
#             0.850, 0.325, 0.098,
#             0.929, 0.694, 0.125,
#             0.494, 0.184, 0.556,
#             0.466, 0.674, 0.188,
#             0.301, 0.745, 0.933,
#             0.635, 0.078, 0.184,
#             0.300, 0.300, 0.300,
#             0.600, 0.600, 0.600,
#             1.000, 0.000, 0.000,
#             1.000, 0.500, 0.000,
#             0.749, 0.749, 0.000,
#             0.000, 1.000, 0.000,
#             0.000, 0.000, 1.000,
#             0.667, 0.000, 1.000,
#             0.333, 0.333, 0.000,
#             0.333, 0.667, 0.000,
#             0.333, 1.000, 0.000,
#             0.667, 0.333, 0.000,
#             0.667, 0.667, 0.000,
#             0.667, 1.000, 0.000,
#             1.000, 0.333, 0.000,
#             1.000, 0.667, 0.000,
#             1.000, 1.000, 0.000,
#             0.000, 1.000, 1.000,
#             0.333, 0.000, 1.000,
#             0.333, 0.333, 1.000,
#             0.333, 0.667, 1.000,
#             0.333, 1.000, 1.000,
#             0.667, 0.000, 1.000,
#             0.667, 0.333, 1.000,
#             0.667, 0.667, 1.000,
#             0.667, 1.000, 1.000,
#             1.000, 0.000, 1.000,
#             1.000, 0.333, 1.000,
#             1.000, 0.667, 1.000,
#             0.000, 0.000, 1.000,
#             0.000, 0.000, 0.000,
#             0.143, 0.143, 0.143,
#             0.286, 0.286, 0.286,
#             0.429, 0.429, 0.429,
#             0.571, 0.571, 0.571,
#             0.714, 0.714, 0.714,
#             0.857, 0.857, 0.857,
#             1.000, 1.000, 1.000
#         ]).astype(np.float32)
# color_list = color_list.reshape((-1, 3)) * 255

class DensePoseDataset():
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.num_surfaces = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []
        self.dp_utils = DensePoseMethods()
        self.is_train = is_train
        self.translation = True
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP

        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        # self.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

        self.transform = transform
        self.db = []
        self.idx_to_surface = {1: [1, 2],
                          2: [3],
                          3: [4],
                          4: [5],
                          5: [6],
                          6: [7, 9],
                          7: [8, 10],
                          8: [11, 13],
                          9: [12, 14],
                          10: [15, 17],
                          11: [16, 18],
                          12: [19, 21],
                          13: [20, 22],
                          14: [23, 24]}

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        if not self.is_train:
            return self.get_inference_data(idx)
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']

        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        # print('image_file:',image_file)
        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        im_h, im_w = data_numpy.shape[0], data_numpy.shape[1]
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))
        # keypoints ground truth
        # joints = db_rec['joints_3d']
        # joints_vis = db_rec['joints_3d_vis']

        # densepose ground truth
        if db_rec['dp_tag']:
            GT_I = db_rec['dp_I']
            GT_U = db_rec['dp_U']
            GT_V = db_rec['dp_V']
            GT_x = db_rec['dp_x']
            GT_y = db_rec['dp_y']
            dp_mask = self.get_dp_mask(db_rec['dp_masks'])
        else:
            GT_I = np.zeros((1,))
            GT_U = np.zeros((1,))
            GT_V = np.zeros((1,))
            GT_x = np.zeros((1,))
            GT_y = np.zeros((1,))
            dp_mask = np.zeros([256, 256])

        # bbox ground truth
        ref_box = np.asarray(db_rec['bbox'], np.int32)
        # if db_rec['proposal'] is not None:
        #     proposal = np.asarray(db_rec['proposal'], np.int32)
        # else:
        #     proposal = None

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0
        # vis debug
        # br_x = im_w - 1 if ref_box[0] + ref_box[2] >= im_w else ref_box[0] + ref_box[2]
        # br_y = im_h - 1 if ref_box[1] + ref_box[3] >= im_h else ref_box[1] + ref_box[3]
        # cv2.rectangle(data_numpy, (int(ref_box[0]), int(ref_box[1])), (int(br_x), int(br_y)), (255, 0, 0))
        # cv2.imwrite('ori_im.jpg',data_numpy)

        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                # full_mask = full_mask[:, ::-1]
                # joints, joints_vis = fliplr_joints(
                #     joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1
                GT_I,GT_U,GT_V,GT_x,GT_y,dp_mask = self.dp_utils.get_symmetric_densepose(GT_I,GT_U,GT_V,GT_x,GT_y,dp_mask)
                ref_box[0] = data_numpy.shape[1] - ref_box[0] - ref_box[2]
                # if proposal is not None:
                #     proposal[0] = data_numpy.shape[1] - proposal[0] - proposal[2]

        # GT_I = GT_I[:, np.newaxis]
        GT_U = GT_U[:, np.newaxis]
        GT_V = GT_V[:, np.newaxis]
        GT_x = GT_x[:, np.newaxis]
        GT_y = GT_y[:, np.newaxis]
        box_x, box_y, box_w, box_h = ref_box[0], ref_box[1], ref_box[2], ref_box[3]
        box_w = np.maximum(box_w, 1)
        box_h = np.maximum(box_h, 1)
        global_p_mask = self.adjust_dp_mask(dp_mask, ref_box, im_h, im_w)

        dp_body_mask = np.max(global_p_mask, axis=0)
        # dp_body_mask = np.max(dp_mask,axis=0)
        # input
        # input = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.float32)
        # target scale
        # if box_h > box_w:
        #     target_h = self.image_size[1]
        #     im_scale = target_h / box_h
        # else:
        #     target_w = self.image_size[0]
        #     im_scale = target_w / box_w
        # input_im = cv2.resize(
        #     data_numpy[box_y:box_y+box_h, box_x:box_x+box_w, :],
        #     None,
        #     None,
        #     fx=im_scale,
        #     fy=im_scale,
        #     interpolation=cv2.INTER_LINEAR
        # )
        # input[0:input_im.shape[0], 0:input_im.shape[1], :] = input_im

        trans = get_affine_transform(c, s, r, self.image_size)
        # vis_input = np.copy(input)
        # cv2.imwrite('ori_vis_input.jpg', vis_input)
        # step1: data preprocess
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        # input = input.astype(np.float32)
        # input -= self.PIXEL_MEANS
        # input = np.transpose(input,(2,0,1))

        # vis debug
        # vis_input = np.copy(input)
        # input = torch.from_numpy(input)
        # vis_input = cv2.warpAffine(
        #     vis_data,
        #     trans,
        #     (int(self.image_size[0]), int(self.image_size[1])),
        #     flags=cv2.INTER_LINEAR)
        # vis_input = cv2.resize(vis_input,(self.heatmap_size[0],self.heatmap_size[1]))
        # cv2.imwrite('vis_input.jpg', vis_input)
        # cv2.imwrite('dp_img.jpg', input)
        # cv2.imwrite('im.jpg', input)

        if self.transform:
            input = self.transform(input)

        # step2 densepose preprocess
        # for i in range(self.num_joints):
        #     if joints_vis[i, 0] > 0.0:
        #         joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        # -- dp masks
        dp_body_mask = cv2.warpAffine(
            dp_body_mask,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        # _dp_body_mask = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.float32)
        # dp_body_mask = cv2.resize(
        #     dp_body_mask,
        #     None,
        #     None,
        #     fx=im_scale,
        #     fy=im_scale,
        #     interpolation=cv2.INTER_LINEAR
        # )
        # _dp_body_mask[0:dp_body_mask.shape[0], 0:dp_body_mask.shape[1]] = dp_body_mask
        dp_body_mask = cv2.resize(dp_body_mask, (int(self.heatmap_size[0]), int(self.heatmap_size[1])))
        dp_body_mask_target = (dp_body_mask>0.5).astype(np.float32)
        # cv2.imwrite('body_mask.jpg',dp_body_mask_target*255)
        # -- dp part masks
        dp_body_part_target = np.zeros((15, dp_body_mask_target.shape[0], dp_body_mask_target.shape[1]), dtype=np.float32)
        for i in range(1,15):
            _part_mask = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.float32)
            i_part_mask = cv2.warpAffine(
                global_p_mask[i],
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR)
            # i_part_mask = cv2.resize(
            #     global_p_mask[i],
            #     None,
            #     None,
            #     fx=im_scale,
            #     fy=im_scale,
            #     interpolation=cv2.INTER_LINEAR
            # )
            _part_mask[0:i_part_mask.shape[0], 0:i_part_mask.shape[1]] = i_part_mask
            i_part_mask = cv2.resize(_part_mask, (int(self.heatmap_size[0]), int(self.heatmap_size[1])))
            dp_body_part_target[i] = (i_part_mask > 0.5).astype(np.float32)
        dp_body_part_target = np.argmax(dp_body_part_target,axis=0)
        dp_body_part_target = dp_body_part_target.astype(np.float32)
        # cv2.imwrite('parts.jpg',dp_body_part_target*255)

        # -- dp XY coordinates
        dp_xy_coords = np.hstack([GT_x, GT_y])
        dp_xy_coords[:, 0] = dp_xy_coords[:, 0] / 256. * ref_box[2] + ref_box[0]
        dp_xy_coords[:, 1] = dp_xy_coords[:, 1] / 256. * ref_box[3] + ref_box[1]
        #
        # dp_tl_xy_coords = np.zeros((196, 2), dtype=np.float32)
        # dp_tr_xy_coords = np.zeros((196, 2), dtype=np.float32)
        # dp_bl_xy_coords = np.zeros((196, 2), dtype=np.float32)
        # dp_br_xy_coords = np.zeros((196, 2), dtype=np.float32)
        # dp_tl_w = np.zeros((196, 1), dtype=np.float32)
        # dp_tr_w = np.zeros((196, 1), dtype=np.float32)
        # dp_bl_w = np.zeros((196, 1), dtype=np.float32)
        # dp_br_w = np.zeros((196, 1), dtype=np.float32)
        #
        feat_stride = self.image_size / self.heatmap_size
        for idx, ipoint in enumerate(dp_xy_coords):
            dp_xy_coords[idx] = affine_transform(dp_xy_coords[idx], trans)
            # dp_xy_coords[idx] = dp_xy_coords[idx] * im_scale
            dp_xy_coords[idx][0] = dp_xy_coords[idx][0] / feat_stride[0]
            dp_xy_coords[idx][1] = dp_xy_coords[idx][1] / feat_stride[1]
            # low_x, high_x, x_w = self.point_linear_interpolation(dp_xy_coords[idx][0], self.heatmap_size[0]-1)
            # low_y, high_y, y_w = self.point_linear_interpolation(dp_xy_coords[idx][1], self.heatmap_size[1]-1)
            # w_xlo_ylo = (1.0 - x_w) * (1.0 - y_w)
            # w_xlo_yhi = (1.0 - x_w) * y_w
            # w_xhi_ylo = x_w * (1 - y_w)
            # w_xhi_yhi = x_w * y_w
            # dp_tl_xy_coords[idx][0] = low_x
            # dp_tl_xy_coords[idx][1] = low_y
            # dp_tl_xy_coords[idx][2] = w_xlo_ylo
            # dp_tl_w[idx] = w_xlo_ylo
            #
            # dp_tr_xy_coords[idx][0] = high_x
            # dp_tr_xy_coords[idx][1] = low_y
            # dp_tr_xy_coords[idx][2] = w_xhi_ylo
            # dp_tr_w[idx] = w_xhi_ylo
            #
            # dp_bl_xy_coords[idx][0] = low_x
            # dp_bl_xy_coords[idx][1] = high_y
            # dp_bl_xy_coords[idx][2] = w_xlo_yhi
            # dp_bl_w[idx] = w_xlo_yhi
            #
            # dp_br_xy_coords[idx][0] = high_x
            # dp_br_xy_coords[idx][1] = high_y
            # dp_br_xy_coords[idx][2] = w_xhi_yhi
            # dp_br_w[idx] = w_xhi_yhi
            # vis debug
            # cv2.circle(input, (int(dp_xy_coords[idx][0]), int(dp_xy_coords[idx][1])), 3, (255,255,0), -1)
            # cv2.imwrite('dp_coord_img.jpg', input)
        dp_xy_coords = dp_xy_coords.astype(np.int32)
        dp_xy_coords[dp_xy_coords<0]=0
        dp_xy_coords[:, 0][dp_xy_coords[:, 0] >= self.heatmap_size[0]] = self.heatmap_size[0] - 1
        dp_xy_coords[:, 1][dp_xy_coords[:, 1] >= self.heatmap_size[1]] = self.heatmap_size[1] - 1
        #
        # index_uv_targets = np.zeros((196,))
        # index_uv_targets[:GT_I.shape[0]] = GT_I.copy()
        # index_uv_targets = index_uv_targets.astype(np.int32)
        # u_targets = np.zeros((196,))
        # u_targets[:GT_U.shape[0]] = GT_U.copy().reshape((-1))
        # u_targets = u_targets.astype(np.float32)
        # v_targets = np.zeros((196,))
        # v_targets[:GT_V.shape[0]] = GT_V.copy().reshape((-1))
        # v_targets = v_targets.astype(np.float32)
        #
        # vis debug
        # for idx, ipoint in enumerate(dp_xy_coords):
        #     cv2.circle(vis_input, (int(dp_xy_coords[idx][0]), int(dp_xy_coords[idx][1])), 1, (255,255,0), -1)
        # cv2.imwrite('dp_gt_img.jpg', vis_input)
        # -- dp UV coordinates
        dp_uv_coords = np.hstack([GT_U, GT_V])
        # GT_I, dp_xy_coords, dp_uv_cooreds = self.expand_dp_labels(GT_I, dp_xy_coords, dp_uv_cooreds)
        target_sf, target_u, target_v, target_weight, target_uv_weight = self.generate_dp_target(GT_I, dp_xy_coords, dp_uv_coords)
        # debug vis
        # cv2.imwrite('parts.jpg', dp_body_part_target * 255)
        # cv2.imwrite('target_sf.jpg', (target_sf > 0) * 255)
        # cv2.imwrite('target_u.jpg', np.max((target_u > 0),axis=0) * 255)
        # cv2.imwrite('target_v.jpg', np.max((target_v > 0),axis=0) * 255)
        # cv2.imwrite('target_w.jpg', (target_weight > 0)* 255)
        # cv2.imwrite('target_uv_w.jpg', np.max((target_uv_weight > 0),axis=0) * 255)
        # expand IUV targets
        # IUV augmentation
        if random.random() <= 0.5:
            dp_body_part_target, target_sf, target_u, target_v, target_weight, target_uv_weight = self.expand_dp_targets(dp_body_part_target,
                                                                                                                 target_sf, target_u, target_v, target_weight, target_uv_weight)
            # if random.random() <= 0.5:
            #     target_weight[dp_body_mask_target==0] = 1
        # vis_im = np.zeros((64,64,3))
        # for surface in range(1, 25):
        #     vis_im[target_sf == surface, :] = color_list[surface]
        # cv2.imwrite('target_color_sf.jpg', vis_im)
        # cv2.imwrite('parts.jpg', dp_body_part_target * 255)
        # cv2.imwrite('target_sf.jpg', (target_sf > 0) * 255)
        # cv2.imwrite('target_u.jpg', np.max((target_u > 0), axis=0) * 255)
        # cv2.imwrite('target_v.jpg', np.max((target_v > 0), axis=0) * 255)
        # cv2.imwrite('target_w.jpg', (target_weight > 0) * 255)
        # cv2.imwrite('target_uv_w.jpg', np.max((target_uv_weight > 0), axis=0) * 255)

        dp_body_mask_target = torch.from_numpy(dp_body_mask_target)
        target_sf = torch.from_numpy(target_sf)
        target_u = torch.from_numpy(target_u)
        target_v = torch.from_numpy(target_v)
        target_weight = torch.from_numpy(target_weight)
        target_uv_weight = torch.from_numpy(target_uv_weight)
        dp_body_part_target = torch.from_numpy(dp_body_part_target)
        # target_full_mask = torch.from_numpy(input_full_mask)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            # 'joints': joints,
            # 'joints_vis': joints_vis,
            'bbox': ref_box,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            # 'tl_points': dp_tl_xy_coords,
            # 'tr_points': dp_tr_xy_coords,
            # 'bl_points': dp_bl_xy_coords,
            # 'br_points': dp_br_xy_coords,
            # 'tl_points_w': dp_tl_w,
            # 'tr_points_w': dp_tr_w,
            # 'bl_points_w': dp_bl_w,
            # 'br_points_w': dp_br_w,
            # 'index_uv': index_uv_targets,
            # 'u_gt': u_targets,
            # 'v_gt': v_targets
        }
        return input, dp_body_mask_target, dp_body_part_target, target_sf, target_u, target_v, target_weight, target_uv_weight, meta
        # return input,target_full_mask, dp_body_mask_target, dp_body_part_target, target_sf, target_u, target_v, target_weight, target_uv_weight, meta
        # return input, target_weight, meta

    def point_linear_interpolation(self, value, border):

        v_low = np.maximum(0, np.minimum(int(value), border))
        v_high = np.minimum((v_low + 1), border)
        value = np.minimum(float(v_high), value)
        v_w = value - float(v_low)
        return v_low, v_high, v_w

    def expand_dp_targets(self, dp_body_part_target, target_sf, target_u, target_v, target_weight, target_uv_weight):
        for part_id in range(1,15):
            surface_id = self.idx_to_surface[part_id]

            part_map = (dp_body_part_target==part_id).astype(np.int32)
            if part_map.max()==0:
                continue
            surface_map = np.zeros_like(part_map)
            for s in surface_id:
                surface_map = np.logical_or(surface_map, target_sf==s)
            surface_map = surface_map.astype(np.int32)
            if surface_map.max()==0:
                continue
            diff_map = part_map - surface_map
            points_to_add = np.asarray(np.where(diff_map == 1)).transpose()
            labeled_points = np.asarray(np.where(surface_map == 1)).transpose()
            for i_point in range(points_to_add.shape[0]):
                point = points_to_add[i_point].reshape((1,2))
                dist = np.sqrt(np.sum((labeled_points - point)**2,axis=1))
                idx_target_point = np.argmin(dist,axis=0)
                loc_y, loc_x = labeled_points[idx_target_point, 0], labeled_points[idx_target_point, 1]
                point = point.reshape((2,))
                # expand I U V
                I = int(target_sf[loc_y, loc_x])
                target_sf[point[0], point[1]] = I
                U = target_u[I, loc_y, loc_x]
                V = target_v[I, loc_y, loc_x]
                u_to_add = (point[1]+1) * U / (loc_x+1)
                v_to_add = (point[0]+1) * V / (loc_y+1)
                target_u[I, point[0], point[1]] = u_to_add
                target_v[I, point[0], point[1]] = v_to_add
                # expand weight
                target_weight[point[0], point[1]] = 1
                target_uv_weight[I, point[0], point[1]] = 1
        return dp_body_part_target, target_sf, target_u, target_v, target_weight, target_uv_weight

    def polys_to_mask(self, polygons, height, width):
        """Convert from the COCO polygon segmentation format to a binary mask
        encoded as a 2D array of data type numpy.float32. The polygon segmentation
        is understood to be enclosed inside a height x width image. The resulting
        mask is therefore of shape (height, width).
        """
        rle = mask_util.frPyObjects(polygons, height, width)
        mask = np.array(mask_util.decode(rle), dtype=np.float32)
        # Flatten in case polygons was a list
        mask = np.sum(mask, axis=2)
        mask = np.array(mask > 0, dtype=np.float32)
        return mask
    def get_inference_data(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        im_h, im_w = data_numpy.shape[0], data_numpy.shape[1]
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        # bbox ground truth
        ref_box = np.asarray(db_rec['bbox'],np.int32)#db_rec['bbox']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        trans = get_affine_transform(c, s, r, self.image_size)
        # # step1: data preprocess
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        # box_x, box_y, box_w, box_h = ref_box[0], ref_box[1], ref_box[2], ref_box[3]
        # input = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.float32)
        # target scale
        # if box_h > box_w:
        #     target_h = self.image_size[1]
        #     im_scale = target_h / box_h
        # else:
        #     target_w = self.image_size[0]
        #     im_scale = target_w / box_w
        # input_im = cv2.resize(
        #     data_numpy[box_y:box_y + box_h, box_x:box_x + box_w, :],
        #     None,
        #     None,
        #     fx=im_scale,
        #     fy=im_scale,
        #     interpolation=cv2.INTER_LINEAR
        # )
        # input[0:input_im.shape[0], 0:input_im.shape[1], :] = input_im
        # input = input.astype(np.float32)
        # input -= self.PIXEL_MEANS
        # input = np.transpose(input, (2, 0, 1))
        # input = torch.from_numpy(input)
        if self.transform:
            input = self.transform(input)

        # if 'dp_masks' in db_rec:
        #     GT_I = db_rec['dp_I']
        #     GT_U = db_rec['dp_U']
        #     GT_V = db_rec['dp_V']
        #     GT_x = db_rec['dp_x']
        #     GT_y = db_rec['dp_y']
        #     dp_mask = self.get_dp_mask(db_rec['dp_masks'])
        #     GT_U = GT_U[:, np.newaxis]
        #     GT_V = GT_V[:, np.newaxis]
        #     GT_x = GT_x[:, np.newaxis]
        #     GT_y = GT_y[:, np.newaxis]
        #     global_p_mask = self.adjust_dp_mask(dp_mask, [int(ref_box[0]),int(ref_box[1]),int(ref_box[2]),int(ref_box[3])], im_h, im_w)
        #     dp_body_mask = np.max(global_p_mask, axis=0)
        #     dp_body_mask = cv2.warpAffine(
        #         dp_body_mask,
        #         trans,
        #         (int(self.image_size[0]), int(self.image_size[1])),
        #         flags=cv2.INTER_LINEAR)
        #     dp_body_mask = cv2.resize(dp_body_mask, (int(self.heatmap_size[0]), int(self.heatmap_size[1])))
        #     dp_body_mask_target = (dp_body_mask > 0.5).astype(np.float32)
        #     # -- dp part masks
        #     dp_body_part_target = np.zeros((15, dp_body_mask_target.shape[0], dp_body_mask_target.shape[1]),
        #                                    dtype=np.float32)
        #     for i in range(1, 15):
        #         i_part_mask = cv2.warpAffine(
        #             global_p_mask[i],
        #             trans,
        #             (int(self.image_size[0]), int(self.image_size[1])),
        #             flags=cv2.INTER_LINEAR)
        #         i_part_mask = cv2.resize(i_part_mask, (int(self.heatmap_size[0]), int(self.heatmap_size[1])))
        #         dp_body_part_target[i] = (i_part_mask > 0.5).astype(np.float32)
        #     dp_body_part_target = np.argmax(dp_body_part_target, axis=0)
        #     dp_body_part_target = dp_body_part_target.astype(np.float32)
        #
        #
        #     # -- dp XY coordinates
        #     dp_xy_coords = np.hstack([GT_x, GT_y])
        #     dp_xy_coords[:, 0] = dp_xy_coords[:, 0] * ref_box[2] / 255.  #+ ref_box[0]
        #     dp_xy_coords[:, 1] = dp_xy_coords[:, 1] * ref_box[3] / 255.  #+ ref_box[1]
        #     # feat_stride = self.image_size / self.heatmap_size
        #     # for idx, ipoint in enumerate(dp_xy_coords):
        #     #     dp_xy_coords[idx] = affine_transform(dp_xy_coords[idx], trans)
        #     #     dp_xy_coords[idx][0] = int(dp_xy_coords[idx][0] / feat_stride[0] + 0.5)
        #     #     dp_xy_coords[idx][1] = int(dp_xy_coords[idx][1] / feat_stride[1] + 0.5)
        #         # cv2.circle(input, (int(dp_xy_coords[idx][0]), int(dp_xy_coords[idx][1])), 3, (255,255,0), -1)
        #         # cv2.imwrite('dp_img.jpg', input)
        #     dp_xy_coords = dp_xy_coords.astype(np.int32)
        #     # dp_xy_coords[dp_xy_coords < 0] = 0
        #     # dp_xy_coords[:, 0][dp_xy_coords[:, 0] >= self.heatmap_size[0]] = self.heatmap_size[0] - 1
        #     # dp_xy_coords[:, 1][dp_xy_coords[:, 1] >= self.heatmap_size[1]] = self.heatmap_size[1] - 1
        #     dp_uv_cooreds = np.hstack([GT_U, GT_V])
        #     target_sf = np.zeros((196,1),dtype=np.uint8)
        #     target_xy = np.zeros((196,2),dtype=dp_xy_coords.dtype)
        #     target_uv = np.zeros((196,2),dtype=dp_uv_cooreds.dtype)
        #     target_tag = np.zeros((196,1),dtype=np.uint8)
        #     target_sf[0:len(GT_I)] = GT_I.reshape((-1,1))
        #     target_xy[0:len(dp_xy_coords)] = dp_xy_coords
        #     target_uv[0:len(dp_uv_cooreds)] = dp_uv_cooreds
        #     target_tag[0:len(GT_I)] = 1
        #
        #
        #     target_sf = torch.from_numpy(target_sf)
        #     target_xy = torch.from_numpy(target_xy)
        #     target_uv = torch.from_numpy(target_uv)
        #     target_tag = torch.from_numpy(target_tag)
            # GT_I, dp_xy_coords, dp_uv_cooreds = self.expand_dp_labels(GT_I, dp_xy_coords, dp_uv_cooreds)
            # target_sf, target_u, target_v, target_weight, target_uv_weight = self.generate_dp_inference_target(GT_I, dp_xy_coords,
            #                                                                                          dp_uv_cooreds,[ref_box[2],ref_box[3]])
        #     dp_body_mask_target = torch.from_numpy(dp_body_mask_target)
        #     target_sf = torch.from_numpy(target_sf)
        #     target_u = torch.from_numpy(target_u)
        #     target_v = torch.from_numpy(target_v)
        #     target_weight = torch.from_numpy(target_weight)
        #     target_uv_weight = torch.from_numpy(target_uv_weight)
        #     dp_body_part_target = torch.from_numpy(dp_body_part_target)
        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'bbox': ref_box,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            # 'dp_xy':dp_xy_coords,
            # 'target_u':target_u,
            # 'target_u':target_v,
            # 'dp_part':dp_body_part_target,
            # 'target_sf':target_sf
        }
        return input, meta
        # return input, target_sf, target_xy, target_uv, target_tag, meta
        # return input, dp_body_mask_target, dp_body_part_target, target_sf, target_u, target_v, target_weight, target_uv_weight, meta

    def get_dp_mask(self, Polys):
        MaskGen = np.zeros([256, 256])
        for i in range(1, 15):
            if (Polys[i - 1]):
                current_mask = mask_util.decode(Polys[i - 1])
                MaskGen[current_mask > 0] = i
        return MaskGen

    def adjust_dp_mask(self, part_anns, ref_box, height, width):
        # part_masks = []
        box_x, box_y, box_w, box_h = ref_box[0], ref_box[1], ref_box[2], ref_box[3]
        box_w = np.maximum(box_w,1)
        box_h = np.maximum(box_h,1)
        global_p_mask = np.zeros([15, height, width],dtype=np.uint8)

        for i in range(1, 15):
            # global_p_mask = np.zeros([height, width])
            p_mask = (part_anns == i).astype(np.float32)
            p_mask = cv2.resize(p_mask, (box_w, box_h))
            try:
                global_p_mask[i, box_y:box_y+box_h, box_x:box_x+box_w] = (p_mask>0.5).astype(np.uint8)
            except:
                print(global_p_mask.shape)
                print(ref_box,p_mask.shape)
            # part_masks.append(p_mask)
        return global_p_mask

    def _adjust_dp_mask(self, part_anns, ref_box, height, width):
        # part_masks = []
        box_w, box_h = ref_box[2], ref_box[3]
        global_p_mask = np.zeros([15, height, width],dtype=np.uint8)

        for i in range(1, 15):
            # global_p_mask = np.zeros([height, width])
            p_mask = (part_anns == i).astype(np.float32)
            p_mask = cv2.resize(p_mask, (box_w, box_h))
            try:
                global_p_mask[i, ref_box[1]:ref_box[1]+box_h, ref_box[0]:ref_box[0]+box_w] = (p_mask>0.5).astype(np.uint8)
            except:
                print(global_p_mask.shape)
                print(ref_box,p_mask.shape)
            # part_masks.append(p_mask)
        # part_masks = np.asarray(part_masks)
        # part_masks_labels = np.argmax(part_masks,axis=0)
        # part_masks = np.max(part_masks, axis=0)
        # global_p_mask[ref_box[1]:ref_box[1]+box_h, ref_box[0]:ref_box[0]+box_w] = (part_masks>0.5).astype(np.uint8)
        return global_p_mask

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

    def expand_dp_labels(self, GT_I, dp_xy_coords, dp_uv_cooreds):
        '''

        :param GT_I: (num_points, 1)
        :param dp_xy_coords: (num_points, 2)
        :param dp_uv_cooreds: (num_points, 2)
        :return: expanded_GT_I, expanded_dp_xy_coords, expanded_dp_uv_cooreds
        '''
        point_tags = np.zeros((self.heatmap_size[1], self.heatmap_size[0]))
        expanded_points_I = [GT_I.reshape((-1,1))]
        expanded_points_xy = [dp_xy_coords]
        expanded_points_uv = [dp_uv_cooreds]
        def expand_points(id_tag, start_pt_xy, end_pt_xy, start_pt_uv, end_pt_uv):
            # xy base
            vec_x = end_pt_xy[0] - start_pt_xy[0]
            vec_y = end_pt_xy[1] - start_pt_xy[1]
            min_x = max(0, int(min(start_pt_xy[0], end_pt_xy[0])))
            min_y = max(0, int(min(start_pt_xy[1], end_pt_xy[1])))
            max_x = min(self.heatmap_size[0]-1, int(max(start_pt_xy[0], end_pt_xy[0])))
            max_y = min(self.heatmap_size[1]-1, int(max(start_pt_xy[1], end_pt_xy[1])))
            norm = np.sqrt(vec_x ** 2 + vec_y ** 2)
            vec_x /= (norm + 1e-8)
            vec_y /= (norm + 1e-8)
            # uv base
            xy_length = [np.abs(end_pt_xy[0] - start_pt_xy[0]), np.abs(end_pt_xy[1] - start_pt_xy[1])]

            uv_diff = [np.abs(end_pt_uv[0] - start_pt_uv[0]), np.abs(end_pt_uv[1] - start_pt_uv[1])]
            uv_step_size = [0, 0]
            uv_step_size[0] = uv_diff[0] / xy_length[0] if xy_length[0] > 0 else 0
            uv_step_size[1] = uv_diff[1] / xy_length[1] if xy_length[1] > 0 else 0

            expand_xy = []
            expand_uv = []
            expand_i = []
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    if point_tags[y][x] == 1:
                        continue
                    bec_x = x - start_pt_xy[0]
                    bec_y = y - start_pt_xy[1]
                    dist = abs(bec_x * vec_y - bec_y * vec_x)
                    if dist > 1:
                        continue
                    point_tags[y][x] = 1
                    expand_i.append([id_tag])
                    expand_xy.append([x, y])
                    expand_uv.append([start_point_uv[0] + (x-start_pt_xy[0])*uv_step_size[0],
                                      start_point_uv[1] + (y-start_pt_xy[1])*uv_step_size[1]])
            if len(expand_i)>0:
                expanded_points_I.append(np.asarray(expand_i).reshape((-1,1)))
                expanded_points_xy.append(np.asarray(expand_xy))
                expanded_points_uv.append(np.asarray(expand_uv))

        for i in range(1, self.num_surfaces+1):
            pointx_xy = dp_xy_coords[GT_I==i]
            points_uv = dp_uv_cooreds[GT_I==i]
            num_points = len(pointx_xy)
            if num_points > 0:
                range_y, range_x = np.meshgrid(range(num_points),range(num_points))
                range_x = range_x.reshape((-1))
                range_y = range_y.reshape((-1))
                keep_idx = range_y < range_x
                range_y = range_y[keep_idx]
                range_x = range_x[keep_idx]
                for j in range(len(range_y)):
                    start_point_xy = pointx_xy[range_y[j]]
                    end_point_xy = pointx_xy[range_x[j]]
                    start_point_uv = points_uv[range_y[j]]
                    end_point_uv = points_uv[range_x[j]]
                    expand_points(i,start_point_xy,end_point_xy, start_point_uv, end_point_uv)
        return np.vstack(expanded_points_I).reshape((-1,)),np.vstack(expanded_points_xy),np.vstack(expanded_points_uv)

    def generate_dp_target(self, dp_sf, dp_xy, dp_uv):
        '''
        :param dp_sf:  index to 3D surface
        :param dp_xy:  xy coordinates
        :param dp_uv:  uv coordinates
        :return:
        index-to-surface ground truth map
        uv coordinate groud truth map
        '''

        target_sf = np.zeros((self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
        target_u = np.zeros((self.num_surfaces+1, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
        target_v = np.zeros((self.num_surfaces+1, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
        target_uv_weight = np.zeros((self.num_surfaces+1, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
        target_weight = np.zeros((self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
        loss_weights = np.zeros((25,))
        for i in range(1, self.num_surfaces + 1):
            loss_weights[i]=len(dp_xy[dp_sf==i])
        if np.sum(loss_weights) == 0:
            return target_sf, target_u, target_v, target_weight, target_uv_weight
            # print('no points')
        # loss_weights /= np.sum(loss_weights)
        # loss_weights[loss_weights>0] = 1 - loss_weights[loss_weights>0]
        for i in range(1, self.num_surfaces+1):
            locs = dp_xy[dp_sf==i]

            if len(locs) > 0:
                # x aug
                aug_locs = []
                aug_locs.append(locs)
                repeats=1
                # if self.is_train:
                #     for step in [1,2]:
                #         x_plus_locs = np.copy(locs)
                #         y_plus_locs = np.copy(locs)
                #         x_minus_locs = np.copy(locs)
                #         y_minus_locs = np.copy(locs)
                #
                #         x_plus_locs[:, 0] += step
                #         y_plus_locs[:, 1] += step
                #         x_minus_locs[:, 0] -= step
                #         y_minus_locs[:, 1] -= step
                #         aug_locs.append(x_plus_locs)
                #         aug_locs.append(y_plus_locs)
                #         aug_locs.append(x_minus_locs)
                #         aug_locs.append(y_minus_locs)
                #     locs = np.vstack(aug_locs)
                #     for loc in locs:
                #         loc[0] = loc[0] if loc[0] < self.heatmap_size[0] else self.heatmap_size[0] - 1
                #         loc[1] = loc[1] if loc[1] < self.heatmap_size[1] else self.heatmap_size[1] - 1
                #     # locs[locs[:, 0] > self.heatmap_size[0]][:, 0] = self.heatmap_size[0] - 1
                #     # locs[locs[:, 1] > self.heatmap_size[1]][:, 1] = self.heatmap_size[1] - 1
                #     # locs[locs>=self.heatmap_size[0]] = self.heatmap_size[0] - 1
                #     locs[locs < 0] = 0
                #     repeats = 9


                target_sf[locs[:,1],locs[:,0]] = i
                aug_dp_uv = dp_uv[dp_sf == i].reshape((-1,2)) #np.vstack([dp_uv[dp_sf == i].reshape((-1,2)) for rep in range(repeats)])
                target_u[i][locs[:,1],locs[:,0]] = aug_dp_uv[:, 0]
                target_v[i][locs[:,1], locs[:,0]] = aug_dp_uv[:, 1]
                target_uv_weight[i][locs[:,1], locs[:,0]] = 1.
                target_weight[locs[:,1], locs[:,0]] = 1. #loss_weights[i]


        return target_sf, target_u, target_v, target_weight, target_uv_weight

    def generate_dp_inference_target(self, dp_sf, dp_xy, dp_uv, map_size=(0,0)):
        '''
        :param dp_sf:  index to 3D surface
        :param dp_xy:  xy coordinates
        :param dp_uv:  uv coordinates
        :return:
        index-to-surface ground truth map
        uv coordinate groud truth map
        '''

        target_sf = np.zeros((map_size[1], map_size[0]), dtype=np.float32)
        target_u = np.zeros((self.num_surfaces+1, map_size[1], map_size[0]), dtype=np.float32)
        target_v = np.zeros((self.num_surfaces+1, map_size[1], map_size[0]), dtype=np.float32)
        target_uv_weight = np.zeros((self.num_surfaces+1, map_size[1], map_size[0]), dtype=np.float32)
        target_weight = np.zeros((map_size[1], map_size[0]), dtype=np.float32)
        for i in range(1, self.num_surfaces+1):
            locs = dp_xy[dp_sf==i]

            if len(locs) > 0:
                # x aug
                aug_locs = []
                aug_locs.append(locs)
                repeats=1

                target_sf[locs[:,1],locs[:,0]] = i
                aug_dp_uv = np.vstack([dp_uv[dp_sf == i].reshape((-1,2)) for rep in range(repeats)])
                target_u[i][locs[:,1],locs[:,0]] = aug_dp_uv[:, 0]
                target_v[i][locs[:,1], locs[:,0]] = aug_dp_uv[:, 1]
                target_uv_weight[i][locs[:,1], locs[:,0]] = 1
                target_weight[locs[:,1], locs[:,0]] = 1


        return target_sf, target_u, target_v, target_weight, target_uv_weight

class DensePoseMethods:
    def __init__(self):
        #
        ALP_UV = loadmat(os.path.join(os.path.dirname(__file__), '../../DensePoseData/UV_data/UV_Processed.mat'))
        self.FaceIndices = np.array(ALP_UV['All_FaceIndices']).squeeze()
        self.FacesDensePose = ALP_UV['All_Faces'] - 1
        self.U_norm = ALP_UV['All_U_norm'].squeeze()
        self.V_norm = ALP_UV['All_V_norm'].squeeze()
        self.All_vertices = ALP_UV['All_vertices'][0]
        ## Info to compute symmetries.
        self.SemanticMaskSymmetries = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14]
        self.Index_Symmetry_List = [1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24,
                                    23];
        UV_symmetry_filename = os.path.join(os.path.dirname(__file__),
                                            '../../DensePoseData/UV_data/UV_symmetry_transforms.mat')
        self.UV_symmetry_transformations = loadmat(UV_symmetry_filename)

    def get_symmetric_densepose(self, I, U, V, x, y, Mask):
        ### This is a function to get the mirror symmetric UV labels.
        Labels_sym = np.zeros(I.shape)
        U_sym = np.zeros(U.shape)
        V_sym = np.zeros(V.shape)
        ###
        for i in (range(24)):
            if i + 1 in I:
                Labels_sym[I == (i + 1)] = self.Index_Symmetry_List[i]
                jj = np.where(I == (i + 1))
                ###
                U_loc = (U[jj] * 255).astype(np.int64)
                V_loc = (V[jj] * 255).astype(np.int64)
                ###
                V_sym[jj] = self.UV_symmetry_transformations['V_transforms'][0, i][V_loc, U_loc]
                U_sym[jj] = self.UV_symmetry_transformations['U_transforms'][0, i][V_loc, U_loc]
        ##
        Mask_flip = np.fliplr(Mask)
        Mask_flipped = np.zeros(Mask.shape)
        #
        for i in (range(14)):
            Mask_flipped[Mask_flip == (i + 1)] = self.SemanticMaskSymmetries[i + 1]
        #
        [y_max, x_max] = Mask_flip.shape
        y_sym = y
        x_sym = x_max - x
        #
        return Labels_sym, U_sym, V_sym, x_sym, y_sym, Mask_flipped

    def barycentric_coordinates_exists(self, P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0
        #
        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)
        if (np.dot(vCrossW, vCrossU) < 0):
            return False;
        #
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        #
        if (np.dot(uCrossW, uCrossV) < 0):
            return False;
        #
        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom
        #
        return ((r <= 1) & (t <= 1) & (r + t <= 1))

    def barycentric_coordinates(self, P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0
        #
        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)
        #
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        #
        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom
        #
        return (1 - (r + t), r, t)

    def IUV2FBC(self, I_point, U_point, V_point):
        P = [U_point, V_point, 0]
        FaceIndicesNow = np.where(self.FaceIndices == I_point)
        FacesNow = self.FacesDensePose[FaceIndicesNow]
        #
        P_0 = np.vstack((self.U_norm[FacesNow][:, 0], self.V_norm[FacesNow][:, 0],
                         np.zeros(self.U_norm[FacesNow][:, 0].shape))).transpose()
        P_1 = np.vstack((self.U_norm[FacesNow][:, 1], self.V_norm[FacesNow][:, 1],
                         np.zeros(self.U_norm[FacesNow][:, 1].shape))).transpose()
        P_2 = np.vstack((self.U_norm[FacesNow][:, 2], self.V_norm[FacesNow][:, 2],
                         np.zeros(self.U_norm[FacesNow][:, 2].shape))).transpose()
        #

        for i, [P0, P1, P2] in enumerate(zip(P_0, P_1, P_2)):
            if (self.barycentric_coordinates_exists(P0, P1, P2, P)):
                [bc1, bc2, bc3] = self.barycentric_coordinates(P0, P1, P2, P)
                return (FaceIndicesNow[0][i], bc1, bc2, bc3)
        #
        # If the found UV is not inside any faces, select the vertex that is closest!
        #
        D1 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_0[:, 0:2]).squeeze()
        D2 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_1[:, 0:2]).squeeze()
        D3 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_2[:, 0:2]).squeeze()
        #
        minD1 = D1.min()
        minD2 = D2.min()
        minD3 = D3.min()
        #
        if ((minD1 < minD2) & (minD1 < minD3)):
            return (FaceIndicesNow[0][np.argmin(D1)], 1., 0., 0.)
        elif ((minD2 < minD1) & (minD2 < minD3)):
            return (FaceIndicesNow[0][np.argmin(D2)], 0., 1., 0.)
        else:
            return (FaceIndicesNow[0][np.argmin(D3)], 0., 0., 1.)

    def FBC2PointOnSurface(self, FaceIndex, bc1, bc2, bc3, Vertices):
        ##
        Vert_indices = self.All_vertices[self.FacesDensePose[FaceIndex]] - 1
        ##
        p = Vertices[Vert_indices[0], :] * bc1 + \
            Vertices[Vert_indices[1], :] * bc2 + \
            Vertices[Vert_indices[2], :] * bc3
        ##
        return (p)