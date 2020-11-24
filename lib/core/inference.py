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

def post(coords,batch_heatmaps):
    '''
    DARK post-pocessing
    :param coords: batchsize*num_kps*2
    :param batch_heatmaps:batchsize*num_kps*high*width
    :return:
    '''

    shape_pad = list(batch_heatmaps.shape)
    shape_pad[2] = shape_pad[2] + 2
    shape_pad[3] = shape_pad[3] + 2

    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            mapij=batch_heatmaps[i,j,:,:]
            maxori = np.max(mapij)
            mapij= cv2.GaussianBlur(mapij,(7, 7), 0)
            max = np.max(mapij)
            min = np.min(mapij)
            mapij = (mapij-min)/(max-min) * maxori
            batch_heatmaps[i, j, :, :]= mapij
    batch_heatmaps = np.clip(batch_heatmaps,0.001,50)
    batch_heatmaps = np.log(batch_heatmaps)
    batch_heatmaps_pad = np.zeros(shape_pad,dtype=float)
    batch_heatmaps_pad[:, :, 1:-1,1:-1] = batch_heatmaps
    batch_heatmaps_pad[:, :, 1:-1, -1] = batch_heatmaps[:, :, :,-1]
    batch_heatmaps_pad[:, :, -1, 1:-1] = batch_heatmaps[:, :, -1, :]
    batch_heatmaps_pad[:, :, 1:-1, 0] = batch_heatmaps[:, :, :, 0]
    batch_heatmaps_pad[:, :, 0, 1:-1] = batch_heatmaps[:, :, 0, :]
    batch_heatmaps_pad[:, :, -1, -1] = batch_heatmaps[:, :, -1 , -1]
    batch_heatmaps_pad[:, :, 0, 0] = batch_heatmaps[:, :, 0, 0]
    batch_heatmaps_pad[:, :, 0, -1] = batch_heatmaps[:, :, 0, -1]
    batch_heatmaps_pad[:, :, -1, 0] = batch_heatmaps[:, :, -1, 0]
    I = np.zeros((shape_pad[0],shape_pad[1]))
    Ix1 = np.zeros((shape_pad[0], shape_pad[1]))
    Iy1 = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1y1 = np.zeros((shape_pad[0],shape_pad[1]))
    Ix1_y1_ = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1_ = np.zeros((shape_pad[0], shape_pad[1]))
    Iy1_ = np.zeros((shape_pad[0], shape_pad[1]))
    coords = coords.astype(np.int32)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            I[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1]+1, coords[i, j, 0]+1]
            Ix1[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1]+1, coords[i, j, 0] + 2]
            Ix1_[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1]+1, coords[i, j, 0] ]
            Iy1[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1] + 2, coords[i, j, 0]+1]
            Iy1_[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1] , coords[i, j, 0]+1]
            Ix1y1[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1] + 2, coords[i, j, 0] + 2]
            Ix1_y1_[i, j] = batch_heatmaps_pad[i, j, coords[i, j, 1], coords[i, j, 0]]
    dx = 0.5 * (Ix1 -  Ix1_)
    dy = 0.5 * (Iy1 - Iy1_)
    D = np.zeros((shape_pad[0],shape_pad[1],2))
    D[:,:,0]=dx
    D[:,:,1]=dy
    D.reshape((shape_pad[0],shape_pad[1],2,1))
    dxx = Ix1 - 2*I + Ix1_
    dyy = Iy1 - 2*I + Iy1_
    dxy = 0.5*(Ix1y1- Ix1 -Iy1 + I + I -Ix1_-Iy1_+Ix1_y1_)
    hessian = np.zeros((shape_pad[0],shape_pad[1],2,2))
    hessian[:, :, 0, 0] = dxx
    hessian[:, :, 1, 0] = dxy
    hessian[:, :, 0, 1] = dxy
    hessian[:, :, 1, 1] = dyy
    inv_hessian = np.zeros(hessian.shape)
    # hessian_test = np.zeros(hessian.shape)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            hessian_tmp = hessian[i,j,:,:]
            try:
                inv_hessian[i,j,:,:] = np.linalg.inv(hessian_tmp)
            except LinAlgError:
                inv_hessian[i, j, :, :] = np.zeros((2,2))
            # hessian_test[i,j,:,:] = np.matmul(hessian[i,j,:,:],inv_hessian[i,j,:,:])
            # print( hessian_test[i,j,:,:])
    res = np.zeros(coords.shape)
    coords = coords.astype(np.float)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            D_tmp = D[i,j,:]
            D_tmp = D_tmp[:,np.newaxis]
            shift = np.matmul(inv_hessian[i,j,:,:],D_tmp)
            # print(shift.shape)
            res_tmp = coords[i, j, :] -  shift.reshape((-1))
            res[i,j,:] = res_tmp
    return res

def get_final_preds_HW(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)  # coords:(64, 17, 2)  maxvals (64, 17, 1) batch_heatmaps: (batch_size, 17, 64, 48)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    # if config.TEST.POST_PROCESS:
    #     for n in range(coords.shape[0]):
    #         for p in range(coords.shape[1]):
    #             hm = batch_heatmaps[n][p]  # (64,48)
    #             px = int(math.floor(coords[n][p][0] + 0.5))
    #             py = int(math.floor(coords[n][p][1] + 0.5))
    #             if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
    #                 diff = np.array(
    #                     [
    #                         hm[py][px+1] - hm[py][px-1],
    #                         hm[py+1][px]-hm[py-1][px]
    #                     ]
    #                 )
    #                 coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()  # preds:(64, 17, 2), the coordinate of heatmaps

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals  # preds:(64, 17, 2), the coordinate of intial images

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

def get_final_preds_dark(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)  # coords:(64, 17, 2)  maxvals (64, 17, 1) batch_heatmaps: (batch_size, 17, 64, 48)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    # tmp_coords = coords.copy()

    # post-processing
    if config.TEST.POST_PROCESS:
        coords = post(coords, batch_heatmaps)
        # for n in range(tmp_coords.shape[0]):
        #     for p in range(tmp_coords.shape[1]):
        #         hm = batch_heatmaps[n][p]  # (64,48)
        #         px = int(math.floor(tmp_coords[n][p][0] + 0.5))
        #         py = int(math.floor(tmp_coords[n][p][1] + 0.5))
        #         if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
        #             diff = np.array(
        #                 [
        #                     hm[py][px+1] - hm[py][px-1],
        #                     hm[py+1][px]-hm[py-1][px]
        #                 ]
        #             )
        #             tmp_coords[n][p] += np.sign(diff) * .25
    # coords = (coords + tmp_coords) * 0.5
    preds = coords.copy()  # preds:(64, 17, 2), the coordinate of heatmaps

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals # preds:(64, 17, 2), the coordinate of intial images

def get_final_preds_with_gt(config, batch_heatmaps, target_heatmaps, center, scale):

    coords, maxvals = get_max_preds(batch_heatmaps)  # coords:(64, 17, 2)  maxvals (64, 17, 1) batch_heatmaps: (batch_size, 17, 64, 48)
    coords_gt, maxvals_gt = get_max_preds(target_heatmaps)
    count = 0
    dis = 0
    for i in range(batch_heatmaps.shape[0]):
        for index, (vis, gt_coord, pred_coord) in enumerate(zip(maxvals_gt[i], coords_gt[i], coords[i])):
            if vis != 1:
                continue
            count += 1
            dis += np.sqrt((gt_coord[0]-pred_coord[0])**2 + (gt_coord[1]-pred_coord[1])**2)
    dis_aver = dis/count
    print('distance:', dis_aver)

    # compute the distance of preds and gt_joints


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

def get_final_preds_with_bboxes(config, batch_heatmaps, batch_bboxes, center, scale):

    batch_size = batch_heatmaps.shape[0]
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    image_width = 192
    image_height = 256

    # batch_tight_heatmaps = np.zeros_like(batch_heatmaps) - 1

    for k in range(batch_size):
        bboxes = batch_bboxes[k]
        xmin = bboxes[0] * image_width
        # xmin = int(math.floor(xmin/4 + 0.5)) - 12
        ymin = bboxes[1] * image_height
        # ymin = int(math.floor(ymin/4 + 0.5)) + 12
        xmax = bboxes[0] * image_width + bboxes[2] * image_width
        # xmax = int(math.floor(xmax/4 + 0.5)) - 12
        ymax = bboxes[1] * image_height + bboxes[3] * image_height
        # ymax = int(math.floor(ymax/4 + 0.5)) + 12

        # boundary check
        # xmin = xmin if xmin > 0 else 0
        # xmax = xmax if xmax < heatmap_width else heatmap_width - 1
        # ymin = ymin if ymin > 0 else 0
        # ymax = ymax if ymax < heatmap_height else heatmap_height - 1

        batch_heatmaps[k][:, ymin:ymax, xmin:xmax] = batch_heatmaps[k][:, ymin:ymax, xmin:xmax] + 0.2

    # batch_heatmaps = torch.from_numpy(batch_heatmaps)
    # batch_cropped_heatmaps = F.interpolate(batch_heatmaps, (image_height, image_width), mode="bilinear", align_corners=False)
    # batch_cropped_heatmaps = batch_cropped_heatmaps.numpy()
    # batch_tight_heatmaps = np.zeros_like(batch_cropped_heatmaps) - 1
    #
    # for k in range(batch_size):
    #     bboxes = batch_bboxes[k]
    #     xmin = bboxes[0] * image_width
    #     xmin = int(math.floor(xmin + 0.5)) - 16
    #     ymin = bboxes[1] * image_height
    #     ymin = int(math.floor(ymin + 0.5)) - 16
    #     xmax = bboxes[0] * image_width + bboxes[2] * image_width
    #     xmax = int(math.floor(xmax + 0.5)) + 16
    #     ymax = bboxes[1] * image_height + bboxes[3] * image_height
    #     ymax = int(math.floor(ymax + 0.5)) + 16
    #
    #     # boundary check
    #     xmin = xmin if xmin > 0 else 0
    #     xmax = xmax if xmax < image_width else image_width - 1
    #     ymin = ymin if ymin > 0 else 0
    #     ymax = ymax if ymax < image_height else image_height - 1
    #
    #     batch_tight_heatmaps[k][:, ymin:ymax, xmin:xmax] = batch_cropped_heatmaps[k][:, ymin:ymax, xmin:xmax]

    coords, maxvals = get_max_preds(batch_heatmaps)  # coords:(64, 17, 2)  maxvals (64, 17, 1) batch_heatmaps: (batch_size, 17, 64, 48)

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
    # for i in range(coords.shape[0]):
    #     preds[i] = transform_preds(
    #         coords[i], center[i], scale[i], [image_width, image_height]
    #     )

    return preds, maxvals  # preds:(64, 17, 2), the coordinate of cropped  images

def get_final_dp_preds(person_mask, Index_UV, U_uv, V_uv, ref_boxes, part_segms, input_image):
    if person_mask.ndim == 3:
        person_mask = np.expand_dims(person_mask, axis=0)
    if Index_UV.ndim == 3:
        Index_UV = np.expand_dims(Index_UV, axis=0)
    if U_uv.ndim == 3:
        U_uv = np.expand_dims(U_uv, axis=0)
    if V_uv.ndim == 3:
        V_uv = np.expand_dims(V_uv, axis=0)
    if input_image.ndim == 3:
        input_image = np.expand_dims(input_image, axis=0)
    if part_segms.ndim == 3:
        part_segms = np.expand_dims(part_segms, axis=0)


    K = 25
    cls_bodys = []
    heat_h, heat_w = person_mask.shape[2]*4, person_mask.shape[3]*4
    for ind, entry in enumerate(ref_boxes):
        # Compute ref box width and height
        bx = int(entry[2])
        by = int(entry[3])
        if bx > by:
            crop_w = heat_w
            crop_h = int(by * heat_w / bx)
        else:
            crop_h = heat_h
            crop_w = int(bx * heat_h / by)
        # preds[ind] axes are CHW; bring p axes to WHC
        CurAnnIndex = np.swapaxes(person_mask[ind], 0, 2)
        CurIndex_UV = np.swapaxes(Index_UV[ind], 0, 2)
        CurU_uv = np.swapaxes(U_uv[ind], 0, 2)
        CurV_uv = np.swapaxes(V_uv[ind], 0, 2)
        Cur_image = np.swapaxes(input_image[ind], 0, 2)
        Cur_parts = np.swapaxes(part_segms[ind], 0, 2)

        # Resize p from (HEATMAP_SIZE, HEATMAP_SIZE, c) to (int(bx), int(by), c)
        CurAnnIndex = cv2.resize(CurAnnIndex, None, None, fx=4, fy=4)
        CurIndex_UV = cv2.resize(CurIndex_UV, None, None, fx=4, fy=4)
        CurU_uv = cv2.resize(CurU_uv, None, None, fx=4, fy=4)
        CurV_uv = cv2.resize(CurV_uv, None, None, fx=4, fy=4)
        Cur_parts = cv2.resize(Cur_parts, None, None, fx=4, fy=4)

        CurAnnIndex = CurAnnIndex[0:crop_w, 0:crop_h]
        CurIndex_UV = CurIndex_UV[0:crop_w, 0:crop_h]
        CurU_uv = CurU_uv[0:crop_w, 0:crop_h]
        CurV_uv = CurV_uv[0:crop_w, 0:crop_h]
        Cur_image = Cur_image[0:crop_w, 0:crop_h]
        Cur_parts = Cur_parts[0:crop_w, 0:crop_h]


        CurAnnIndex = cv2.resize(CurAnnIndex, (by, bx))
        # CurAnnIndex = CurAnnIndex[:,:,np.newaxis]
        CurIndex_UV = cv2.resize(CurIndex_UV, (by, bx))
        CurU_uv = cv2.resize(CurU_uv, (by, bx))
        CurV_uv = cv2.resize(CurV_uv, (by, bx))
        Cur_image = cv2.resize(Cur_image, (by, bx))
        Cur_parts = cv2.resize(Cur_parts, (by, bx))


        # Bring Cur_Preds axes back to CHW
        CurAnnIndex = np.swapaxes(CurAnnIndex, 0, 2)
        CurIndex_UV = np.swapaxes(CurIndex_UV, 0, 2)
        CurU_uv = np.swapaxes(CurU_uv, 0, 2)
        CurV_uv = np.swapaxes(CurV_uv, 0, 2)
        Cur_image = np.swapaxes(Cur_image, 0, 2)
        Cur_parts = np.swapaxes(Cur_parts, 0, 2)

        # Removed squeeze calls due to singleton dimension issues
        CurPMaskProb = softmax(CurAnnIndex)
        CurAnnIndex = np.argmax(CurAnnIndex, axis=0)
        CurIndex_UV = np.argmax(CurIndex_UV, axis=0)
        Cur_parts = np.argmax(Cur_parts, axis=0)

        # Index_mask = (CurAnnIndex>0.).astype(np.float32)
        Index_mask = (CurPMaskProb[1] > 0.3).astype(np.float32)

        CurIndex_UV = CurIndex_UV * Index_mask

        output = np.zeros([9, int(by), int(bx)], dtype=np.float32)
        output[0] = CurIndex_UV
        output[3] = Cur_parts
        output[4] = Cur_image[0]
        output[5] = Cur_image[1]
        output[6] = Cur_image[2]
        output[7] = (CurAnnIndex>0.).astype(np.float32)

        for part_id in range(1, K):
            CurrentU = CurU_uv[part_id]
            CurrentV = CurV_uv[part_id]
            output[1, CurIndex_UV==part_id] = CurrentU[CurIndex_UV==part_id]
            output[2, CurIndex_UV==part_id] = CurrentV[CurIndex_UV==part_id]
        # vis_output = np.copy(output)
        # vis_output[1:,:,:]*=255
        # cv2.imwrite('dp_out.jpg',vis_output.transpose((1,2,0)))

        # add gt
        # gt_i = gt[0][ind]
        # gt_xy = gt[1][ind]
        # gt_uv = gt[2][ind]
        # gt_tag = gt[3][ind]
        # for g in range(len(gt_i)):
        #     if gt_tag[g] == 0:
        #         continue
        #     locs = np.asarray([[gt_xy[g, 0], gt_xy[g, 1]]])
        #     aug_locs = []
        #     aug_locs.append(locs)
        #     for step in [1, 2]:
        #         x_plus_locs = np.copy(locs)
        #         y_plus_locs = np.copy(locs)
        #         x_minus_locs = np.copy(locs)
        #         y_minus_locs = np.copy(locs)
        #
        #         x_plus_locs[:, 0] += step
        #         y_plus_locs[:, 0] += step
        #         x_minus_locs[:, 1] -= step
        #         y_minus_locs[:, 1] -= step
        #         aug_locs.append(x_plus_locs)
        #         aug_locs.append(y_plus_locs)
        #         aug_locs.append(x_minus_locs)
        #         aug_locs.append(y_minus_locs)
        #     locs = np.vstack(aug_locs)
        #     locs[locs < 0] = 0
        #     for loc in locs:
        #         loc[0] = loc[0] if loc[0] < bx else bx - 1
        #         loc[1] = loc[1] if loc[1] < by else by - 1
        #     output[0, locs[:, 1], locs[:, 0]] = gt_i[g]
        #     output[1, locs[:, 1], locs[:, 0]] = gt_uv[g,0]
        #     output[2, locs[:, 1], locs[:, 0]] = gt_uv[g,1]

            #vis debug
        # vis_out = np.copy(output)
        # vis_out[1]*=255
        # vis_out[2]*=255
        # vis_out = np.transpose(vis_out,(1,2,0))
        # cv2.imwrite(('gt_dp_pred.jpg'),vis_out)
        cls_bodys.append(output)

    # num_classes = 2
    # cls_bodys = [[] for _ in range(num_classes)]
    # person_idx = 1
    # cls_bodys[person_idx] = outputs

    return cls_bodys

def softmax(logits):
    # logits :C,H, W
    max_logits = np.max(logits,axis=0)[np.newaxis]
    logits = logits - np.repeat(max_logits, logits.shape[0], axis=0)
    logits = np.exp(logits)
    sum_logits = np.sum(logits,axis=0)[np.newaxis]
    probs = logits / (np.repeat(sum_logits, logits.shape[0], axis=0))
    return probs