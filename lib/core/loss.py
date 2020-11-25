# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)

class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta=1., loss_weight=0.1, size_average=True):
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        self.size_average = size_average
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
        sample_ind = target_weight.eq(1)
        diff = torch.abs(output[sample_ind] - target[sample_ind])
        cond = diff < self.beta
        loss = torch.where(cond, 0.5 * diff ** 2 / self.beta, diff - 0.5 * self.beta)
        if loss.nelement() == 0:
            weighted_loss = loss.sum()
        else:
            weighted_loss = loss.mean()
        # weighted_loss = loss * target_weight #.view((loss.size(0),1,loss.size(2),loss.size(3)))
        # weighted_loss = weighted_loss.sum() / target_weight.sum()
        return weighted_loss * self.loss_weight
        # if self.size_average:
        #     return weighted_loss.mean()
        # return weighted_loss.sum()

class WeightedNLLLoss(nn.Module):
    def __init__(self, loss_weight=0.1, size_average=True, balance_loss=False):
        super(WeightedNLLLoss, self).__init__()
        self.size_average = size_average
        self.activation = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss(reduction='none')
        self.loss_weight = loss_weight
        self.balance_loss = balance_loss

    def forward(self, output, target, target_weight):
        loss = self.nll_loss(self.activation(output), target.long())
        sample_ind = target_weight.gt(0)
        weighted_loss = loss[sample_ind] #* target_weight
        if self.balance_loss:
            prob = torch.exp(-loss)[sample_ind]
            weighted_loss = torch.pow((1 - prob), 2) * weighted_loss
        weighted_loss = weighted_loss.mean()# / target_weight.sum()
        return weighted_loss * self.loss_weight
        # if self.size_average:
        #     return weighted_loss.mean()
        # return weighted_loss.sum()

class SpatialNLLLoss(nn.Module):
    def __init__(self, size_average=True, balance_loss=False):
        super(SpatialNLLLoss, self).__init__()
        self.size_average = size_average
        self.activation = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss(reduction='none')
        self.balance_loss = balance_loss

    def forward(self, output, target):
        loss = self.nll_loss(self.activation(output), target.long())
        if self.balance_loss:
            prob = torch.exp(-loss)
            loss = torch.pow((1-prob), 2) * loss
        if self.size_average:
            return loss.mean()
        return loss.sum()

class SegmsMSELoss(nn.Module):
    def __init__(self):
        super(SegmsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)


    def forward(self, output, target):

        # preprocess
        batch_size = output.size(0)
        num_segms = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_segms, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_segms, -1)).split(1, 1)
        loss = 0

        for idx in range(num_segms):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_segms

class SegmsBCELoss(nn.Module):
    def __init__(self):
        super(SegmsBCELoss, self).__init__()
        ### mask weights
        # x_kernel = [[1, 0, -1],
        #             [2, 0, -2],
        #             [1, 0, -1]]
        # y_kernel = [[1, 2, 1],
        #             [0, 0, 0],
        #             [-1, -2, -1]]
        # x_kernel = torch.FloatTensor(x_kernel).expand(1, 1, 3, 3)
        # y_kernel = torch.FloatTensor(y_kernel).expand(1, 1, 3, 3)
        #
        # self.x_weight = nn.Parameter(data=x_kernel, requires_grad=False)
        # self.y_weight = nn.Parameter(data=y_kernel, requires_grad=False)
        # self.eps = 1e-8
        #####
        # self.criterion = nn.BCEWithLogitsLoss(size_average=True)
        self.criterion = nn.BCELoss(size_average=True)

    def forward(self, output, target):
        # mask weight generation
        # x_edge_h = nn.functional.conv2d(target, self.x_weight, padding=1)
        # x_edge_v = nn.functional.conv2d(target, self.y_weight, padding=1)
        # x_edge_h = x_edge_h * x_edge_h
        # x_edge_v = x_edge_v * x_edge_v
        # x_edge = x_edge_h + x_edge_v + self.eps
        # mask_weights = torch.sqrt(x_edge) + 1.
        # batch_size = output.size(0)
        # num_segms = output.size(1)
        heatmaps_pred = output.reshape((-1))
        heatmaps_gt = target.reshape((-1))
        # mask_weights = mask_weights.view(-1)
        loss = self.criterion(heatmaps_pred, heatmaps_gt.float())
        return loss
        # heatmaps_pred = output.reshape((batch_size, -1)).split(1, 1)
        # heatmaps_gt = target.reshape((batch_size, -1)).split(1, 1)
        # loss = 0
        #
        # for idx in range(num_segms):
        #     heatmap_pred = heatmaps_pred[idx].squeeze()
        #     heatmap_gt = heatmaps_gt[idx].squeeze()
        #     loss += self.criterion(heatmap_pred, heatmap_gt)

        # return loss / num_segms

class WeightedBCELoss(nn.Module):
    def __init__(self, loss_weight=0.1, size_average=True):
        super(WeightedBCELoss, self).__init__()
        self.size_average = size_average
        self.criterion = nn.BCELoss(size_average=True)
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
        heatmaps_pred = output.reshape((-1))
        heatmaps_gt = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        sample_ind = target_weight.eq(1)
        loss = self.criterion(heatmaps_pred[sample_ind], heatmaps_gt.float()[sample_ind])
        # weighted_loss = loss * target_weight
        # weighted_loss = weighted_loss.sum() / target_weight.sum()
        return loss * self.loss_weight

class UDPLosses(nn.Module):
    def __init__(self, use_target_weight):
        super(UDPLosses, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss_hm = 0
        loss_offset = 0
        num_joints = output.size(1) // 3
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx*3].squeeze()
            heatmap_gt = heatmaps_gt[idx*3].squeeze()
            offset_x_pred =  heatmaps_pred[idx*3+1].squeeze()
            offset_x_gt =  heatmaps_gt[idx*3+1].squeeze()
            offset_y_pred = heatmaps_pred[idx * 3 + 2].squeeze()
            offset_y_gt = heatmaps_gt[idx * 3 + 2].squeeze()
            if self.use_target_weight:
                loss_hm += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
                loss_offset += 0.5 * self.criterion(
                    heatmap_gt * offset_x_pred,
                    heatmap_gt * offset_x_gt
                )
                loss_offset += 0.5 * self.criterion(
                    heatmap_gt * offset_y_pred,
                    heatmap_gt * offset_y_gt
                )

        return [loss_hm / num_joints, loss_offset/num_joints]