from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

# ----------------------------------------------------------------------------------- #
#  Association Head
# ----------------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AssociationHead(nn.Module):

    def __init__(self, cfg):
        super(AssociationHead, self).__init__()
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.cls_emb_u = nn.Sequential(
                nn.Linear(self.num_joints, 128, bias=False),
                nn.ReLU(inplace=True),
                )
        self.cls_emb_v = nn.Sequential(
                nn.Linear(self.num_joints, 128, bias=False),
                nn.ReLU(inplace=True)
                )
        self.loc_emb = nn.Sequential(
                nn.Conv2d(4, 128, 1, bias=False),
                nn.ReLU(inplace=True),
                )
        self.vis_emb = nn.Sequential(
            nn.Linear(32, 128, bias=False),
            nn.ReLU(inplace=True),
        )
        self.relation_predictor = nn.Sequential(
                nn.Conv2d(128*3, 1, 1, bias=False)
                )
        self.initialize_module_params()

    def initialize_module_params(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                print('init ', name)
                nn.init.constant_(param, 0)
            elif "weight" in name:
                print('init weight ',name)
                nn.init.normal_(param, std=0.001)
    def extract_geometry_relation(self, locations):
        # locations: N x 6, 4 for bbox 2 for keypoint
        # bbox_c_x = locations[:, 0]
        # bbox_c_y = locations[:, 1]
        # bbox_w = locations[:, 2]
        # bbox_h = locations[:, 3]
        kpt_x = locations[:, 3] / self.heatmap_size[0] # norm
        kpt_y = locations[:, 4] / self.heatmap_size[1] # norm
        delta_x = (locations[:, 3] - self.heatmap_size[0] / 2) / self.heatmap_size[0]
        delta_y = (locations[:, 4] - self.heatmap_size[1] / 2) / self.heatmap_size[1]
        # bbox relation
        dx_dist = (delta_x[:, None] - delta_x)**2
        dy_dist = (delta_y[:, None] - delta_y)**2
        # kpt position relation
        dx_kpt = (kpt_x[:, None] - kpt_x)**2
        dy_kpt = (kpt_y[:, None] - kpt_y)**2

        return torch.cat([dx_dist[..., None], dy_dist[..., None], dx_kpt[..., None], dy_kpt[..., None]], dim=2)

    def cls2onehot(self, cls_idx):
        assert (cls_idx < self.num_joints).all()

        onehot = torch.arange(self.num_joints).to(cls_idx.device)
        onehot = (cls_idx[:, None] == onehot).float()
        return onehot

    def cls_relation(self, onehot):
        feat1 = self.cls_emb_u(onehot)
        feat2 = self.cls_emb_v(onehot)
        cls_relation = feat1[:, None, :] * feat2
        return cls_relation

    def forward(self, vis_maps, locations, kpt_cls_idx):

        # locations: N x 5, bbox_c_x, bbox_c_y, bbox_s, kpt_x, kpt_y
        # kpt_cls_idx: N

        onehot = self.cls2onehot(kpt_cls_idx).to(vis_maps.device)

        cls_relation_feat = self.cls_relation(onehot)
        geo_relation = self.extract_geometry_relation(locations)
        # print('geo_feat:', geo_relation.size())
        geo_feat = self.loc_emb(geo_relation.permute(2, 0, 1)[None, ...]).squeeze(0).permute(1, 2, 0)
        # try:
        vis_feat = self.vis_emb(vis_maps)
        vis_rel_feat = vis_feat[:, None, :] * vis_feat
        # feat = vis_rel_feat
        feat = torch.cat([vis_rel_feat, cls_relation_feat, geo_feat], dim=2)
        logit = self.relation_predictor(feat.permute(2, 0, 1)[None, ...])
        ass_maps = torch.sigmoid(logit).squeeze(0).squeeze(0)
        # except:
        #     print()

        return ass_maps, feat

class KeypointRelationHead(nn.Module):

    def __init__(self, cfg):
        super(KeypointRelationHead, self).__init__()
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.loc_maps = self.build_geometry_embeddings((28, 28))
        num_channels = cfg['MODEL']['EXTRA']['STAGE4']['NUM_CHANNELS'][0]
        self.cls_emb_u = nn.Sequential(
                nn.Conv2d(self.num_joints + 1, num_channels, 3, padding=1, stride=2, bias=False),
                nn.ReLU(inplace=True),
                )
        self.loc_emb = nn.Sequential(
                nn.Conv2d(4, num_channels, 1, bias=False),
                nn.ReLU(inplace=True),
                )
        self.vis_emb = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, stride=2,padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.relation_predictor = nn.Sequential(
                nn.Conv2d(num_channels*3, 1, 1, bias=False)
                )
        self.initialize_module_params()

    def initialize_module_params(self):
        for name, param in self.named_parameters():
            print('init:', name)
            if 'loc_map' in name:
                print('ignore...')
            if "bias" in name:
                print('init ', name)
                nn.init.constant_(param, 0)
            elif "weight" in name:
                print('init weight ',name)
                nn.init.normal_(param, std=0.001)
    def build_geometry_embeddings(self, map_size=(64,64)):
        hm_w, hm_h = map_size[0], map_size[1]
        loc_map = np.zeros((4, hm_h, hm_w))
        abs_loc_x = np.arange(hm_w).astype(np.float32)
        abs_loc_y = np.arange(hm_h).astype(np.float32)
        loc_x = abs_loc_x / hm_w
        loc_y = abs_loc_y / hm_h
        offset_x = abs_loc_x - hm_w / 2.
        offset_y = abs_loc_y - hm_h / 2.
        loc_map[0] += np.repeat(loc_x.reshape((1,-1)), hm_h, 0)
        loc_map[1] += np.repeat(loc_y.reshape((-1,1)), hm_w, 1)
        loc_map[2] += np.repeat(offset_x.reshape((1, -1)), hm_h, 0)
        loc_map[3] += np.repeat(offset_y.reshape((-1, 1)), hm_w, 1)
        loc_map = loc_map.reshape((4,-1))
        loc_diff = loc_map[:, :, None] - loc_map[:, None, :]
        loc_diff = loc_diff.reshape((4, hm_h*hm_w, hm_h*hm_w))
        loc_map = torch.FloatTensor(loc_diff[np.newaxis])
        # loc_map = nn.Parameter(data=loc_map, requires_grad=False)
        return loc_map


    def cls_relation(self, onehot):
        b, c, h, w = onehot.size()
        feat1 = self.cls_emb_u(onehot)
        feat1 = F.interpolate(feat1, (28, 28), mode="bilinear", align_corners=False)
        feat1 = feat1.reshape((b, -1, 28**2)).permute(0, 2, 1)
        # feat2 = self.cls_emb_v(onehot).reshape((b, -1, h*w)).permute(0, 2, 1)
        cls_relation = feat1[:, None] * feat1[:, :, None]
        return cls_relation.permute(0, 3, 1, 2)

    def forward(self, vis_maps, kpt_cls_idx):
        '''

        :param vis_maps: N x C x H x W
        :param kpt_cls_idx: N x num_joints+1 x H x W
        :return: Association Maps: N x 1 x HW x HW
        '''
        b, c, h, w = vis_maps.size()
        cls_relation_feat = self.cls_relation(kpt_cls_idx)

        geo_feat = self.loc_emb(self.loc_maps.to(vis_maps.device))
        geo_feat = geo_feat.repeat(b, 1, 1, 1)

        # try:

        vis_feat = self.vis_emb(vis_maps)
        vis_feat = F.interpolate(vis_feat, (28, 28), mode="bilinear", align_corners=False)
        vis_feat = vis_feat.reshape((b, -1, 28**2)).permute(0, 2, 1)
        vis_rel_feat = vis_feat[:, None] * vis_feat[:, :, None]
        vis_rel_feat = vis_rel_feat.permute(0, 3, 1, 2)
        # vis_rel_feat = vis_rel_feat.reshape(vis_rel_feat.size())

        feat = torch.cat([vis_rel_feat, cls_relation_feat, geo_feat], dim=1)
        logit = self.relation_predictor(feat)
        ass_maps = torch.sigmoid(logit).squeeze(1)
        # ass_maps = F.interpolate(ass_maps, (h*w, h*w), mode="bilinear", align_corners=False).squeeze(1)
        return ass_maps

class SpatialRelationHead(nn.Module):
    def __init__(
            self, in_channels, inter_channels=None, dimension=2, sub_sample=False, bn_layer=True
    ):
        super(SpatialRelationHead, self).__init__()
        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.GroupNorm  # (32, hidden_dim) #nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.GroupNorm  # (32, hidden_dim)nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.GroupNorm  # (32, hidden_dim)nn.BatchNorm1d
        self.g = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                bn(8, self.in_channels),
            )
        else:
            self.W = conv_nd(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        if sub_sample:
            self.max_pool_layer = max_pool_layer
            up_sample_layer = nn.Sequential(
                *[nn.ConvTranspose2d(
                    in_channels=self.inter_channels,
                    out_channels=self.inter_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False),
                nn.BatchNorm2d(self.inter_channels, momentum=0.1),
                nn.ReLU(inplace=True)]
            )
            self.W = nn.Sequential(up_sample_layer, self.W)
        self.initialize_module_params()

    def initialize_module_params(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                print('init ', name)
                nn.init.constant_(param, 0)
            elif "weight" in name:
                print('init weight ',name)
                nn.init.normal_(param, std=0.001)

    def forward(self, x):
        """
        :param x: (b, c, h, w)
        :return:
        """
        batch_size = x.size(0)
        if self.sub_sample:
            x = self.max_pool_layer(x)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = x.view(batch_size, self.inter_channels, -1)
        affinity_matrix = torch.matmul(theta_x, phi_x)
        relation_score = F.sigmoid(affinity_matrix)

        relation_embs = torch.matmul(relation_score, g_x)
        relation_embs = relation_embs.permute(0, 2, 1).contiguous()
        relation_embs = relation_embs.view(batch_size, self.inter_channels, *x.size()[2:])
        relation_embs = self.W(relation_embs)
        return [relation_embs, relation_score]


class RelationLoss(nn.Module):

    def __init__(self):
        super(RelationLoss, self).__init__()

    def forward(self, relation_mat, gt, new=False):
        num_ins = relation_mat.size(0)
        if num_ins <= 1:
            return torch.tensor(0.).to(relation_mat.device)

        if not new:
            relation_mat = relation_mat + relation_mat.transpose(0, 1)


        loss = F.mse_loss(relation_mat, gt)
        if False and new and relation_mat.device == torch.device("cuda:0"):
            print('\n')

            format_str = '{:>4} : {:>3}/{:>3}, {:.4f}'
            try:
                one_item = relation_mat[gt > 0.5]
                zero_item = relation_mat[gt < 0.5]
                one_true_num = (one_item > 0.5).sum().item()
                zero_true_num = (zero_item < 0.5).sum().item()
                print(format_str.format('one', one_true_num, one_item.size(0), one_true_num * 1.0 / one_item.size(0)))
                print(one_item)
                print(format_str.format('zero', zero_true_num, zero_item.size(0), zero_true_num * 1.0 / zero_item.size(0)))
                print(zero_item)
            except ZeroDivisionError:
                print('Zero is divided!')
        return loss