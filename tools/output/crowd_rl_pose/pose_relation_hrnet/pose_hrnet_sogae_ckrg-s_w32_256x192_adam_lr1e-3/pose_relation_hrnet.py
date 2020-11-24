# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pickle
from .association import AssociationHead,KeypointRelationHead
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        super(PoseHighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)

        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels)

        # self.deconv_layers_p5 = self._make_deconv_layer(
        #     3,
        #     pre_stage_channels[-1],
        #     [pre_stage_channels[-1],pre_stage_channels[-1],pre_stage_channels[-1]],
        #     [4,4,4],
        # )
        # self.deconv_layers_p4 = self._make_deconv_layer(
        #     2,
        #     pre_stage_channels[-2],
        #     [pre_stage_channels[-2], pre_stage_channels[-2]],
        #     [4, 4],
        # )
        # self.deconv_layers_p3 = self._make_deconv_layer(
        #     1,
        #     pre_stage_channels[-3],
        #     [pre_stage_channels[-3]],
        #     [4],
        # )
        # self.conv_layers_p2 = nn.Sequential(
        #     nn.Conv2d(pre_stage_channels[-4], pre_stage_channels[-4],
        #               kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(pre_stage_channels[-4], momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=True),
        # )
        # self.pre_stage_channels = pre_stage_channels
        # self.refine_net = self._make_stack_conv_layer(4, np.int(np.sum(pre_stage_channels)), 256, 3)
        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def _make_deconv_layer(self, num_layers, dim_in, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        inplanes = dim_in
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))

            inplanes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_stack_conv_layer(self, num_layers, dim_in, num_filters, num_kernels):

        layers = []
        inplanes = dim_in
        planes = num_filters
        for i in range(num_layers):
            layers.append(
                nn.Conv2d(
                    in_channels=inplanes,
                    out_channels=planes,
                    kernel_size=num_kernels,
                    stride=1,
                    padding=1
                )
            )
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        y_list = self.stage2(x_list)
        x_list = []

        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        # p5
        p5 = self.deconv_layers_p5(y_list[-1])

        # p4
        p4 = self.deconv_layers_p4(y_list[-2])

        # p3
        p3 = self.deconv_layers_p3(y_list[-3])

        # p2
        p2 = self.conv_layers_p2(y_list[0])
        x = torch.cat([p2, p3, p4, p5], 1)
        x = self.refine_net(x)
        x = self.final_layer(x)
        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))

class RelationModule(nn.Module):

    def __init__(self, cfg, input_dims=256):

        super(RelationModule, self).__init__()

        relation_dims = cfg.MODEL.POSE_RELATION.KPT_RELATION_DIMS
        self.fc = nn.Linear(17, 600)
        self.R_emb = nn.Sequential(
            nn.Conv2d(relation_dims + input_dims, input_dims, 1, stride=1, padding=0))


    def forward(self, features, kpt_scores, kpt_rel_matrix, kpt_word_emb):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        tmp = self.fc(kpt_rel_matrix)
        kpt_rel_emb = F.linear(kpt_rel_matrix, kpt_word_emb.permute(1,0))

        emb_dims = kpt_rel_emb.size(1)
        B, n_dims, feat_h, feat_w = features.size(0), features.size(1), features.size(2), features.size(3)
        kpt_scores = F.interpolate(kpt_scores, (feat_h, feat_w), mode="bilinear", align_corners=False)
        kpt_scores = kpt_scores.reshape((B, kpt_scores.size(1), feat_h * feat_w)).permute(0, 2, 1)
        kpt_rel_emb = torch.matmul(kpt_scores, kpt_rel_emb)
        kpt_rel_emb = kpt_rel_emb.permute(0, 2, 1).reshape((B, emb_dims, feat_h, feat_w))
        vis_rel_feats = torch.cat([features, kpt_rel_emb], 1)
        vis_rel_feats = self.R_emb(vis_rel_feats)
        return vis_rel_feats


class KTMachine(nn.Module):

    def __init__(self, cfg, weight_size):

        super(KTMachine, self).__init__()

        relation_dims = cfg.MODEL.POSE_RELATION.KPT_RELATION_DIMS
        out_weight_size = 256 * 1 * 1

        # kpt word emb
        kpt_word_emb_dir = cfg.MODEL.POSE_RELATION.KPT_WORD_EMB_DIR

        kpt_word_emb = pickle.load(open(kpt_word_emb_dir, 'rb'))
        kpt_word_emb = torch.FloatTensor(kpt_word_emb)
        self.kpt_word_emb = nn.Parameter(data=kpt_word_emb, requires_grad=True)
        kpt_transformer = []
        kpt_transformer.append(nn.Linear(weight_size, out_weight_size))
        kpt_transformer.append(nn.LeakyReLU(0.02))
        kpt_transformer.append(nn.Linear(out_weight_size, out_weight_size))
        self.kpt_transformer = nn.Sequential(*kpt_transformer)
        self.initialize_module_params()

    def generate_knowledge_graph(self):
        norm = torch.norm(self.kpt_word_emb, p=2, dim=1).reshape((self.kpt_word_emb.size(0),1))
        kpt_word_emb = self.kpt_word_emb / norm
        sim_matrix = torch.matmul(kpt_word_emb, kpt_word_emb.permute(1,0))
        return sim_matrix

    def initialize_module_params(self):
        for name, param in self.named_parameters():
            if 'kpt_word_emb' in name:
                print('ignore init ', name)
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, weights):
        n_out, n_in, h, w = weights.size(0), weights.size(1), weights.size(2), weights.size(3)
        # print(weights.size())
        weights = weights.reshape((n_out,n_in*w*h))
        sim_matrix = self.generate_knowledge_graph()
        kt_weights = torch.matmul(sim_matrix, weights)
        kt_weights = self.kpt_transformer(kt_weights)
        kt_weights = kt_weights.reshape((n_out, 256, h, w))
        return kt_weights

class KnowledgeTransferKptNet(PoseHighResolutionNet):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        kpt_weight_size = self.final_layer.weight.size(1) * self.final_layer.weight.size(2) *self.final_layer.weight.size(3)
        # relation modules
        self.kt_machine = KTMachine(cfg, kpt_weight_size)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        y_list = self.stage2(x_list)
        x_list = []

        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        inter_kpt_scores = self.final_layer(y_list[0])
        # p5
        p5 = self.deconv_layers_p5(y_list[-1])

        # p4
        p4 = self.deconv_layers_p4(y_list[-2])

        # p3
        p3 = self.deconv_layers_p3(y_list[-3])

        # p2
        p2 = self.conv_layers_p2(y_list[0])
        x = torch.cat([p2, p3, p4, p5], 1)
        x = self.refine_net(x)
        refine_weights = self.kt_machine(self.final_layer.weight)
        kpt_scores = nn.functional.conv2d(x, weight=refine_weights, padding=0, stride=1)
        #
        return [kpt_scores, inter_kpt_scores]

class PoseRelationHighResolutionNet(PoseHighResolutionNet):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # kpt relation matrix
        kpt_rel_matrix_dir = cfg.MODEL.POSE_RELATION.KPT_RELATION_DIR
        kpt_rel_matrix = pickle.load(open(kpt_rel_matrix_dir, 'rb'))
        kpt_rel_matrix = torch.FloatTensor(kpt_rel_matrix)
        self.kpt_rel_matrix = nn.Parameter(data=kpt_rel_matrix, requires_grad=True)
        # kpt word emb
        kpt_word_emb_dir = cfg.MODEL.POSE_RELATION.KPT_WORD_EMB_DIR
        kpt_word_emb = pickle.load(open(kpt_word_emb_dir, 'rb'))
        kpt_word_emb = torch.FloatTensor(kpt_word_emb)
        self.kpt_word_emb = nn.Parameter(data=kpt_word_emb, requires_grad=True)
        # relation modules
        self.relation_module_p2 = RelationModule(cfg, self.pre_stage_channels[0])
        self.relation_module_p3 = RelationModule(cfg, self.pre_stage_channels[1])
        self.relation_module_p4 = RelationModule(cfg, self.pre_stage_channels[2])
        self.relation_module_p5 = RelationModule(cfg,self.pre_stage_channels[3])


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        y_list = self.stage2(x_list)
        x_list = []

        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        # p5
        p5 = self.deconv_layers_p5(y_list[-1])

        # p4
        p4 = self.deconv_layers_p4(y_list[-2])

        # p3
        p3 = self.deconv_layers_p3(y_list[-3])

        # p2
        p2 = self.conv_layers_p2(y_list[0])
        x = torch.cat([p2, p3, p4, p5], 1)
        x = self.refine_net(x)
        inter_kpt_scores = self.final_layer(x)
        # relation embedding
        kpt_rel_p5 = self.relation_module_p5(y_list[-1], inter_kpt_scores, self.kpt_rel_matrix, self.kpt_word_emb)
        kpt_rel_p5 = self.deconv_layers_p5(kpt_rel_p5)

        kpt_rel_p4 = self.relation_module_p4(y_list[-2], inter_kpt_scores, self.kpt_rel_matrix, self.kpt_word_emb)
        kpt_rel_p4 = self.deconv_layers_p4(kpt_rel_p4)

        kpt_rel_p3 = self.relation_module_p3(y_list[-3], inter_kpt_scores, self.kpt_rel_matrix, self.kpt_word_emb)
        kpt_rel_p3 = self.deconv_layers_p3(kpt_rel_p3)

        kpt_rel_p2 = self.relation_module_p2(y_list[0], inter_kpt_scores, self.kpt_rel_matrix, self.kpt_word_emb)
        kpt_rel_p2 = self.conv_layers_p2(kpt_rel_p2)

        kpt_rel_x = torch.cat([kpt_rel_p2, kpt_rel_p3, kpt_rel_p4, kpt_rel_p5], 1)
        kpt_rel_x = self.refine_net(kpt_rel_x)
        kpt_scores = self.final_layer(kpt_rel_x)
        return [kpt_scores, inter_kpt_scores]


class AssociationKptNet(PoseHighResolutionNet):

    def __init__(self, cfg, is_train=True, **kwargs):
        super().__init__(cfg, **kwargs)
        self.is_train = is_train
        # kpt_weight_size = self.final_layer.weight.size(1) * self.final_layer.weight.size(2) *self.final_layer.weight.size(3)
        num_channels = self.stage4_cfg['NUM_CHANNELS'][0]
        self.vis_rel_layer = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # relation modules
        self.association_net = AssociationHead(cfg)


    def _forward_backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        y_list = self.stage2(x_list)
        x_list = []

        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        return y_list

    def forward(self, x, meta=None, is_train=False):
        # visual embedding
        features = self._forward_backbone(x)
        # multi-instances kpt maps & visual features
        all_kpt_scores = self.final_layer(features[0])
        vis_maps = self.vis_rel_layer(features[0])
        vis_maps = F.relu(vis_maps)
        interference_scores = None
        # relation modeling
        # step 1 prepare candidate points
        # with torch.no_grad():
        #     if meta is not None:
        #         # use ground truth
        #         bbox_all, class_all, locs_all, \
        #         kpt_x_all, kpt_y_all = _extract_at_points_from_labels(meta)
        #         interference_scores = meta['interference_maps']
        #     else:
        #         # use predictions
        #         bbox_all, class_all, locs_all, \
        #         kpt_x_all, kpt_y_all = _extract_at_points_from_predictions(all_kpt_scores.detach().cpu().numpy(), meta)
        ampas_all = []
        rel_loss = torch.mean(vis_maps)
        # compute assocation maps
        # if len(bbox_all) > 0:
        #     num_bbox = len(bbox_all)
        #     for i in range(num_bbox):
        #         b_i = bbox_all[i].long()
        #         c_i = class_all[i]
        #         locs_i = locs_all[i]
        #         kpt_x_i = kpt_x_all[i].long()
        #         kpt_y_i = kpt_y_all[i].long()
        #
        #         # vis association feats
        #         vis_embs = vis_maps[b_i, slice(None), kpt_y_i, kpt_x_i].float()
        #         # association modeling with 3 types of feature
        #         amaps, a_feats = self.association_net(vis_embs, locs_i, c_i)
        #         if is_train:
        #             gt_amap = meta['association_maps'][i][:amaps.size(0), :amaps.size(1)].float().to(amaps.device)
        #             rel_loss += (F.mse_loss(amaps, gt_amap) / num_bbox)
        #         else:
        #             rel_loss += torch.mean(amaps)
        #
        #         ampas_all.append(amaps)

        # if interference_scores is None and len(ampas_all)!=0:
        #     interference_scores = []
        #     with torch.no_grad():
        #         for amap, locs in enumerate(ampas_all, locs_all):
        #             amap = amap.cpu().numpy()
        #             locs = locs.cpu().numpy()
        #             interference_scores.append(_generate_interference_maps(amap,locs[:,-3:]),(all_kpt_scores.size[1], all_kpt_scores.size[0]))
        #         interference_scores = torch.cat(interference_scores, dim=0).to(all_kpt_scores.device)
        # else:
        #     interference_scores = torch.zeros(all_kpt_scores.size()).to(all_kpt_scores.device)

        # remove inteference logits
        kpt_scores = all_kpt_scores - 0 #- interference_scores
        #
        return [kpt_scores, all_kpt_scores, rel_loss]

class RelationKptNet(PoseHighResolutionNet):

    def __init__(self, cfg, is_train=True, **kwargs):
        super().__init__(cfg, **kwargs)
        self.is_train = is_train
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        num_channels = self.stage4_cfg['NUM_CHANNELS'][0]
        self.vis_rel_layer = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # relation modules
        self.association_net = KeypointRelationHead(cfg)
        self.refine_layer = nn.Conv2d(
            in_channels=num_channels+cfg.MODEL.NUM_JOINTS,
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=1,
            stride=1,
            padding=0
        )


    def _forward_backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        y_list = self.stage2(x_list)
        x_list = []

        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        return y_list

    def forward(self, x, kpt_cat_maps=None, interference_scores = None, is_train=False):
        # visual embedding
        features = self._forward_backbone(x)
        # multi-instances kpt maps & visual features
        all_kpt_scores = self.final_layer(features[0])
        vis_maps = self.vis_rel_layer(features[0])
        vis_maps = F.relu(vis_maps)
        # relation modeling
        # step 1 prepare candidate points
        if is_train:
            assert kpt_cat_maps is not None,'kpt onehot is none'
            assert interference_scores is not None, 'interference score is none'
            assert kpt_cat_maps.size(1) == 15,'class idx invalid'
        else:
            kpt_cat_maps = _generate_category_maps_from_predictions(all_kpt_scores.detach().cpu().numpy()).to(all_kpt_scores.device)

        # compute assocation maps
        amaps = self.association_net(vis_maps, kpt_cat_maps)
        if interference_scores is None:
            interference_scores = []
            with torch.no_grad():
                up_size = all_kpt_scores.size(2)*all_kpt_scores.size(3)
                amaps = F.interpolate(amaps.unsqueeze(1), (up_size, up_size), mode="bilinear", align_corners=False).squeeze(1)
                for amap, hmap in zip(amaps, all_kpt_scores):
                    amap = amap.detach().cpu().numpy()
                    hmap = hmap.detach().cpu().numpy()
                    interference_scores.append(_generate_interference_maps_from_predictions(amap, hmap, self.heatmap_size))
                interference_scores = torch.cat(interference_scores, dim=0).to(all_kpt_scores.device)
                # print(interference_scores.size())


        # remove inteference logits
        kpt_scores = all_kpt_scores - interference_scores
        refine_feat = torch.cat([features[0],kpt_scores],dim=1)
        kpt_scores = self.refine_layer(refine_feat)
        #
        return [kpt_scores, all_kpt_scores, amaps]

class KBRelationKptNet(PoseHighResolutionNet):

    def __init__(self, cfg, is_train=True, **kwargs):
        super().__init__(cfg, **kwargs)
        self.is_train = is_train
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        num_channels = self.stage4_cfg['NUM_CHANNELS'][0]
        self.vis_rel_layer = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # relation modules
        self.association_net = KeypointRelationHead(cfg)
        # knowledge based refine modules
        kpt_weight_size = self.final_layer.weight.size(1) * self.final_layer.weight.size(
            2) * self.final_layer.weight.size(3)
        self.refine_net = self._make_stack_conv_layer(4, num_channels+cfg.MODEL.NUM_JOINTS, 256, 3)
        self.kt_machine = KTMachine(cfg, kpt_weight_size)


    def _forward_backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        y_list = self.stage2(x_list)
        x_list = []

        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        return y_list

    def forward(self, x, kpt_cat_maps=None, interference_scores = None, is_train=False):
        # visual embedding
        features = self._forward_backbone(x)
        # multi-instances kpt maps & visual features
        all_kpt_scores = self.final_layer(features[0])
        vis_maps = self.vis_rel_layer(features[0])
        vis_maps = F.relu(vis_maps)
        # relation modeling
        # step 1 prepare candidate points
        if is_train:
            assert kpt_cat_maps is not None,'kpt onehot is none'
            assert interference_scores is not None, 'interference score is none'
        else:
            kpt_cat_maps = _generate_category_maps_from_predictions(all_kpt_scores.detach().cpu().numpy()).to(all_kpt_scores.device)

        # compute assocation maps
        amaps = self.association_net(vis_maps, kpt_cat_maps)
        if interference_scores is None:
            interference_scores = []
            with torch.no_grad():
                up_size = all_kpt_scores.size(2)*all_kpt_scores.size(3)
                amaps = F.interpolate(amaps.unsqueeze(1), (up_size, up_size), mode="bilinear", align_corners=False).squeeze(1)
                for amap, hmap in zip(amaps, all_kpt_scores):
                    amap = amap.detach().cpu().numpy()
                    hmap = hmap.detach().cpu().numpy()
                    interference_scores.append(_generate_interference_maps_from_predictions(amap, hmap, self.heatmap_size))
                interference_scores = torch.cat(interference_scores, dim=0).to(all_kpt_scores.device)
                # print(interference_scores.size())

        # remove inteference logits
        with torch.no_grad():
            calibration_map = all_kpt_scores - interference_scores
        refine_feat = torch.cat([features[0], calibration_map], dim=1)
        refine_feat = self.refine_net(refine_feat)
        refine_weights = self.kt_machine(self.final_layer.weight)
        kpt_scores = nn.functional.conv2d(refine_feat, weight=refine_weights, padding=0, stride=1)
        #
        return [kpt_scores, all_kpt_scores, amaps]

def _extract_at_points_from_labels(labels):
    relation_points = labels['relation_joints']
    num_points = labels['num_points']
    i_bbox_all = []
    i_class_all = []
    i_locs_all = []
    i_kpt_x =[]
    i_kpt_y = []
    for i in range(len(num_points)):
        if num_points[i] == 0:
            continue
        i_bbox_all += [(torch.zeros((num_points[i],)) + i).long()]
        points = relation_points[i, :num_points[i], :]
        i_class_all += [points[:, 6].long()]
        i_locs_all += [points[:, :7].float()]
        i_kpt_x += [points[:, 4].long()]
        i_kpt_y += [points[:, 5].long()]
    # i_bbox_all = torch.cat(i_bbox_all, dim=0).long()
    # i_class_all = torch.cat(i_class_all, dim=0).long()
    # i_locs_all = torch.cat(i_locs_all, dim=0).long()
    # i_kpt_x = torch.cat(i_kpt_x, dim=0).long()
    # i_kpt_y = torch.cat(i_kpt_y, dim=0).long()
    return i_bbox_all, i_class_all, i_locs_all, i_kpt_x, i_kpt_y

def _extract_at_points_from_predictions(heatmaps, meta):
    relation_points = _extract_points_from_heatmaps(heatmaps)
    if len(relation_points) == 0:
        return [], [], [], [], []
    for i in range(len(heatmaps)):
        # center = meta['center'][i].numpy()
        # obj_size = meta['obj_size'][i].numpy()
        i_cand_points = relation_points[i]
        # bbox = np.asarray([center[0], center[1], obj_size[0], obj_size[1]]).reshape((1, -1))
        if len(i_cand_points) > 0:
            bbox = np.asarray([-1,-1,-1,-1]).reshape((1, -1))
            bbox = np.repeat(bbox, len(i_cand_points),0)
            relation_points[i] = np.hstack([bbox, i_cand_points])
        else:
            relation_points[i] = np.asarray([-1,-1,-1,-1,0,0,0],dtype=np.int32).reshape((1, -1))
    #
    i_bbox_all = []
    i_class_all = []
    i_locs_all = []
    i_kpt_x = []
    i_kpt_y = []
    for i in range(len(relation_points)):
        points = torch.from_numpy(relation_points[i])
        i_bbox_all += [torch.zeros((len(points),)) + i]
        i_class_all += [points[:, -1]]
        i_locs_all += [points[:,:6].float()]
        i_kpt_x += [points[:, 4].long()]
        i_kpt_y += [points[:, 5].long()]
    return i_bbox_all, i_class_all, i_locs_all, i_kpt_x, i_kpt_y


def _extract_points_from_heatmaps(heatmaps, thresh=0.95):
    points_all = []
    for _, heatmaps_per_bbox in enumerate(heatmaps):
        points = []
        for i in range(len(heatmaps_per_bbox)):
            locs = np.where(heatmaps_per_bbox[i] >= thresh)
            if len(locs[0])==0:
                locs = np.where(heatmaps_per_bbox[i] >= heatmaps_per_bbox[i].max())
            scores = heatmaps_per_bbox[i][locs]
            locs = np.asarray(locs).transpose()
            ind = np.argsort(scores)[::-1]
            scores = scores[ind]
            locs = locs[ind]
            dist = compute_points_dist(locs)
            remove_tags = np.zeros((len(scores)))
            target_points = []
            for j in range(len(scores)):
                if remove_tags[j] == 1:
                    continue
                target_points += [locs[j, 0], locs[j, 1], i, scores[j]]
                current_score = scores[j]
                remove_cands = dist[j].flatten()
                remove_targets = np.logical_and(remove_cands < 3, scores < current_score)
                remove_targets[j] = 0
                remove_tags[remove_targets] = 1
            points += target_points
        if len(points)>0:
            points_all.append(np.asarray(points).reshape((-1, 3)))
        else:
            points_all.append([])
    return points_all

def compute_points_dist(points):
    loc_x = points[:,0].reshape((-1,1))
    loc_y = points[:,1].reshape((-1,1))
    dx = (loc_x[:,None] - loc_x)**2
    dy = (loc_y[:,None] - loc_y)**2
    dist = np.sqrt(dx + dy)
    return dist

def _generate_gaussian_map(joint, map_size=(64,64)):
    g_map = np.zeros((map_size[1], map_size[0]))
    sigma = 1
    tmp_size = sigma * 3

    mu_x = int(joint[0])
    mu_y = int(joint[1])
        # Check that any part of the gaussian is in-bounds
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= map_size[0] or ul[1] >= map_size[1] \
            or br[0] < 0 or br[1] < 0:
        return g_map

    # # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], map_size[0]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], map_size[1]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], map_size[0])
    img_y = max(0, ul[1]), min(br[1], map_size[1])
    g_map[img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return g_map

def _generate_category_maps_from_predictions(heatmaps, thresh=0.95):
    bg_map = (np.max(heatmaps, axis=1) < thresh).astype(np.float32)
    fg_map = (heatmaps >= thresh).astype(np.float32)
    onehot = np.concatenate([bg_map[:,None], fg_map], axis=1)
    return torch.from_numpy(onehot)
def _generate_interference_maps_from_predictions(amap, kpt_map, map_size, thresh=0.99):

    inter_maps_all = []
    for k, hm in enumerate(kpt_map):
        inter_maps = np.zeros((map_size[1], map_size[0]))
        locs = np.where(hm >= thresh)
        if len(locs[0]) == 0:
            locs = np.where(hm >= hm.max())
        scores = hm[locs]
        locs = np.asarray(locs).transpose()
        idx_amap = locs[:,0]*map_size[0] + locs[:,1]
        amap_scores = amap[idx_amap, idx_amap]
        scores = (scores + amap_scores) / 2
        ind = np.argsort(scores)[::-1]
        scores = scores[ind]
        locs = locs[ind]
        dist = compute_points_dist(locs)
        remove_tags = np.zeros((len(scores)))
        target_points = []
        # extract candidates
        for j in range(len(scores)):
            if remove_tags[j] == 1:
                continue
            target_points.append([locs[j, 0], locs[j, 1], k, scores[j]])
            current_score = scores[j]
            remove_cands = dist[j].flatten()
            remove_targets = np.logical_and(remove_cands < 3, scores < current_score)
            remove_targets[j] = 0
            remove_tags[remove_targets] = 1
        target_points = np.asarray(target_points)
        max_score = np.max(target_points[:,3])
        for p in target_points:
            if p[3] < max_score:
                map = _generate_gaussian_map([p[1], p[0]], map_size)
                inter_maps = np.maximum(inter_maps, map)
        inter_maps_all.append(inter_maps[None])
        # print(inter_maps_all[-1].shape)
    inter_maps_all = torch.from_numpy(np.concatenate(inter_maps_all, axis=0)).float()

    return inter_maps_all.unsqueeze(0)

def _generate_interference_maps(amap, kpts, map_size):
    search_order = [12,13,0,1,6,7,2,3,4,5,8,9,10,11]
    tags = np.zeros((len(amap)))
    inter_maps = np.zeros((map_size))
    for part_idx in search_order:
        ind = np.where(kpts[:, 2] == part_idx)
        if len(ind) <= 1:
            continue

        a_scores = np.sum(amap[ind], axis=1)
        score_ind = np.argsort(a_scores)[::-1]
        tags[ind[score_ind[1:]]] = 1
    for t in len(tags):
        if tags[t] == 1:
            map = _generate_gaussian_map(kpts[t], map_size)
            inter_maps[int(kpts[2])] = np.maximum(inter_maps[int(kpts[2])], map)

    return torch.from_numpy(inter_maps[np.newaxis])

def get_pose_net(cfg, is_train, **kwargs):
    # model = PoseHighResolutionNet(cfg, **kwargs)
    # model = PoseRelationHighResolutionNet(cfg, **kwargs)
    # model = KnowledgeTransferKptNet(cfg, **kwargs)
    # model = AssociationKptNet(cfg, **kwargs)
    # model = RelationKptNet(cfg, **kwargs)
    model = KBRelationKptNet(cfg, **kwargs)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
