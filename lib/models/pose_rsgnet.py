from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from .registry import Registry
from .association import SpatialRelationHead
import pickle
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
POSE_RSG_NET_REGISTRY = Registry("POSE_RSG_NET")

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

class GlobalAveragePooling2D(nn.Module):
    def __init__(self, output_size):
        super(GlobalAveragePooling2D, self).__init__()
        self.pooler = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.pooler(x)
        x = x.reshape(b, c)
        return x

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

@POSE_RSG_NET_REGISTRY.register()
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
            self.stage4_cfg, num_channels, multi_scale_output=False)
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

        x = self.final_layer(y_list[0])
        return None, x, None

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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

@POSE_RSG_NET_REGISTRY.register()
class TargetAwareRelationParserNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        super(TargetAwareRelationParserNet, self).__init__()

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
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.multi_final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']
        num_for_channels = cfg['MODEL']['EXTRA']['OUTPUT_CONVS'][0]  # 32
        self.num_joints = cfg.MODEL.NUM_JOINTS

        # visual features
        self.vis_conv = nn.Sequential(
            *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )

        # type features
        # type_features = torch.randn(self.num_joints, num_for_channels*4)
        # self.type_features = nn.Parameter(data=type_features, requires_grad=True)
        # self.type_fc = nn.Sequential(
        #     *[nn.Linear(num_for_channels*4, num_for_channels*4, bias=False),
        #       nn.BatchNorm1d(num_for_channels*4, momentum=BN_MOMENTUM),
        #       nn.ReLU(inplace=True)]
        # )
        # self.type_conv = nn.Sequential(
        #     *[nn.Conv2d(num_for_channels*4, num_for_channels, kernel_size=3, stride=1, padding=1,
        #                 bias=False),
        #       nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
        #       nn.ReLU(inplace=True)]
        # )

        # location features
        # self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        # self.loc_features = self.build_geometry_embeddings((self.heatmap_size[0], self.heatmap_size[1]))
        # self.loc_conv = nn.Sequential(
        #     *[nn.Conv2d(4, num_for_channels, 1, bias=False),
        #     nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=True)]
        # )

        # contact features
        # self.contact_conv = nn.Sequential(
        #     *[nn.Conv2d(num_for_channels*2, num_for_channels*2, kernel_size=3, stride=1, padding=1,
        #                 bias=False),
        #       nn.BatchNorm2d(num_for_channels*2, momentum=BN_MOMENTUM),
        #       nn.ReLU(inplace=True)]
        # )
        #
        # self.predict_contact_net = nn.Sequential(
        #     *[nn.Conv2d(num_for_channels*2, num_for_channels, kernel_size=3, stride=1, padding=1,
        #                 bias=False),
        #       nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
        #       nn.ReLU(inplace=True)]
        # )

        self.relation_head = SpatialRelationHead(num_for_channels, num_for_channels,
                                                 sub_sample=cfg.MODEL.RELATION_SUB_SAMPLE)
        self.kpt_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels * 2, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )



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
        loc_map = torch.FloatTensor(loc_map[np.newaxis])
        loc_map = nn.Parameter(data=loc_map, requires_grad=False)
        return loc_map

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

        # enhance visual features
        vis_features = y_list[0]  # tensor ([1,32,64,48]), B*C*H*W
        multi_predict_scores = self.multi_final_layer(vis_features)  # y_list[0](1,32,64,48)  x(1,17,64,48), multi-joints scores loss

        # visual information
        vis_features = self.vis_conv(vis_features)

        # # type information
        # predict_size = multi_predict_scores.size()
        # tmp_predict_scores = torch.reshape(multi_predict_scores, (predict_size[0], predict_size[1], predict_size[2] * predict_size[3]))  # B*17*S
        # tmp_predict_scores = tmp_predict_scores.permute(0, 2, 1)  # B*S*17
        # tmp_type_features = self.type_features.to(tmp_predict_scores.device)  # 17*T
        # tmp_type_features = self.type_fc(tmp_type_features)
        # type_size = tmp_type_features.size()
        # type_features = torch.matmul(tmp_predict_scores, tmp_type_features)  # B*S*T
        # type_features = type_features.permute(0, 2, 1)
        # type_features = torch.reshape(type_features, (predict_size[0], type_size[1], predict_size[2], predict_size[3]))
        # type_features = self.type_conv(type_features)  # B*T*H*W

        # location information
        # predict_size = multi_predict_scores.size()
        # loc_features = self.loc_conv(self.loc_features)  # B*L*H*W64
        # loc_features = loc_features.repeat(predict_size[0], 1, 1, 1)

        # contact these three information
        # final_vis_features = torch.cat((vis_features, loc_features), dim=1)
        # final_vis_features = self.contact_conv(final_vis_features)  # to smooth contact features
        # final_vis_features = self.predict_contact_net(final_vis_features)  # 32*2 -> 32


        # relation_feats, relation_scores = self.relation_head(final_vis_features)
        relation_feats, relation_scores = self.relation_head(vis_features)

        # keypoints branch
        # B*C*H*W
        # kpt_features = torch.cat((relation_feats, final_vis_features), dim=1)
        kpt_features = torch.cat((relation_feats, vis_features), dim=1)
        kpt_features = self.kpt_net(kpt_features)
        predict_scores = self.final_layer(kpt_features)  # 32 -> 17

        return predict_scores, multi_predict_scores, None

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)

                for name, _ in m.named_parameters():
                    print('init:', name)
                    if name in ['bias']:
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

@POSE_RSG_NET_REGISTRY.register()
class SkeletonGraphNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        super(SkeletonGraphNet, self).__init__()

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
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        kpt_weight_size = self.final_layer.weight.size(1) * self.final_layer.weight.size(
            2) * self.final_layer.weight.size(3)
        self.kt_machine = KTMachine(cfg, kpt_weight_size)

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

        num_for_channels = cfg['MODEL']['EXTRA']['OUTPUT_CONVS'][0]  # 32
        self.predict_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )

        self.limbs_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )

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

        # keypoints branch
        vis_features = y_list[0]  # tensor ([1,32,64,48]), B*C*H*W
        vis_features = self.predict_net(vis_features)  # B*C*H*W
        kpt_scores = self.final_layer(vis_features)  # y_list[0](1,32,64,48)  x(1,17,64,48)

        # limbs branch
        limbs_features = y_list[0]  # tensor ([1,32,64,48]), B*C*H*W
        limbs_features = self.limbs_net(limbs_features)  # B*C*H*W
        refine_weights = self.kt_machine(self.final_layer.weight)
        limbs_scores = nn.functional.conv2d(limbs_features, weight=refine_weights, padding=0, stride=1)
        limbs_scores = torch.sigmoid(limbs_scores)

        return None, kpt_scores, limbs_scores

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    print('init:', name)
                    if name in ['bias']:
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

class KTMachine(nn.Module):

    def __init__(self, cfg, weight_size):

        super(KTMachine, self).__init__()

        out_weight_size = weight_size

        num_joints = cfg.MODEL.NUM_JOINTS
        num_limbs = cfg.MODEL.NUM_LIMBS
        matrix_limb = torch.randn(num_limbs,  num_joints)  # random initial 18*17
        self.matrix_limb = nn.Parameter(data=matrix_limb, requires_grad=True)

        real_matrix_limb = self.generate_commonsense_knowledge_graph(cfg)  # binary relation of keypoints and limbs
        real_matrix_limb = torch.FloatTensor(real_matrix_limb)
        self.real_matrix_limb = nn.Parameter(data=real_matrix_limb, requires_grad=False)

        kpt_transformer = []
        kpt_transformer.append(nn.Linear(weight_size, out_weight_size))
        kpt_transformer.append(nn.LeakyReLU(0.02))
        kpt_transformer.append(nn.Linear(out_weight_size, out_weight_size))
        self.kpt_transformer = nn.Sequential(*kpt_transformer)
        self.initialize_module_params()

    def _get_connection_rules(self):
        KEYPOINT_CONNECTION_RULES = [
            # face
            ("head", "neck", (255, 195, 77)),
            ("right_shoulder", "neck", (102, 0, 204)),
            ("left_shoulder", "neck", (255, 128, 0)),
            # upper-body
            ("left_shoulder", "left_elbow", (153, 255, 204)),
            ("right_shoulder", "right_elbow", (128, 229, 255)),
            ("left_elbow", "left_wrist", (153, 255, 153)),
            ("right_elbow", "right_wrist", (102, 255, 224)),
            # lower-body
            ("left_hip", "right_hip", (255, 102, 0)),
            ("left_hip", "left_knee", (255, 255, 77)),
            ("right_hip", "right_knee", (153, 255, 204)),
            ("left_knee", "left_ankle", (191, 255, 128)),
            ("right_knee", "right_ankle", (255, 195, 77)),
            # upper-lower-body
            ("left_shoulder", "left_hip", (255, 195, 77)),
            ("right_shoulder", "right_hip", (255, 195, 77)),
        ]
        return KEYPOINT_CONNECTION_RULES

    def _get_keypoints_name(self):
        COCO_PERSON_KEYPOINT_NAMES = (
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'head', 'neck')
        return COCO_PERSON_KEYPOINT_NAMES


    def generate_commonsense_knowledge_graph(self, cfg):

        num_joints = cfg.MODEL.NUM_JOINTS
        num_limbs = cfg.MODEL.NUM_LIMBS
        COCO_PERSON_KEYPOINT_NAMES = self._get_keypoints_name()
        KEYPOINT_CONNECTION_RULES = self._get_connection_rules()
        name_to_idx = {}
        for i, name in enumerate(COCO_PERSON_KEYPOINT_NAMES):
            name_to_idx[name] = i
        # connection_rules = []
        matrix_limb = np.zeros((num_limbs, num_joints))
        for i, limb in enumerate(KEYPOINT_CONNECTION_RULES):
            matrix_limb[i][name_to_idx[limb[0]]] = 1
            matrix_limb[i][name_to_idx[limb[1]]] = 1

        return matrix_limb


    def generate_knowledge_graph(self):

        return self.matrix_limb*self.real_matrix_limb


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
        weights = weights.reshape((n_out, n_in*w*h))  # 17* S
        matrix_relation = self.generate_knowledge_graph()  # 18*17
        kt_weights = torch.matmul(matrix_relation, weights)  # 18*S
        kt_weights = self.kpt_transformer(kt_weights)
        kt_weights = kt_weights.reshape((kt_weights.size(0), n_in, h, w))  # 18*C*1*1
        return kt_weights

@POSE_RSG_NET_REGISTRY.register()
class RelationPoseGraphNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        super(RelationPoseGraphNet, self).__init__()

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
            self.stage4_cfg, num_channels, multi_scale_output=False)

        # the multi-keypoints parameters
        self.multi_final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        # the keypoints parameters
        num_for_channels = cfg['MODEL']['EXTRA']['OUTPUT_CONVS'][0]  # 32
        self.num_joints = cfg.MODEL.NUM_JOINTS  # 17
        # visual information
        self.vis_conv = nn.Sequential(
            *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # type information
        channels_type_vector = cfg.MODEL.NUM_TYPE_VECTOR  # 600
        type_vec = open('/home/daiyan/code/deep-high-resolution-net/data/crowd_pose/cp_kpt_word_embs.pkl', 'rb')
        type_features = pickle.load(type_vec)  # 17*600
        type_features = torch.from_numpy(type_features).float()
        self.type_features = nn.Parameter(data=type_features, requires_grad=True)
        self.type_fc = nn.Sequential(
            *[nn.Linear(channels_type_vector, channels_type_vector, bias=False),
              nn.BatchNorm1d(channels_type_vector, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        self.type_conv = nn.Sequential(
            *[nn.Conv2d(channels_type_vector, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # location information
        self.loc_heatmap_size = np.array(cfg.MODEL.IMAGE_SIZE) / 4
        self.loc_features = self.build_geometry_embeddings((int(self.loc_heatmap_size[0]), int(self.loc_heatmap_size[1])))
        self.loc_conv = nn.Sequential(
            *[nn.Conv2d(4, num_for_channels, 1, bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # contact features
        self.contact_conv = nn.Sequential(
            *[nn.Conv2d(num_for_channels * 3, num_for_channels * 3, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels * 3, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )

        self.predict_contact_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels * 3, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )

        self.kpt_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # upsampling operation
        deconv_kernel, padding, output_padding = \
            self._get_deconv_cfg(cfg.MODEL.FINAL_DECONV_KERNEL_SIZE)
        self.predict_convtranspose = nn.Sequential(
            *[nn.ConvTranspose2d(
                in_channels=num_for_channels,
                out_channels=num_for_channels,
                kernel_size=deconv_kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False),
            nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)]
        )
        self.predict_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        # the limbs parameters
        kpt_weight_size = self.final_layer.weight.size(1) * self.final_layer.weight.size(
            2) * self.final_layer.weight.size(3)
        self.kt_machine = KTMachine(cfg, kpt_weight_size)
        self.limbs_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']


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
        loc_map = torch.FloatTensor(loc_map[np.newaxis])
        loc_map = nn.Parameter(data=loc_map, requires_grad=False)
        return loc_map

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

    def _get_deconv_cfg(self, deconv_kernel):
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

        # multi-keypoints branch
        vis_features = y_list[0]  # tensor ([1,32,64,48]), B*C*H*W
        multi_kpt_scores = self.multi_final_layer(
            vis_features)  # y_list[0](1,32,64,48)  x(1,17,64,48), multi-joints scores loss


        # visual information
        vis_features = self.vis_conv(vis_features)
        # type information
        predict_size = multi_kpt_scores.size()
        tmp_predict_scores = torch.reshape(multi_kpt_scores, (predict_size[0], predict_size[1],
                                                              predict_size[2] * predict_size[3]))  # B*17*S
        tmp_predict_scores = tmp_predict_scores.permute(0, 2, 1)  # B*S*17
        tmp_type_features = self.type_features.to(tmp_predict_scores.device)  # 17*T
        tmp_type_features = self.type_fc(tmp_type_features)
        type_size = tmp_type_features.size()
        type_features = torch.matmul(tmp_predict_scores, tmp_type_features)  # B*S*T
        type_features = type_features.permute(0, 2, 1)
        type_features = torch.reshape(type_features, (predict_size[0], type_size[1], predict_size[2], predict_size[3]))
        type_features = self.type_conv(type_features)  # B*T*H*W
        # location information
        loc_features = self.loc_conv(self.loc_features)  # B*L*H*W
        loc_features = loc_features.repeat(predict_size[0], 1, 1, 1)
        # contact these three information
        final_vis_features = torch.cat((vis_features, type_features, loc_features), dim=1)
        final_vis_features = self.contact_conv(final_vis_features)  # to smooth contact features
        final_vis_features = self.predict_contact_net(final_vis_features)  # 32*3 -> 32

        # keypoints branch
        kpt_features = self.kpt_net(final_vis_features)  # B*C*H*W
        kpt_features = self.predict_convtranspose(kpt_features)  # heatmap upsampling
        kpt_features = self.predict_net(kpt_features)  # B*C*H*W
        kpt_scores = self.final_layer(kpt_features)  #  x(1,17,64,48)

        # limbs branch
        limbs_features = self.limbs_net(final_vis_features)  # B*C*H*W
        refine_weights = self.kt_machine(self.final_layer.weight)
        limbs_scores = nn.functional.conv2d(limbs_features, weight=refine_weights, padding=0, stride=1)

        # umsampling
        multi_kpt_scores = torch.nn.functional.interpolate(multi_kpt_scores, scale_factor=2, mode='bilinear',
                                                           align_corners=True)
        limbs_scores = torch.nn.functional.interpolate(limbs_scores, scale_factor=2, mode='bilinear',
                                                       align_corners=True)
        limbs_scores = torch.sigmoid(limbs_scores)


        return multi_kpt_scores, kpt_scores, limbs_scores

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)

                for name, _ in m.named_parameters():
                    print('init:', name)
                    if name in ['bias']:
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

@POSE_RSG_NET_REGISTRY.register()
class RSGNet_initial(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        super(RSGNet_initial, self).__init__()

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
            self.stage4_cfg, num_channels, multi_scale_output=False)

        # the multi-keypoints parameters
        self.multi_final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        # the keypoints parameters
        num_for_channels = cfg['MODEL']['EXTRA']['OUTPUT_CONVS'][0]  # 32
        self.num_joints = cfg.MODEL.NUM_JOINTS  # 17
        # visual information
        self.vis_conv = nn.Sequential(
            *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # type information
        channels_type_vector = cfg.MODEL.NUM_TYPE_VECTOR  # 600
        # type_vec = open('/home/daiyan/code/deep-high-resolution-net/data/crowd_pose/cp_kpt_word_embs.pkl', 'rb')
        type_vec = open('/home/daiyan/code/deep-high-resolution-net/kpt_word_embs.pkl', 'rb')
        type_features = pickle.load(type_vec)  # 17*600
        type_features = torch.from_numpy(type_features).float()
        self.type_features = nn.Parameter(data=type_features, requires_grad=True)
        self.type_fc = nn.Sequential(
            *[nn.Linear(channels_type_vector, channels_type_vector, bias=False),
              nn.BatchNorm1d(channels_type_vector, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        self.type_conv = nn.Sequential(
            *[nn.Conv2d(channels_type_vector, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # location information
        self.loc_heatmap_size = np.array(cfg.MODEL.IMAGE_SIZE) / 4
        self.loc_features = self.build_geometry_embeddings((int(self.loc_heatmap_size[0]), int(self.loc_heatmap_size[1])))
        self.loc_conv = nn.Sequential(
            *[nn.Conv2d(4, num_for_channels, 1, bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # contact features
        self.contact_conv = nn.Sequential(
            *[nn.Conv2d(num_for_channels * 3, num_for_channels * 3, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels * 3, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )

        self.predict_contact_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels * 3, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # self.relation_head = SpatialRelationHead(num_for_channels, num_for_channels)
        self.relation_head = SpatialRelationHead(num_for_channels,num_for_channels,sub_sample=cfg.MODEL.RELATION_SUB_SAMPLE)

        self.kpt_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels * 2, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # upsampling operation
        deconv_kernel, padding, output_padding = \
            self._get_deconv_cfg(cfg.MODEL.FINAL_DECONV_KERNEL_SIZE)
        self.predict_convtranspose = nn.Sequential(
            *[nn.ConvTranspose2d(
                in_channels=num_for_channels,
                out_channels=num_for_channels,
                kernel_size=deconv_kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False),
            nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)]
        )
        self.predict_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        # the limbs parameters
        kpt_weight_size = self.final_layer.weight.size(1) * self.final_layer.weight.size(
            2) * self.final_layer.weight.size(3)
        self.kt_machine = KTMachine(cfg, kpt_weight_size)
        self.limbs_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

        self.UDP_ON = cfg.MODEL.UDP_POSE_ON

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
        loc_map = torch.FloatTensor(loc_map[np.newaxis])
        loc_map = nn.Parameter(data=loc_map, requires_grad=False)
        return loc_map

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

    def _get_deconv_cfg(self, deconv_kernel):
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

    def _forward_visual_encoder(self, x):
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

    def forward(self, x, relation_target=None):

        y_list = self._forward_visual_encoder(x)

        # multi-keypoints branch
        vis_features = y_list[0]  # tensor ([1,32,64,48]), B*C*H*W
        multi_kpt_scores = self.multi_final_layer(
            vis_features)  # y_list[0](1,32,64,48)  x(1,17,64,48), multi-joints scores loss

        # visual information
        vis_features = self.vis_conv(vis_features)
        # type information
        predict_size = multi_kpt_scores.size()
        tmp_predict_scores = torch.reshape(multi_kpt_scores, (predict_size[0], predict_size[1],
                                                              predict_size[2] * predict_size[3]))  # B*17*S
        tmp_predict_scores = tmp_predict_scores.permute(0, 2, 1)  # B*S*17
        tmp_type_features = self.type_features.to(tmp_predict_scores.device)  # 17*T
        tmp_type_features = self.type_fc(tmp_type_features)
        type_size = tmp_type_features.size()
        type_features = torch.matmul(tmp_predict_scores, tmp_type_features)  # B*S*T
        type_features = type_features.permute(0, 2, 1)
        type_features = torch.reshape(type_features, (predict_size[0], type_size[1], predict_size[2], predict_size[3]))
        type_features = self.type_conv(type_features)  # B*T*H*W
        # location information
        loc_features = self.loc_conv(self.loc_features)  # B*L*H*W
        loc_features = loc_features.repeat(predict_size[0], 1, 1, 1)
        # contact these three information
        final_vis_features = torch.cat((vis_features, type_features, loc_features), dim=1)
        final_vis_features = self.contact_conv(final_vis_features)  # to smooth contact features
        final_vis_features = self.predict_contact_net(final_vis_features)  # 32*3 -> 32

        # relation branch
        relation_feats, relation_scores = self.relation_head(final_vis_features)

        # keypoints branch
        # B*C*H*W
        kpt_features = torch.cat((relation_feats, final_vis_features), dim=1)
        kpt_features = self.kpt_net(kpt_features)
        if self.scale_factor > 1:
            kpt_features = self.predict_convtranspose(kpt_features)  # heatmap upsampling
        if self.UDP_ON:
            udp_features = self.udp_predict_net(kpt_features)
            udp_kpt_scores = self.udp_final_layer(udp_features)

        kpt_features = self.predict_net(kpt_features)  # B*C*H*W
        kpt_scores = self.final_layer(kpt_features)  # x(1,17,64,48)

        # limbs branch
        limbs_features = self.limbs_net(final_vis_features)  # B*C*H*W
        refine_weights = self.kt_machine(self.final_layer.weight)
        limbs_scores = nn.functional.conv2d(limbs_features, weight=refine_weights, padding=0, stride=1)

        # umsampling
        if self.scale_factor > 1:
            multi_kpt_scores = torch.nn.functional.interpolate(multi_kpt_scores, scale_factor=2, mode='bilinear',
                                                               align_corners=True)
            limbs_scores = torch.nn.functional.interpolate(limbs_scores, scale_factor=2, mode='bilinear',
                                                           align_corners=True)
        limbs_scores = torch.sigmoid(limbs_scores)
        if relation_target is not None:
            with torch.no_grad():
                relation_target = relation_target.to(relation_scores.device)
            diffs = (relation_target - relation_scores) ** 2
            relation_scores = torch.mean(diffs, dim=(1, 2))
        if self.UDP_ON:
            return multi_kpt_scores, [kpt_scores, udp_kpt_scores], limbs_scores, relation_scores
        return multi_kpt_scores, kpt_scores, limbs_scores, relation_scores

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)

                for name, _ in m.named_parameters():
                    print('init:', name)
                    if name in ['bias']:
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

@POSE_RSG_NET_REGISTRY.register()
class RSGNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        super(RSGNet, self).__init__()
        self.UDP_ON = cfg.MODEL.UDP_POSE_ON

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
            self.stage4_cfg, num_channels, multi_scale_output=False)

        # the multi-keypoints parameters
        self.multi_final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        # the keypoints parameters
        num_for_channels = cfg['MODEL']['EXTRA']['OUTPUT_CONVS'][0]  # 32
        self.num_joints = cfg.MODEL.NUM_JOINTS  # 17
        # visual information
        self.vis_conv = nn.Sequential(
            *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # type information
        channels_type_vector = cfg.MODEL.NUM_TYPE_VECTOR  # 600
        type_vec = open('/home/daiyan/code/deep-high-resolution-net/cp_kpt_word_embs.pkl', 'rb')
        # type_vec = open('/home/daiyan/code/deep-high-resolution-net/kpt_word_embs.pkl', 'rb')
        type_features = pickle.load(type_vec)  # 17*600
        type_features = torch.from_numpy(type_features).float()
        self.type_features = nn.Parameter(data=type_features, requires_grad=True)
        self.type_fc = nn.Sequential(
            *[nn.Linear(channels_type_vector, channels_type_vector, bias=False),
              nn.BatchNorm1d(channels_type_vector, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        self.type_conv = nn.Sequential(
            *[nn.Conv2d(channels_type_vector, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # location information
        self.loc_heatmap_size = np.array(cfg.MODEL.IMAGE_SIZE) / 4
        self.loc_features = self.build_geometry_embeddings((int(self.loc_heatmap_size[0]), int(self.loc_heatmap_size[1])))
        self.loc_conv = nn.Sequential(
            *[nn.Conv2d(4, num_for_channels, 1, bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # contact features
        self.contact_conv = nn.Sequential(
            *[nn.Conv2d(num_for_channels * 3, num_for_channels * 3, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels * 3, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )

        self.predict_contact_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels * 3, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # self.relation_head = SpatialRelationHead(num_for_channels, num_for_channels)
        self.relation_head = SpatialRelationHead(num_for_channels,num_for_channels,sub_sample=cfg.MODEL.RELATION_SUB_SAMPLE)

        self.kpt_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels * 2, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # upsampling operation
        self.scale_factor = cfg.MODEL.UP_SCALE
        if self.scale_factor > 1:
            deconv_kernel, padding, output_padding = \
                self._get_deconv_cfg(cfg.MODEL.FINAL_DECONV_KERNEL_SIZE)
            self.predict_convtranspose = nn.Sequential(
                *[nn.ConvTranspose2d(
                    in_channels=num_for_channels,
                    out_channels=num_for_channels,
                    kernel_size=deconv_kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False),
                nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)]
            )
        self.predict_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
        if self.UDP_ON:
            self.udp_predict_net = nn.Sequential(
                *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
                            bias=False),
                  nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
                  nn.ReLU(inplace=True)]
            )
            self.udp_final_layer = nn.Conv2d(
                in_channels=pre_stage_channels[0],
                out_channels=cfg.MODEL.NUM_JOINTS * 3,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
            )

        # the limbs parameters
        kpt_weight_size = self.final_layer.weight.size(1) * self.final_layer.weight.size(
            2) * self.final_layer.weight.size(3)
        self.kt_machine = KTMachine(cfg, kpt_weight_size)
        self.limbs_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

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
        loc_map = torch.FloatTensor(loc_map[np.newaxis])
        loc_map = nn.Parameter(data=loc_map, requires_grad=False)
        return loc_map

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

    def _get_deconv_cfg(self, deconv_kernel):
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

    def _forward_visual_encoder(self, x):
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

    def forward(self, x, relation_target=None):

        y_list = self._forward_visual_encoder(x)

        # multi-keypoints branch
        vis_features = y_list[0]  # tensor ([1,32,64,48]), B*C*H*W
        multi_kpt_scores = self.multi_final_layer(
            vis_features)  # y_list[0](1,32,64,48)  x(1,17,64,48), multi-joints scores loss

        # visual information
        vis_features = self.vis_conv(vis_features)
        # type information
        predict_size = multi_kpt_scores.size()
        tmp_predict_scores = torch.reshape(multi_kpt_scores, (predict_size[0], predict_size[1],
                                                              predict_size[2] * predict_size[3]))  # B*17*S
        tmp_predict_scores = tmp_predict_scores.permute(0, 2, 1)  # B*S*17
        tmp_type_features = self.type_features.to(tmp_predict_scores.device)  # 17*T
        tmp_type_features = self.type_fc(tmp_type_features)
        type_size = tmp_type_features.size()
        type_features = torch.matmul(tmp_predict_scores, tmp_type_features)  # B*S*T
        type_features = type_features.permute(0, 2, 1)
        type_features = torch.reshape(type_features, (predict_size[0], type_size[1], predict_size[2], predict_size[3]))
        type_features = self.type_conv(type_features)  # B*T*H*W
        # location information
        loc_features = self.loc_conv(self.loc_features)  # B*L*H*W
        loc_features = loc_features.repeat(predict_size[0], 1, 1, 1)
        # contact these three information
        final_vis_features = torch.cat((vis_features, type_features, loc_features), dim=1)
        final_vis_features = self.contact_conv(final_vis_features)  # to smooth contact features
        final_vis_features = self.predict_contact_net(final_vis_features)  # 32*3 -> 32

        # relation branch
        relation_feats, relation_scores = self.relation_head(final_vis_features)

        # keypoints branch
        # B*C*H*W
        kpt_features = torch.cat((relation_feats, final_vis_features), dim=1)
        kpt_features = self.kpt_net(kpt_features)
        if self.scale_factor>1:
            kpt_features = self.predict_convtranspose(kpt_features)  # heatmap upsampling
        if self.UDP_ON:
            udp_features = self.udp_predict_net(kpt_features)
            udp_kpt_scores = self.udp_final_layer(udp_features)

        kpt_features = self.predict_net(kpt_features)  # B*C*H*W
        kpt_scores = self.final_layer(kpt_features)  #  x(1,17,64,48)

        # limbs branch
        limbs_features = self.limbs_net(final_vis_features)  # B*C*H*W
        refine_weights = self.kt_machine(self.final_layer.weight)
        limbs_scores = nn.functional.conv2d(limbs_features, weight=refine_weights, padding=0, stride=1)

        # umsampling
        if self.scale_factor>1:
            multi_kpt_scores = torch.nn.functional.interpolate(multi_kpt_scores, scale_factor=2, mode='bilinear',
                                                               align_corners=True)
            limbs_scores = torch.nn.functional.interpolate(limbs_scores, scale_factor=2, mode='bilinear',
                                                           align_corners=True)
        limbs_scores = torch.sigmoid(limbs_scores)
        if relation_target is not None:
            with torch.no_grad():
                relation_target = relation_target.to(relation_scores.device)
            diffs = (relation_target - relation_scores) ** 2
            relation_scores = torch.mean(diffs, dim=(1, 2))
        if self.UDP_ON:
            return multi_kpt_scores, [kpt_scores, udp_kpt_scores], limbs_scores, relation_scores
        return multi_kpt_scores, kpt_scores, limbs_scores, relation_scores

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)

                for name, _ in m.named_parameters():
                    print('init:', name)
                    if name in ['bias']:
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

@POSE_RSG_NET_REGISTRY.register()
class TRPNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        super(TRPNet, self).__init__()
        self.UDP_ON = cfg.MODEL.UDP_POSE_ON

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
            self.stage4_cfg, num_channels, multi_scale_output=False)

        # the multi-keypoints parameters
        self.multi_final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        # the keypoints parameters
        num_for_channels = cfg['MODEL']['EXTRA']['OUTPUT_CONVS'][0]  # 32
        self.num_joints = cfg.MODEL.NUM_JOINTS  # 17
        # visual information
        self.vis_conv = nn.Sequential(
            *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # type information
        channels_type_vector = cfg.MODEL.NUM_TYPE_VECTOR  # 600
        type_vec = open('/home/daiyan/code/deep-high-resolution-net/cp_kpt_word_embs.pkl', 'rb')
        # type_vec = open('/home/daiyan/code/deep-high-resolution-net/kpt_word_embs.pkl', 'rb')
        type_features = pickle.load(type_vec)  # 17*600
        type_features = torch.from_numpy(type_features).float()
        self.type_features = nn.Parameter(data=type_features, requires_grad=True)
        self.type_fc = nn.Sequential(
            *[nn.Linear(channels_type_vector, channels_type_vector, bias=False),
              nn.BatchNorm1d(channels_type_vector, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        self.type_conv = nn.Sequential(
            *[nn.Conv2d(channels_type_vector, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # location information
        self.loc_heatmap_size = np.array(cfg.MODEL.IMAGE_SIZE) / 4
        self.loc_features = self.build_geometry_embeddings((int(self.loc_heatmap_size[0]), int(self.loc_heatmap_size[1])))
        self.loc_conv = nn.Sequential(
            *[nn.Conv2d(4, num_for_channels, 1, bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # contact features
        self.contact_conv = nn.Sequential(
            *[nn.Conv2d(num_for_channels * 3, num_for_channels * 3, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels * 3, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )

        self.predict_contact_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels * 3, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        self.relation_head = SpatialRelationHead(num_for_channels, num_for_channels)
        # self.relation_head = SpatialRelationHead(num_for_channels, num_for_channels, sub_sample=cfg.MODEL.RELATION_SUB_SAMPLE)

        self.kpt_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels * 2, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        # upsampling operation
        self.scale_factor = cfg.MODEL.UP_SCALE
        if self.scale_factor > 1:
            deconv_kernel, padding, output_padding = \
                self._get_deconv_cfg(cfg.MODEL.FINAL_DECONV_KERNEL_SIZE)
            self.predict_convtranspose = nn.Sequential(
                *[nn.ConvTranspose2d(
                    in_channels=num_for_channels,
                    out_channels=num_for_channels,
                    kernel_size=deconv_kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False),
                nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)]
            )
        self.predict_net = nn.Sequential(
            *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
                        bias=False),
              nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)]
        )
        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
        if self.UDP_ON:
            self.udp_predict_net = nn.Sequential(
                *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
                            bias=False),
                  nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
                  nn.ReLU(inplace=True)]
            )
            self.udp_final_layer = nn.Conv2d(
                in_channels=pre_stage_channels[0],
                out_channels=cfg.MODEL.NUM_JOINTS * 3,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
            )

        # the limbs parameters
        # kpt_weight_size = self.final_layer.weight.size(1) * self.final_layer.weight.size(
        #     2) * self.final_layer.weight.size(3)
        # self.kt_machine = KTMachine(cfg, kpt_weight_size)
        # self.limbs_net = nn.Sequential(
        #     *[nn.Conv2d(num_for_channels, num_for_channels, kernel_size=3, stride=1, padding=1,
        #                 bias=False),
        #       nn.BatchNorm2d(num_for_channels, momentum=BN_MOMENTUM),
        #       nn.ReLU(inplace=True)]
        # )

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

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
        loc_map = torch.FloatTensor(loc_map[np.newaxis])
        loc_map = nn.Parameter(data=loc_map, requires_grad=False)
        return loc_map

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

    def _get_deconv_cfg(self, deconv_kernel):
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

    def _forward_visual_encoder(self, x):
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

    def forward(self, x, relation_target=None):

        y_list = self._forward_visual_encoder(x)

        # multi-keypoints branch
        vis_features = y_list[0]  # tensor ([1,32,64,48]), B*C*H*W
        multi_kpt_scores = self.multi_final_layer(
            vis_features)  # y_list[0](1,32,64,48)  x(1,17,64,48), multi-joints scores loss

        # visual information
        vis_features = self.vis_conv(vis_features)

        # type information
        predict_size = multi_kpt_scores.size()
        tmp_predict_scores = torch.reshape(multi_kpt_scores, (predict_size[0], predict_size[1],
                                                              predict_size[2] * predict_size[3]))  # B*17*S
        tmp_predict_scores = tmp_predict_scores.permute(0, 2, 1)  # B*S*17
        tmp_type_features = self.type_features.to(tmp_predict_scores.device)  # 17*T
        tmp_type_features = self.type_fc(tmp_type_features)
        type_size = tmp_type_features.size()
        type_features = torch.matmul(tmp_predict_scores, tmp_type_features)  # B*S*T
        type_features = type_features.permute(0, 2, 1)
        type_features = torch.reshape(type_features, (predict_size[0], type_size[1], predict_size[2], predict_size[3]))
        type_features = self.type_conv(type_features)  # B*T*H*W
        # location information
        loc_features = self.loc_conv(self.loc_features)  # B*L*H*W
        loc_features = loc_features.repeat(predict_size[0], 1, 1, 1)
        # contact these three information
        final_vis_features = torch.cat((vis_features, type_features, loc_features), dim=1)
        final_vis_features = self.contact_conv(final_vis_features)  # to smooth contact features
        final_vis_features = self.predict_contact_net(final_vis_features)  # 32*3 -> 32 (64,48)

        # relation branch
        relation_feats, relation_scores = self.relation_head(final_vis_features)

        # keypoints branch
        # B*C*H*W
        kpt_features = torch.cat((relation_feats, final_vis_features), dim=1)
        kpt_features = self.kpt_net(kpt_features)
        if self.scale_factor > 1:
            kpt_features = self.predict_convtranspose(kpt_features)  # heatmap upsampling
        # if self.UDP_ON:
        #     udp_features = self.udp_predict_net(kpt_features)
        #     udp_kpt_scores = self.udp_final_layer(udp_features)

        kpt_features = self.predict_net(kpt_features)  # B*C*H*W
        kpt_scores = self.final_layer(kpt_features)  #  x(1,17,64,48)

        # limbs branch
        # limbs_features = self.limbs_net(final_vis_features)  # B*C*H*W
        # refine_weights = self.kt_machine(self.final_layer.weight)
        # limbs_scores = nn.functional.conv2d(limbs_features, weight=refine_weights, padding=0, stride=1)

        # umsampling
        if self.scale_factor > 1:
            multi_kpt_scores = torch.nn.functional.interpolate(multi_kpt_scores, scale_factor=2, mode='bilinear',
                                                               align_corners=True)
            # b, h, w = relation_scores.size()
            # relation_scores = relation_scores.reshape(b, 1, h, w)
            # relation_scores = torch.nn.functional.interpolate(relation_scores, scale_factor=4, mode='bilinear',
            #                                                    align_corners=True)

        #     limbs_scores = torch.nn.functional.interpolate(limbs_scores, scale_factor=2, mode='bilinear',
        #                                                    align_corners=True)
        # limbs_scores = torch.sigmoid(limbs_scores)
        if relation_target is not None:
            with torch.no_grad():
                relation_target = relation_target.to(relation_scores.device)
            diffs = (relation_target - relation_scores) ** 2
            relation_scores = torch.mean(diffs, dim=(1, 2))
        # if self.UDP_ON:
        #     return multi_kpt_scores, [kpt_scores, udp_kpt_scores], limbs_scores, relation_scores
        return multi_kpt_scores, kpt_scores, relation_scores
        # return multi_kpt_scores, kpt_scores, limbs_scores, relation_scores


    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)

                for name, _ in m.named_parameters():
                    print('init:', name)
                    if name in ['bias']:
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


def get_pose_net(cfg, is_train, **kwargs):
    # POSE_NAME = cfg.MODEL.POSE_NAME
    # logger.info('=> construct pose model {}'.format(POSE_NAME))
    # model = POSE_RSG_NET_REGISTRY.get(POSE_NAME)(cfg, **kwargs)
    model = RSGNet(cfg, **kwargs)
    model = TRPNet(cfg, **kwargs)
    # model = TRPNetNoRelation(cfg, **kwargs)
    # model = TargetAwareRelationParserNet(cfg, **kwargs)


    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
