import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, normal_init, constant_init, kaiming_init

from ..registry import HEADS
from ..builder import build_loss
from ..utils import ConvModule

import torch
import numpy as np


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

@HEADS.register_module
class MaskEnhanceHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level,
                 end_level,
                 num_classes,
                 conv_cfg=None,
                 norm_cfg=None):
        super(MaskEnhanceHead, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # self.convs_all_levels = nn.ModuleList()
        ###############这个里面是正序，input是倒序
        """
        3
        
        2
        
        1
           
        0  /-----/
          /     /  
         /-----/
        
        
        
        """
        self.conv0head = nn.Sequential(ConvModule(
            self.in_channels + 2,
            self.out_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False),)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv0bot0 = nn.Sequential(ConvModule(
            self.out_channels,
            self.out_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False),)
        self.conv0bot1 = nn.Sequential(ConvModule(
            self.out_channels,
            self.out_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False),)
        #################
        self.conv1head = nn.Sequential(ConvModule(
            self.in_channels + self.out_channels,
            self.out_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False),)
        self.conv1bot0 = nn.Sequential(ConvModule(
            self.out_channels,
            self.out_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False),)
        ##############
        self.conv2head = nn.Sequential(ConvModule(
            self.in_channels + self.out_channels,
            self.out_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False),)
        #############
        # self.gcn = ContextBlock(inplanes=128, ratio=1./4.)
        self.conv3head = nn.Sequential(ConvModule(
            self.in_channels + self.out_channels,
            self.out_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False),)
        ##############
        self.conv_pred = nn.Sequential(ConvModule(
                self.out_channels,
                self.num_classes,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg),)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # for m in self.conv0head:
        #     normal_init(m.conv, std=0.01)
        # for m in self.conv0bot0:
        #     normal_init(m.conv, std=0.01)
        # for m in self.conv0bot1:
        #     normal_init(m.conv, std=0.01)
        # for m in self.conv1head:
        #     normal_init(m.conv, std=0.01)
        # for m in self.conv1bot0:
        #     normal_init(m.conv, std=0.01)
        # for m in self.conv2head:
        #     normal_init(m.conv, std=0.01)
        # for m in self.conv3head:
        #     normal_init(m.conv, std=0.01)
        # for m in self.conv_pred:
        #     normal_init(m.conv, std=0.01)

    def forward(self, inputs):
        assert len(inputs) == (self.end_level - self.start_level + 1)
        input_p = inputs[3]
        input_feat = input_p
        x_range = torch.linspace(-1, 1, input_feat.shape[-1], device=input_feat.device)
        y_range = torch.linspace(-1, 1, input_feat.shape[-2], device=input_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([input_feat.shape[0], 1, -1, -1])
        x = x.expand([input_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input_p = torch.cat([input_p, coord_feat], 1)
        ############
        conv0 = self.up(self.conv0head(input_p))
        conv0_1 = self.up(self.conv0bot0(conv0))
        conv0_2 = self.up(self.conv0bot1(conv0_1))
        feature_add_all_level = conv0_2

        input_p = inputs[2]
        input_p = torch.cat([input_p, conv0], 1)
        conv1 = self.up(self.conv1head(input_p))
        # conv1_0 = torch.cat([conv1, conv0_1], 1)
        conv1_1 = self.up(self.conv1bot0(conv1))
        feature_add_all_level += conv1_1

        input_p = inputs[1]
        input_p = torch.cat([input_p, conv1], 1)
        conv2 = self.up(self.conv2head(input_p))
        feature_add_all_level += conv2

        input_p = inputs[0]
        # conv2 = self.gcn(conv2)
        input_p = torch.cat([input_p, conv2], 1)
        feature_add_all_level += self.conv3head(input_p)


        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred

if __name__ == '__main__':
    table = []
    a = torch.randn(2, 256, 128, 112)
    b = torch.randn(2, 256, 64, 56)
    c = torch.randn(2, 256, 32, 26)
    d = torch.randn(2, 256, 16, 14)
    e = torch.randn(2, 256, 8, 7)
    table.append(a)
    table.append(b)
    table.append(c)
    table.append(d)
    table.append(e)
    model = MaskFeatHead(256, 128, 0, 3, 128,norm_cfg=dict(type='GN', num_groups=32, requires_grad=True))
    output = model(table)
    print(output.shape)