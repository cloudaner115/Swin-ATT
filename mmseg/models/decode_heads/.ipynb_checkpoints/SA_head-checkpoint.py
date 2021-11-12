# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead




class Attention_block(nn.ModuleList):
    def __init__(self, F_g, F_l, F_int,norm_cfg):
        super(Attention_block, self).__init__()
        self.W_g = ConvModule(in_channels=F_g,
                    out_channels=F_int,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=None)
        self.W_x = ConvModule(in_channels=F_l,
                    out_channels=F_int,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=None)

        self.psi = ConvModule(in_channels=F_int,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='Sigmoid'))
        
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        gx = g1 + x1

        psi = self.relu(gx)
        psi = self.psi(psi)

        return x * psi

def double_conv(in_channels, out_channels,norm_cfg):
    block = nn.Sequential(
                ConvModule(in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3,3),
                stride=1,
                padding=1,
                norm_cfg=norm_cfg),
                      
                ConvModule(in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3,3),
                stride=1,
                padding=1,
                norm_cfg=norm_cfg))
    return block

class expansive_block(nn.ModuleList):
    def __init__(self, in_channels, out_channels,norm_cfg):
        super(expansive_block, self).__init__()

        self.block = ConvModule(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3,3),
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.block(x)
        return out

@HEADS.register_module()
class SAHead(BaseDecodeHead):


    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(
            input_transform='multiple_select', **kwargs)
        
        norm_cfg = self.norm_cfg
        num_inputs = len(self.in_channels)
        in_channels = self.in_channels
        self.conv_decode3 = expansive_block(in_channels[3], in_channels[2],norm_cfg)
        self.att3 = Attention_block(in_channels[2], in_channels[2], F_int=in_channels[1],norm_cfg=norm_cfg)
        self.double_conv3 = double_conv(in_channels[3], in_channels[2],norm_cfg)

        self.conv_decode2 = expansive_block(in_channels[2], in_channels[1],norm_cfg)
        self.att2 = Attention_block(F_g=in_channels[1], F_l=in_channels[1], F_int=in_channels[0],norm_cfg=norm_cfg)
        self.double_conv2 = double_conv(in_channels[2], in_channels[1],norm_cfg)

        self.conv_decode1 = expansive_block(in_channels[1], in_channels[0],norm_cfg)
        self.att1 = Attention_block(F_g=in_channels[0], F_l=in_channels[0], F_int=64,norm_cfg=norm_cfg)
        self.double_conv1 = double_conv(in_channels[1], in_channels[0],norm_cfg)

        self.conv_decode4 = expansive_block(in_channels[0], in_channels[4],norm_cfg)
        self.att4 = Attention_block(F_g=in_channels[4], F_l=in_channels[4], F_int=64,norm_cfg=norm_cfg)
        self.double_conv4 = double_conv(in_channels[0], in_channels[4],norm_cfg)



    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        decode_block3 = self.conv_decode3(inputs[3])
        encode_block3 = self.att3(g=decode_block3, x=inputs[2])
        decode_block3 = torch.cat((encode_block3, decode_block3), dim=1)
        decode_block3 = self.double_conv3(decode_block3)
        
        decode_block2 = self.conv_decode2(decode_block3)
        encode_block2 = self.att2(g=decode_block2, x=inputs[1])
        decode_block2 = torch.cat((encode_block2, decode_block2), dim=1)
        decode_block2 = self.double_conv2(decode_block2)

        decode_block1 = self.conv_decode1(decode_block2)
        encode_block1 = self.att1(g=decode_block1, x=inputs[0])
        decode_block1 = torch.cat((encode_block1, decode_block1), dim=1)
        decode_block1 = self.double_conv1(decode_block1)
        
        decode_block4 = self.conv_decode4(decode_block1)
        encode_block4 = self.att4(g=decode_block1, x=inputs[4])
        decode_block4 = torch.cat((encode_block4, decode_block4), dim=1)
        decode_block4 = self.double_conv4(decode_block4)
            
            
        output = self.cls_seg(decode_block4) 
        
        return output
    



        