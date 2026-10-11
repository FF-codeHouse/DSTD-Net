#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.nn as nn

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from .MMIT import MMIT
from .stogs import STOGSSparseModule


class TensorConv(nn.Module):
    """A lightweight spatio-temporal block for B x C x T x H x W tensors."""
    def __init__(self, channels, act=True):
        super().__init__()
        layers = [
            nn.Conv3d(
                channels, channels, kernel_size=3, padding=1,
                groups=channels, bias=False
            ),
            nn.BatchNorm3d(channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BFEM(nn.Module):
    """MMIT background extraction on the complete frame tensor."""
    def __init__(self, channels, num_blocks=1, residual_scale=0.2):
        super().__init__()
        self.block = MMIT(
            channels=channels,
            num_blocks=num_blocks,
            residual_scale=residual_scale,
        )

    def forward(self, x):
        return self.block(x)


class DecompositionModule(nn.Module):
    def __init__(
        self,
        layers=3,
        channel=64,
        stage_index=0,
        sparse_group_size=(5, 3, 3),
    ):
        super(DecompositionModule, self).__init__()
        self.lowrank = Lowrank_Module(embed_dims=channel)
        self.sparse = STOGSSparseModule(
            channels=channel,
            stage_index=stage_index,
            group_size=sparse_group_size,
        )
        self.merge = Merge_Module(channel=channel, layers=layers)

    def forward(self, D, T):

        B = self.lowrank(D, T)
        T = self.sparse(D, B, T)
        D = self.merge(B, T)
        return D, T,B

class Lowrank_Module(nn.Module):
    def __init__(self, embed_dims=64, num_blocks=1, residual_scale=0.2):
        super().__init__()
        self.f = BFEM(
            channels=embed_dims,
            num_blocks=num_blocks,
            residual_scale=residual_scale,
        )

    def forward(self, D, T):
        x = D - T
        B = self.f(x)
        return B

class Merge_Module(nn.Module):
    def __init__(self, channel=64, layers=3):
        super(Merge_Module, self).__init__()
        convs = [TensorConv(channel)]
        for _ in range(layers):
            convs.append(TensorConv(channel))
        convs.append(nn.Conv3d(channel, channel, kernel_size=1, bias=False))
        self.mapping = nn.Sequential(*convs)

    def forward(self, B, T):
        x = B + T
        D = self.mapping(x)
        return D



class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [256], act = "silu", depthwise = False,):
        super().__init__()
        Conv            = DWConv if depthwise else BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        if torch.is_tensor(inputs):
            inputs = [inputs]
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs

class YOLOPAFPN(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_channels = [256], depthwise = False, act = "silu", num_frame=5):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        # self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")
        if num_frame < 1 or num_frame % 2 == 0:
            raise ValueError(
                "num_frame must be a positive odd number for centered ST-OGS"
            )
        self.num_frame = num_frame
        self.current_frame_index = num_frame // 2
        self.feature_channels = int(in_channels[0] * width)
        self.decomposition_channels = max(self.feature_channels // 4, 1)

        # Keep the backbone and detection head widths unchanged. Only the
        # multi-stage tensor decomposition uses one quarter of the channels.
        self.tensor_reduce = nn.Sequential(
            nn.Conv3d(
                self.feature_channels,
                self.decomposition_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm3d(self.decomposition_channels),
            nn.SiLU(inplace=True),
        )
        self.tensor_restore = nn.Sequential(
            nn.Conv3d(
                self.decomposition_channels,
                self.feature_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm3d(self.feature_channels),
            nn.SiLU(inplace=True),
        )

        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )



        self.decos = nn.ModuleList()
        self.stage_num = 3
        for stage_index in range(self.stage_num):
            self.decos.append(
                DecompositionModule(
                    3,
                    self.decomposition_channels,
                    stage_index=stage_index,
                    sparse_group_size=(self.num_frame, 3, 3),
                )
            )
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def forward(self, input, return_aux=False):
        if input.ndim != 5:
            raise ValueError(
                "YOLOPAFPN expects B x C x T x H x W input, "
                f"got {tuple(input.shape)}"
            )
        if input.shape[2] != self.num_frame:
            raise ValueError(
                f"expected {self.num_frame} centered frames, "
                f"got {input.shape[2]}"
            )
        feat = []
        for i in range(self.num_frame):
            feat.append(self.backbone(input[:, :, i, :, :]))

        # The dataloader orders a centered clip from oldest to newest:
        # t-2, t-1, t, t+1, t+2. Keep the complete temporal tensor through
        # every decomposition stage.
        D = torch.stack(feat, dim=2)
        D = self.tensor_reduce(D)
        T = torch.zeros_like(D)

        # Preserve the original alternating iteration exactly:
        # B <- Lowrank(D, T), T <- Sparse(D, B, T), D <- Merge(B, T).
        sparse_states = []
        for i in range(self.stage_num):
            D, T, B = self.decos[i](D, T)
            sparse_states.append(T)

        # The sparse tensor T represents the target component. Restore its
        # channels and select the middle/current frame for the detection head.
        T = self.tensor_restore(T)
        current_feature = T[:, :, self.current_frame_index, :, :]
        P3_out = self.C3_p3(current_feature)
        if return_aux:
            return [P3_out], sparse_states
        return [P3_out]

class YoloBody(nn.Module):
    def __init__(self, num_classes, phi='s', num_frame=5):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        depth, width    = depth_dict[phi], width_dict[phi]
        depthwise       = True if phi == 'nano' else False 

        self.backbone   = YOLOPAFPN(depth, width, depthwise=depthwise, num_frame=num_frame)
        self.head       = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x, return_aux=False):
        backbone_output = self.backbone.forward(x, return_aux=return_aux)
        if return_aux:
            fpn_outs, sparse_states = backbone_output
        else:
            fpn_outs = backbone_output
        outputs     = self.head.forward(fpn_outs)
        if return_aux:
            return outputs, sparse_states
        return outputs
