#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-09-23 15:45:37

import torch.nn as nn
from .sub_blocks import PixelShuffle3D, PixelUnShuffle3D, MyConv3d

class DerainNet(nn.Module):
    def __init__(self, n_channels=3, upscale_factor=2, n_features=32, n_resblocks=8):  #初始化方法定义了网络的输入通道数、上采样因子、特征图数量和残差块数量
        super(DerainNet, self).__init__()

        self.unshuffle = PixelUnShuffle3D(upscale_factor)  #添加了一个像素Unshuffle层，用于上采样输入数据

        self.head = nn.Sequential(
                    MyConv3d(in_channels=n_channels*(upscale_factor**2),
                             out_channels=n_features,
                             kernel_size=3,
                             stride=1,
                             bias=True),
                    nn.ReLU(True),
                    MyConv3d(in_channels=n_features,
                             out_channels=n_features,
                             kernel_size=3,
                             stride=1,
                             bias=True),
                     )   #代码定义了网络的头部，包含两个卷积层，用于提取特征

        # layers resblocks
        self.body = nn.ModuleList()
        for ii in range(n_resblocks):
            self.body.append(ResBlock(n_features, n_features))  #创建了一个残差块列表，每个块包含两个 MyConv3d 卷积层和 BatchNorm3d 层
            if (ii+1) == n_resblocks:
                self.body.append(nn.BatchNorm3d(n_features))
                self.body.append(nn.ReLU(inplace=True))
                self.body.append(MyConv3d(in_channels=n_features,
                            out_channels=n_channels*(upscale_factor**2),
                            kernel_size=3,
                            stride=1,
                            bias=True))
                self.body.append(PixelShuffle3D(upscale_factor))  #在残差块列表后添加了批量归一化、ReLU激活函数、另一个卷积层和像素Shuffle层

        self.tail = nn.Sequential(
                    MyConv3d(in_channels=n_channels,
                             out_channels=n_features,
                             kernel_size=3,
                             stride=1,
                             bias=False),
                    nn.ReLU(True),
                    MyConv3d(in_channels=n_features,
                             out_channels=n_features,
                             kernel_size=3,
                             stride=1,
                             bias=False),
                    nn.ReLU(True),
                    MyConv3d(in_channels=n_features,
                             out_channels=n_channels,
                             kernel_size=3,
                             stride=1,
                             bias=False)
                )  #定义了网络的尾部，包含三个卷积层，用于生成最终的去雨输出

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.orthogonal_(m.weight)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)  #初始化网络中的权重，使用正交初始化方法，并设置偏置为0

    def forward(self, images):
        x = self.unshuffle(images)
        x = self.head(x)
        for op in self.body:
            x = op(x)
        out = self.tail(images-x)
        return out   #这个前向传播方法定义了数据通过网络的流程，包括上采样、头部卷积、残差块处理和尾部卷积

class ResBlock(nn.Module):
    '''
    Res Block: x + conv(ReLU(BN(conv(ReLU(BN(x))))))
    '''
    def __init__(self, in_channels, out_channels):  #初始化方法定义了残差块的输入和输出通道数
        super(ResBlock, self).__init__()
        self.res = nn.Sequential(
                nn.BatchNorm3d(in_channels),
                nn.ReLU(True),
                MyConv3d(in_channels, out_channels,
                         kernel_size=3,
                         stride=1,
                         bias=True),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(True),
                MyConv3d(out_channels, out_channels,
                         kernel_size=3,
                         stride=1,
                         bias=True)
                )  #定义了一个残差块，包含批量归一化、ReLU激活函数和卷积层

    def forward(self, x):
        out = x + self.res(x)
        return out  #前向传播方法定义了残差块的数据流程，包括卷积操作和残差连接



