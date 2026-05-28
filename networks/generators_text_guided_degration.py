#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-08-10 15:52:29

import torch
from torch import nn
from math import ceil, sqrt

# import torch
# import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

#########################################################################

Conv2d = nn.Conv2d
##########################################################################
## Layer Norm
def to_2d(x):
    return rearrange(x, 'b c h w -> b (h w c)')

def to_3d(x):
#    return rearrange(x, 'b c h w -> b c (h w)')
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
#    return rearrange(x, 'b c (h w) -> b c h w',h=h,w=w)
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class GeneratorState(nn.Module):
    def __init__(self, latent_size=64, motion_size=64, num_feature=128):
        '''
                Input:
                latent_size: dim of latent variable z
                state_size: dim of state variable s
                num_feature: number of units of the hidden layer
        '''
        super(GeneratorState, self).__init__()
        self.motion_size = motion_size
        self.linear1 = nn.Sequential(
            nn.Linear(latent_size, num_feature, bias=True),
            nn.ReLU(True)
        )
        if motion_size > 0:
            self.linear2 = nn.Sequential(
                nn.Linear(motion_size, num_feature, bias=True),
                nn.ReLU(True)
            )
        self.linear3 = nn.Sequential(
            nn.Linear(num_feature, num_feature, bias=True),
            nn.Tanh()
        )

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=sqrt(2))
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z, motion=None):
        x1 = self.linear1(z)
        if self.motion_size > 0 or motion is not None:
            x2 = self.linear2(motion)
            state_next = self.linear3(x1 + x2)
        else:
            state_next = self.linear3(x1)

        return state_next


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

#        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) #* self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

#        self.weight = nn.Parameter(torch.ones(normalized_shape))
#        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) #* self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class ch_shuffle_high_text(nn.Module):
    def __init__(self, ch_dim,num_heads,LayerNorm_type,ffn_expansion_factor, bias,lin_ch=512):
        super(ch_shuffle_high_text, self).__init__()
        self.dim = ch_dim
        self.linear_layer1 = nn.Linear(lin_ch, lin_ch)
        self.linear_layer3 = nn.Linear(lin_ch, 2 * ch_dim)
        # -----------------------------------------------------------------------
        self.conv1x1 = nn.Conv2d(ch_dim, 2 * ch_dim, kernel_size=1, stride=1, padding=0)  #
        self.conv_out = nn.Conv2d(2 * ch_dim, ch_dim, kernel_size=1, stride=1, padding=0)  #
        self.norm1 = LayerNorm(ch_dim, LayerNorm_type)
        self.norm2 = LayerNorm(ch_dim, LayerNorm_type)
        self.norm3 = LayerNorm(ch_dim, LayerNorm_type)
        # self.select_attn = Topm_CrossAttention_Restormer(ch_dim, num_heads, bias=False)
        # self.ffn = FeedForward(ch_dim, ffn_expansion_factor, bias)
    def forward(self, img_featur, text_code):
        b, c, _, _ = img_featur.shape
        img_feature2 = img_featur
        text_code = self.linear_layer1(text_code)
        text_code = self.linear_layer3(text_code)
        # soft_values, soft_indices = torch.topk(text_code, k=2 * self.dim)
        soft_indices = torch.randperm(2 * self.dim).reshape(1, 2 * self.dim)
        img_featur = self.conv1x1(img_featur)
        shuffled_img = img_featur[torch.arange(b).unsqueeze(1), soft_indices, :, :]  # shuffle
        output = self.conv_out(shuffled_img)
        return output, img_feature2



class GeneratorRain(nn.Module):
    def __init__(self, im_size,
                       out_channels=3,
                       filter_size=3,
                       state_size=128,
                       text_size=512,
                       up_scale=2,
                       num_feature=64,
                       dim=128,
                       heads=[1, 2, 4, 8],
                       ffn_expansion_factor=2.66,
                       bias=False,
                       LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                       ):
        '''
        Input:
            im_size: 2-dim tuple or list, [h, w]
            filter_size: integer, filter size default 5
            state_size: dim of state variable s
            up_scale: scale of the last PixelShuffle layer
            num_feature: number of feature maps of the middle convolution layers
        '''
        super(GeneratorRain, self).__init__()
        self.height, self.width = im_size
        self.height_down = ceil(self.height / up_scale)
        self.width_down = ceil(self.width / up_scale)

        # self.encoder_shuffle_channel1 = ch_shuffle_high_text(ch_dim=dim, num_heads=heads[0],
        #                                                      LayerNorm_type=LayerNorm_type,
        #                                                      ffn_expansion_factor=ffn_expansion_factor,
        #                                                      bias=bias)  # encoder level1 shuffle
        # self.encoder_shuffle_channel2 = ch_shuffle_high_text(ch_dim=int(dim * 2 ** 1), num_heads=heads[1],
        #                                                      LayerNorm_type=LayerNorm_type,
        #                                                      ffn_expansion_factor=ffn_expansion_factor,
        #                                                      bias=bias)  # encoder level2 shuffle
        self.encoder_shuffle_channel3 = ch_shuffle_high_text(ch_dim=int(dim), num_heads=heads[2],
                                                             LayerNorm_type=LayerNorm_type,
                                                             ffn_expansion_factor=ffn_expansion_factor,
                                                             bias=bias)  # encoder level3 shuffle
        self.latent_shuffle_channel = ch_shuffle_high_text(ch_dim=int(dim * 2), num_heads=heads[3],
                                                           LayerNorm_type=LayerNorm_type,
                                                           ffn_expansion_factor=ffn_expansion_factor,
                                                           bias=bias)  # latent latent shuffle

        self.linear_layer = nn.Sequential(
            nn.Linear(state_size, self.height_down*self.width_down, bias=True),
            nn.ReLU(inplace=True)
            )

        # self.linear_layer = nn.Sequential(
        #     nn.Linear(state_size + text_size, self.height_down * self.width_down, bias=True),
        #     nn.ReLU(inplace=True)
        # )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=num_feature * 2,
                      kernel_size=filter_size,
                      stride=1,
                      padding=int((filter_size - 1) / 2),
                      bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=num_feature * 2,
                      out_channels=num_feature * 4,
                      kernel_size=filter_size,
                      stride=1,
                      padding=int((filter_size - 1) / 2),
                      bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=num_feature * 4,
                      out_channels=num_feature * 2,
                      kernel_size=filter_size,
                      stride=1,
                      padding=int((filter_size - 1) / 2),
                      bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=num_feature * 4,
                      out_channels=num_feature * (up_scale ** 3),
                      kernel_size=filter_size,
                      stride=1,
                      padding=int((filter_size - 1) / 2),
                      bias=True),
            nn.ReLU(inplace=True),
        )

        self.out_conv =  nn.Sequential(
            nn.PixelShuffle(up_scale),
            nn.Conv2d(in_channels=num_feature * 2,
                      out_channels=out_channels,
                      kernel_size=filter_size,
                      stride=1,
                      padding=int((filter_size - 1) / 2),
                      bias=True),
            nn.ReLU(inplace=True)
        )

        # self.body= nn.Sequential(
        #         nn.Conv2d(in_channels=1,
        #                   out_channels=num_feature*8,
        #                   kernel_size=filter_size,
        #                   stride=1,
        #                   padding=int((filter_size-1)/2),
        #                   bias=True),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(in_channels=num_feature*8,
        #                   out_channels=num_feature*4,
        #                   kernel_size=filter_size,
        #                   stride=1,
        #                   padding=int((filter_size-1)/2),
        #                   bias=True),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(in_channels=num_feature*4,
        #                   out_channels=num_feature*2,
        #                   kernel_size=filter_size,
        #                   stride=1,
        #                   padding=int((filter_size-1)/2),
        #                   bias=True),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(in_channels=num_feature*2,
        #                   out_channels=num_feature*(up_scale**2),
        #                   kernel_size=filter_size,
        #                   stride=1,
        #                   padding=int((filter_size-1)/2),
        #                   bias=True),
        #         nn.ReLU(inplace=True),
        #         nn.PixelShuffle(up_scale),
        #         nn.Conv2d(in_channels=num_feature,
        #                   out_channels=out_channels,
        #                   kernel_size=filter_size,
        #                   stride=1,
        #                   padding=int((filter_size-1)/2),
        #                   bias=True),
        #         nn.ReLU(inplace=True)
        #         )

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, text_code):
        x = self.linear_layer(x)
        x = x.view([-1, 1, self.height_down, self.width_down])
        # rain = self.body(x)[:, :, :self.height, :self.width]
        enc_level1 = self.conv_1(x)
        enc_level2 = self.conv_2(enc_level1)
        latent, _ = self.latent_shuffle_channel(enc_level2, text_code)  # latent latent shuffle

        enc_level3 = self.conv_3(latent)
        outt1, _ = self.encoder_shuffle_channel3(enc_level1, text_code)
        enc_level3 = torch.cat([enc_level3, outt1], 1)
        enc_level4 = self.conv_4(enc_level3)

        degration = self.out_conv(enc_level4)

        return degration

    # def forward(self, x, y):
    #     x = self.linear_layer(torch.cat((x, y), dim=1))
    #     x = x.view([-1, 1, self.height_down, self.width_down])
    #     degration = self.body(x)[:, :, :self.height, :self.width]
    #
    #     return degration
