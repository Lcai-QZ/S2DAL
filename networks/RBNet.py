import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

sys.path.append('/media/zyserver/data16t/cailei/project/SAHistoFormer/networks')

from Attention import *
# from Attention import ChannelAttention
from LSTM import *

class RBNet(nn.Module):
    def __init__(self, recurrent_iter=6, channel=32):
        super(RBNet, self).__init__()

        self.channel = channel
        self.iteration = recurrent_iter
        self.Channel_Attention_1 = ChannelAttention(32)
        self.Channel_Attention_2 = ChannelAttention(32)
        self.Channel_Attention_3 = ChannelAttention(32)
        self.Channel_Attention_4 = ChannelAttention(32)
        self.Channel_Attention_5 = ChannelAttention(32)

        # self.layer0 = AttentionResBlock(32, self.channel)
        self.layer1 = AttentionResBlock(32, self.channel)
        self.layer2 = AttentionResBlock(32, self.channel)
        self.layer3 = AttentionResBlock(32, self.channel)
        self.layer4 = AttentionResBlock(32, self.channel)
        self.layer5 = AttentionResBlock(32, self.channel)

        self.conv_1 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.conv_2 = nn.Conv2d(96, 32, kernel_size=1, bias=False)
        self.conv_3 = nn.Conv2d(128, 32, kernel_size=1, bias=False)
        self.conv_4 = nn.Conv2d(160, 32, kernel_size=1, bias=False)
        self.conv_5 = nn.Conv2d(192, 32, kernel_size=1, bias=False)

        self.conv_in = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.LSTM1 = LSTM(32)
        self.LSTM2 = LSTM(32)
        self.LSTM3 = LSTM(32)
        self.LSTM4 = LSTM(32)
        self.LSTM5 = LSTM(32)

        self.conv1 = nn.Sequential(
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(160, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(192, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(224, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, input):
        batch_size, H, W = input.size(0), input.size(2), input.size(3)

        x = input
        h1, c1, h2, c2, h3, c3, h4, c4, h5, c5 = None, None, None, None, None, None, None, None, None, None

        x_list = []
        for i in range(self.iteration):
            # block_0: input_channel=6
            x = torch.cat((input, x), 1)
            x = self.conv_in(x)
            resx = x

            # the first internal and external dense connection block
            h1, c1 = self.LSTM1(x, h1, c1)
            # x = h1
            x = torch.cat((h1, resx), 1)
            resx = x
            # x = F.relu(self.layer1(x) + resx)
            # x = self.conv1(self.Channel_Attention_1(self.conv_1(x)) + h1)
            # x = self.conv1(torch.cat((self.Channel_Attention_1(self.conv_1(x)), h1), 1))
            # x = self.conv1(torch.cat((self.layer1(self.conv_1(x)), h1), 1))
            x = self.conv1(torch.cat((self.layer1(self.conv_1(x)), x), 1))

            h2, c2 = self.LSTM2(x, h2, c2)
            x = torch.cat((h2, resx), 1)
            resx = x
            # x = self.conv2(torch.cat((self.Channel_Attention_2(self.conv_2(x)), h2), 1))
            # x = self.conv2(torch.cat((self.layer2(self.conv_2(x)), h2), 1))
            x = self.conv2(torch.cat((self.layer2(self.conv_2(x)), x), 1))

            h3, c3 = self.LSTM3(x, h3, c3)
            x = torch.cat((h3, resx), 1)
            resx = x
            # x = self.conv3(torch.cat((self.layer3(self.conv_3(x)), h3), 1))
            x = self.conv3(torch.cat((self.layer3(self.conv_3(x)), x), 1))

            h4, c4 = self.LSTM4(x, h4, c4)
            x = torch.cat((h4, resx), 1)
            resx = x
            # x = self.conv4(torch.cat((self.layer4(self.conv_4(x)), h4), 1))
            x = self.conv4(torch.cat((self.layer4(self.conv_4(x)), x), 1))

            h5, c5 = self.LSTM5(x, h5, c5)
            x = torch.cat((h5, resx), 1)
            # resx = x
            # x = self.conv5(torch.cat((self.layer5(self.conv_5(x)), h5), 1))
            x = self.conv5(torch.cat((self.layer5(self.conv_5(x)), x), 1))

            x = self.conv_out(x)
            x = x + input

            x_list.append(x)

        return x
        # return x, x_list

class AttentionResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AttentionResBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.Channel_Attention_1 = ChannelAttention(32)
        self.Channel_Attention_2 = ChannelAttention(32)
        self.conv_1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=1, padding=1)
        self.LeakyReLU = nn.LeakyReLU(0.2, True)
        self.ReLU = nn.ReLU()
        self.conv_2 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        output = self.Channel_Attention_1(self.ReLU(self.conv_1(input)))
        # x = self.LeakyReLU(self.Channel_Attention_1(self.conv_1(input)))
        # output = self.LeakyReLU(self.Channel_Attention_2(self.conv_2(x)))
        return output