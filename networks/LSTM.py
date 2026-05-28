import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, channel = 32):
        super(LSTM, self).__init__()
        self.channel = channel
        self.conv_i = nn.Sequential(
            nn.Conv2d(self.channel * 2, self.channel, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(self.channel * 2, self.channel, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(self.channel * 2, self.channel, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(self.channel * 2, self.channel, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input, prev_h, prev_c):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        if prev_h is None:
            prev_h = Variable(torch.zeros(batch_size, self.channel, row, col)).cuda()
            prev_c = Variable(torch.zeros(batch_size, self.channel, row, col)).cuda()

        x = torch.cat((input, prev_h), 1)

        i = self.conv_i(x)
        f = self.conv_f(x)
        g = self.conv_g(x)
        o = self.conv_o(x)

        c = f * prev_c + i * g
        h = o * torch.tanh(c)
        return h, c
