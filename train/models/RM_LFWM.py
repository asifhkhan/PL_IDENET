import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.wmad_estimator import Wmad_estimator
from modules.filters import FilterHigh
from .kernel_embed import M_kemb
import math


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class SRResDNet(nn.Module):
    def __init__(self,netT,deblur,scale):
        super(SRResDNet, self).__init__()
        self.ch_in = 3
        self.num_out_ch = 3
        self.num_feat = 64
        self.netT = netT
        self.deblur = deblur
        self.upscale_factor = scale
        self.noise_estimator = Wmad_estimator()
        self.k_embd = M_kemb()
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(self.ch_in,  self.num_feat, 3, 1, 1),
                                                  nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(scale, self.num_feat )
        self.conv_last = nn.Conv2d( self.num_feat,  self.num_out_ch , 3, 1, 1)


    def forward(self, x, kernel):


        # reconstruction block
        x = self.conv_before_upsample(x)
        x_up = self.conv_last(self.upsample(x))
        sigma = self.noise_estimator(x_up)
        sigma *= 255.
        output= self.deblur(x_up, kernel, sigma)

        k_embd = self.k_embd(kernel)
        output = self.netT(output, k_embd)

        return output

