import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.util import imresize
from scipy.io import loadmat
from torch.autograd import Variable
from data import util as ut

def b_Bicubic(variable, scale):
    B, C, H, W = variable.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_v = variable.view((B, C, H, W))
    re_tensor = ut.imresize(tensor_v, 1 / scale)
    return re_tensor

class BatchBlurk(nn.Module):
    def __init__(self, l,kernel,scale):
        super(BatchBlurk, self).__init__()
        #self.kernel = kernel
        B, _, H, W = kernel.size()
        C = 3
        self.l = l

        kernel_var = (
            kernel.contiguous()
            .view((B, 1, self.l, self.l))
            .repeat(1, C, 1, 1)
            .view((B * C, 1, self.l, self.l))
        )

        self.register_buffer("kernel_var", kernel_var)
        self.scale = scale
#        self.kernel = kernel
        if l % 2 == 1:
            self.pad =(l // 2, l // 2, l // 2, l // 2)
        else:
            self.pad = (l // 2, l // 2 - 1, l // 2, l // 2 - 1)
        # self.pad = nn.ZeroPad2d(l // 2)



        #
        # if len(kernel.size()) == 2:
        #     input_CBHW = pad.view((C * B, 1, H_p, W_p))
        #     kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
        #     return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        # else:


    def forward(self, input):
        B, C, H, W = input.size()
        pad = F.pad(input, self.pad, mode='reflect')
        H_p, W_p = pad.size()[-2:]
        input_CBHW = pad.view((1, C * B, H_p, W_p))
        x = F.conv2d(input_CBHW, self.kernel_var, groups=B * C).view((B, C, H, W))
        x = b_Bicubic(x,  self.scale)
        return x



