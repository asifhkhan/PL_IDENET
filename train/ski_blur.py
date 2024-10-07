import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.util import imresize
from scipy.io import loadmat
from torch.autograd import Variable


def b_Bicubic(variable, scale):
    B, C, H, W = variable.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_v = variable.view((B, C, H, W))
    re_tensor = imresize(tensor_v, 1 / scale)
    return re_tensor

class BatchBlur(object):
    def __init__(self, l, scale):
        self.scale = scale
        self.l = l
        if l % 2 == 1:
            self.pad =(l // 2, l // 2, l // 2, l // 2)
        else:
            self.pad = (l // 2, l // 2 - 1, l // 2, l // 2 - 1)
        # self.pad = nn.ZeroPad2d(l // 2)

    def gen_ski(self, input, kernel):
        B, C, H, W = input.size()
        pad = F.pad(input, self.pad, mode='reflect')
        H_p, W_p = pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            input_CBHW = pad.view((1, C * B, H_p, W_p))
            kernel_var = (
                kernel.contiguous()
                .view((B, 1, self.l, self.l))
                .repeat(1, C, 1, 1)
                .view((B * C, 1, self.l, self.l))
            )
            hr_blured_var= F.conv2d(input_CBHW, kernel_var, groups=B * C).view((B, C, H, W))

        # Down sample

        if self.scale != 1:
            lr_blured_t = b_Bicubic(hr_blured_var, self.scale)
        else:
            lr_blured_t = hr_blured_var

        return lr_blured_t



