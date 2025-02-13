import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm
from modules import l2proj
from collections import OrderedDict
from utils.utils_logger import timer

# time_total = OrderedDict()
# time_total['time'] = []

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, bias=True)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, weightnorm=None, shortcut=True):
        super(BasicBlock, self).__init__()
        self.shortcut = shortcut
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.relu1 = nn.PReLU(num_parameters=planes,init=0.1)
        self.relu2 = nn.PReLU(num_parameters=planes, init=0.1)
        self.conv2 = conv3x3(inplanes, planes, stride)
        if weightnorm:
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)


    def forward(self, x):
        # t = timer()
        # t.tic()
        out = self.relu1(x)
        out = F.pad(out,(1,1,1,1),'reflect')
        out = self.conv1(out)
        out = out[:,:, :x.shape[2], :x.shape[3]]
        out = self.relu2(out)
        out = F.pad(out,(1,1,1,1),'reflect')
        out = self.conv2(out)
        out = out[:,:, :x.shape[2], :x.shape[3]]
        if self.shortcut:
            out = x + out
        # end_time = t.toc()
        # time_total['time'].append(end_time)
        # print("Model_Basic_Block_per_iter:", end_time)
        return out


class ResDNet(nn.Module):

    def __init__(self, depth, color=True, weightnorm=True):
        self.inplanes = 64
        super(ResDNet, self).__init__()
        if color:
            in_channels = 3
        else:
            in_channels = 1

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=0,
                               bias=True)
        if weightnorm:
            self.conv1 = weight_norm(self.conv1)

        # inntermediate layer has D-2 depth
        # t = timer()
        # t.tic()
        self.layer1 = self._make_layer(BasicBlock, 64, depth)
        self.conv_out = nn.ConvTranspose2d(64, in_channels, kernel_size=5, stride=1, padding=2,
                                  bias=True)
        if weightnorm:
            self.conv_out = weight_norm(self.conv_out)

        """
        self.l2proj = l2proj.L2Proj()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weights = np.sqrt(2/(9.*64))*np.random.standard_normal(m.weight.data.shape)
                #weights = np.random.normal(size=m.weight.data.shape,
                #                           scale=np.sqrt(1. / m.weight.data.shape[1]))
                m.weight.data = torch.Tensor(weights)
                if m.bias is not None:
                    m.bias.data.zero_()
        # end_time = t.toc()
        # time_total['time'].append(end_time)
        # print("Model_intermediate_layer_Depth_per_iter:", end_time)
        self.zeromean()
        """

    def _make_layer(self, block, planes, blocks, stride=1):
        # t = timer()
        # t.tic()
        layers = []
        layers.append(block(self.inplanes, planes, stride, weightnorm=True, shortcut=False))

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, weightnorm=True, shortcut=True))
        # end_time = t.toc()
        # time_total['time'].append(end_time)
        # print("Model_make_layer:", end_time)
        return nn.Sequential(*layers)


    def zeromean(self):
        # t = timer()
        # t.tic()
        # Function zeromean subtracts the mean E(f) from filters f
        # in order to create zero mean filters
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data - torch.mean(m.weight.data)
        # end_time = t.toc()
        # time_total['time'].append(end_time)
        # print("Model_zeromean:", end_time)


    def forward(self, x, stdn, alpha):
        # t = timer()
        # t.tic()
        #self.zeromean()
        out = F.pad(x,(2,2,2,2),'reflect')
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.conv_out(out)
        #out = self.l2proj(out, stdn, alpha)
        # end_time = t.toc()
        # time_total['time'].append(end_time)
        # print("Model_ResDet_forward_per_iter:", end_time)
        # if len(time_total['time']) / 1053 == 1:
        #     avg_time = sum(time_total['time']) / len(time_total['time'])
        #     print("avg_time_model_ResDNet_epoch_1:", avg_time)
        return out