import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from modules.wmad_estimator import Wmad_estimator


class DEblurSRResDNet(nn.Module):
    def __init__(self,netDe, netK):
        super(DEblurSRResDNet, self).__init__()

        self.netDe = netDe
        self.netK = netK


    def forward(self, input):


        # reconstruction block


        K = self.netK(input)
        # K = None

        sr = self.netDe(input,  K.detach())
        # K = None
        # sr = None

        return sr, K
