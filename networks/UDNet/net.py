#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 09:45:54 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmatis@skoltech.ru
"""
import torch as th
from torch import nn
from ...nnLayers import modules
from ...nnLayers import init

class UDNet(nn.Module):
    
    def __init__(self, kernel_size,\
                 input_channels,\
                 output_features,\
                 rbf_mixtures,\
                 rbf_precision,\
                 stages = 5,\
                 pad = 'same',\
                 padType = 'symmetric',\
                 convWeightSharing = True,\
                 scale_f = True,\
                 scale_t = True,\
                 normalizedWeights = True,\
                 zeroMeanWeights = True,\
                 rbf_start = -100,\
                 rbf_end = 100,\
                 data_min = -100,\
                 data_max = 100,\
                 data_step = 0.1,\
                 alpha = True,\
                 clb = 0,\
                 cub = 255):
        
        super(UDNet, self).__init__()
        
        rbf_centers = th.linspace(rbf_start,rbf_end,rbf_mixtures).type_as(th.Tensor())       
        self.rbf_data_lut = init.rbf_lut(rbf_centers,rbf_precision,data_min,\
                                         data_max,data_step)
        self.stages = stages        
        
        self.resRBF = nn.ModuleList([modules.ResidualRBFLayer(kernel_size,\
                    input_channels,output_features,rbf_mixtures,\
                    rbf_precision,pad,convWeightSharing,alpha,rbf_start,\
                    rbf_end,padType,scale_f,scale_t,normalizedWeights,\
                    zeroMeanWeights) for i in range(self.stages)])        
        self.bbProj = nn.Hardtanh(min_val = clb, max_val = cub)
        
    def forward(self,input,stdn,net_input = None):
        if net_input is None:
            net_input = input
            
        for m in self.resRBF:
            input = m(input,stdn,self.rbf_data_lut.type_as(input),net_input)
        
        return self.bbProj(input)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'stages = ' + str(self.stages) + ')'    

