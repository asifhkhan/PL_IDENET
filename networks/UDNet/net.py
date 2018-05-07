#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 09:45:54 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmatis@skoltech.ru
"""
import torch as th
from torch import nn
from pydl.nnLayers import modules
from pydl.nnLayers import init

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

def loadModel(filePath,location = 'cpu'):
    """Loads a trained model.
    
    filePath : the path of the file containing the parameters and the architecture
               of the model.
    location : Where to load the model. (Default: cpu)
    """
    # We assume that the parameters of the model are saved in GPU format.
    if location == 'gpu' and th.cuda.is_available():
        state = th.load(filePath)
    elif location == 'cpu':
        state =  th.load(filePath,map_location=lambda storage,loc:storage)
    else:
        raise Exception("Unknown device to load the model.")
    
    # state['params'] is an Ordered dictionary with the following keys:
    #    odict_keys(['kernel_size', 'input_channels', 'output_features', \
    #   'rbf_mixtures', 'rbf_precision', 'stages', 'pad', 'padType', \
    #  'convWeightSharing', 'scale_f', 'scale_t', 'normalizedWeights', \
    #   'zeroMeanWeights', 'rbf_start', 'rbf_end', 'data_min', 'data_max', \
    #   'data_step', 'alpha', 'clb', 'cub'])
    
    model = UDNet(*state['params'].values())
    model.load_state_dict(state['model_state_dict'])

    return model

def UDNet_denoise(y,stdn):
    import os.path
    
    assert(isinstance(stdn,(float,int))),"The second argument must be an int or"\
    + " a float."
    
    stdn = th.Tensor([stdn]).type_as(y)
    
    while y.dim() < 4:
        y = y.unsqueeze(0)
    
    currentPath = os.path.dirname(os.path.realpath(__file__))
    mpath = os.path.join(currentPath,'models','UDNet_')
    
    batch,channels,H,W = y.shape
    
    if channels == 1:
        if stdn < 30:
            mpath += "LGJS5.md"
        else:
            mpath += "HGJS5.md"
    elif channels == 3:
        if stdn < 30:
            mpath += "LCJS5.md"
        else:
            mpath += "HCJS5.md"
    else: 
        raise ValueError("Input tensor must have either one or three channels.")
    
    model = loadModel(mpath)
    if y.is_cuda:
        model = model.cuda()
    
    with th.no_grad(): out = model(y,stdn)
    
    return out       
        
    
    