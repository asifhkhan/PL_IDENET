#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:02:19 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmatis@skoltech.ru
"""
import torch as th
from pydl import utils

def dct(tensor):
    r"""Initializes the input tensor with weights from the dct basis or dictionary."""
    assert(tensor.ndimension() == 4),"A 4D tensor is expected."
    output_features,input_channels,H,W = tensor.shape
    
    if H*W*input_channels == output_features+1:
        tensor.data.copy_(utils.gen_dct3_kernel(tensor.shape[1:]).type_as(tensor)[1:,...])
    else:
        if input_channels == 1:
            weights = utils.odctndict((H,W),output_features+1)
        else:
            weights = utils.odctndict((H,W,input_channels),output_features+1)
        weights = weights[:,1:output_features+1].type_as(tensor).view(H,W,input_channels,output_features)
        weights = weights.permute(3,2,0,1)
        tensor.data.copy_(weights)

def rbf_lut(centers,sigma,start,end,step):
    r"""Computes necessary data for the Look-up table of rbf computation."""
    data_samples = th.range(start,end,step).type_as(centers)
    data_samples = data_samples.unsqueeze(1)
    data_samples = (data_samples - centers)/sigma
    return data_samples

def msra(tensor):
    r"""Initializes the input tensor with weights according to He initialization."""
    output_channels,input_channels,H,W = tensor.shape
    tensor.data.copy_(th.randn_like(tensor).\
        mul(th.sqrt(th.Tensor([2])).type_as(tensor).div(H*W*input_channels)))

def convWeights(tensor,init_type = 'dct'):
    if init_type == 'dct':
        dct(tensor)
    elif init_type == 'msra':
        msra(tensor)
    else: 
        raise NotImplementedError