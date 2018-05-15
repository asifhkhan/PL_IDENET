#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 11:46:43 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmatis@skoltech.ru
"""

import torch as th

def cmul(input,other):
    r"""Returns the pointwise product of the elements of two complex tensors."""
    
    assert(th.is_tensor(input) and th.is_tensor(other)),"Inputs are expected "\
    +"to be tensors."
    
    assert(input.size(-1) == 2 and other.size(-1) == 2), "Inputs must be "\
    +"complex tensors (their last dimension should be equal to two)."
    
    assert(input.shape == other.shape), "Dimensions mismatch between inputs."
    
    real = input[...,0].mul(other[...,0])-input[...,1].mul(other[...,1])
    imag = input[...,0].mul(other[...,1])+input[...,1].mul(other[...,0])
    
    return th.cat((real.unsqueeze(-1),imag.unsqueeze(-1)),dim=-1)

def crmul(input,other):
    r"""Returns the pointwise product of the elements of a complex and a 
    real tensor."""
    
    assert(th.is_tensor(input) and th.is_tensor(other)),"Inputs are expected "\
    +"to be tensors."
    
    assert(input.size(-1) == 2 and input.shape[0:-1] == other.shape), "The "\
    "first input must be a complex tensor (its last dimension should be equal "\
    "to two) and the second input must be a real tensor."
        
    return input.mul(other.unsqueeze(-1).expand(*input.shape))

def cabs(input):
    r"""Returns the pointwise magnitude of the elements of the input complex tensor."""
    
    assert(th.is_tensor(input) and input.size(-1) == 2),"Inputs is expected "\
    +"to be a complex tensor."    
    
    return input.pow(2).sum(dim=-1).sqrt()

def cinv(input):
    r"""Returns the pointwise inverse of the elements of the input complex tensor."""
    
    assert(th.is_tensor(input) and input.size(-1) == 2),"Inputs is expected "\
    +"to be a complex tensor."    
    
    out = crdiv(conj(input),cabs(input).pow(2))
    
    return out

def cdiv(input,other):
    r"""Returns the pointwise division of the elements of two complex tensors."""
    
    assert(th.is_tensor(input) and th.is_tensor(other)),"Inputs are expected "\
    +"to be tensors."
    
    assert(input.size(-1) == 2 and other.size(-1) == 2), "Inputs must be "\
    +"complex tensors (their last dimension should be equal to two)."
    
    assert(input.shape == other.shape), "Dimensions mismatch between inputs."
    
    return cmul(input,conj(other)).div((cabs(other)**2).unsqueeze(-1))

def crdiv(input,other):
    r"""Returns the pointwise division of the elements of a complex and a 
    real tensor."""
    
    assert(th.is_tensor(input) and th.is_tensor(other)),"Inputs are expected "\
    +"to be tensors."
    
    assert(input.size(-1) == 2 and input.shape[0:-1] == other.shape), "The "\
    "first input must be a complex tensor (its last dimension should be equal "\
    "to two) and the second input must be a real tensor."
        
    return input.div(other.unsqueeze(-1).expand(*input.shape))

def cradd(input,other):
    r"""Returns the pointwise addition of the elements of a complex and a 
    real tensor."""
    
    assert(th.is_tensor(input) and th.is_tensor(other)),"Inputs are expected "\
    +"to be tensors."
    
    assert(input.size(-1) == 2 and input.shape[0:-1] == other.shape), "The "\
    "first input must be a complex tensor (its last dimension should be equal "\
    "to two) and the second input must be a real tensor."
    
    out = input.clone()
    out[...,0] += other
    return out

def conj(input):
    r"""Returns the complex conjugate of the input complex tensor."""
    
    assert(th.is_tensor(input) and input.size(-1) == 2),"Input is expected "\
    +"to be a complex tensor."
    
    out = input.clone()
    out[...,1] = -out[...,1]
    return out

def conj_(input):
    r"""Returns the complex conjugate of the input complex tensor (inplace operation)."""
    
    assert(th.is_tensor(input) and input.size(-1) == 2),"Input is expected "\
    +"to be a complex tensor."
    
    input[...,1] = - input[...,1]
    
def real(input):
    r"""Returns the real part of the input complex tensor."""
    
    assert(th.is_tensor(input)),"Input is expected to be a tensor."
    
    if input.size(-1) == 2:
        out = input[...,0]
    else:
        out = input.clone()
    
    return out

def imag(input):
    r"""Returns the imaginary part of the input complex tensor."""
    
    assert(th.is_tensor(input)),"Input is expected to be a tensor."
    
    if input.size(-1) == 2:
        out = input[...,1]
    else:
        out = th.zeros_like(input)
    
    return out

def complex(real,imag = None):
    
    if imag is not None:
        assert(real.shape == imag.shape),"Dimensions mismatch between real "\
        +"and imaginary input tensors."
    else:
        imag = th.zeros_like(real)
        
    return th.cat((real.unsqueeze(-1),imag.unsqueeze(-1)),dim = -1)
