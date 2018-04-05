#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 17:36:31 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmiatis@skoltech.ru
"""
import argparse


def tupleOfInts(s):   
    if s.find('(',0,1) > -1: # If the first character of the string is '(' then
        # this is a tuple and we keep only the substring with the values 
        # separated by commas, i.e., s[1:-1]. Then we create a list that holds
        # the characters which corresponds to the entries of the tuple by using
        # s[1:-1].split(',')
        s = tuple(int(i) for i in s[1:-1].split(','))
    else:
        s = int(s)
    return s

def tupleOfFloats(s):   
    if s.find('(',0,1) > -1: # If the first character of the string is '(' then
        # this is a tuple and we keep only the substring with the values 
        # separated by commas, i.e., s[1:-1]. Then we create a list that holds
        # the characters which corresponds to the entries of the tuple by using
        # s[1:-1].split(',')
        s = tuple(float(i) for i in s[1:-1].split(','))
    else:
        s = float(s)
    return s


def tupleOfIntsorString(s):   
    if s == "same":
        return s
    elif s.find('(',0,1) > -1: # If the first character of the string is '(' then
        # this is a tuple and we keep only the substring with the values 
        # separated by commas, i.e., s[1:-1]. Then we create a list that holds
        # the characters which corresponds to the entries of the tuple by using
        # s[1:-1].split(',')
        s = tuple(int(i) for i in s[1:-1].split(','))
    else:
        s = int(s)
    return s


parser = argparse.ArgumentParser(description='Joint-Training of UDNet')
# Network parameters
parser.add_argument('--kernel_size', type = tupleOfInts, default = '(5,5)', help="The spatial support of the filters in the network.")
parser.add_argument('--color', action='store_true', help="Type of images used to train the network.")
parser.add_argument('--num_filters', type = int, default = 74, help="Number of filters used in the convolution layer.")
parser.add_argument('--rbf_mixtures', type = int, default = 51, help="Number of RBF mixtures.")
parser.add_argument('--rbf_precision', type = int, default = 4, help="The precision for the RBF mixtures.")
parser.add_argument('--stages', type = int, default = 5, help="How many stages the network will consist of.")
parser.add_argument('--pad', type = tupleOfIntsorString, default = 'same', help="amount of padding of the input")
parser.add_argument('--padType', type = str, default = 'symmetric', help="The type of padding used before convolutions.")
parser.add_argument('--convWeightSharing', action='store_true',help="use shared weights for the convolution layers?")
parser.add_argument('--scale_f', action = 'store_true', help="use scaling for the convolution weights?")
parser.add_argument('--scale_t', action = 'store_true', help="use scaling for the transpose convolution weights?")
parser.add_argument('--normalizedWeights', action = 'store_true',help="use weightNormalization?")
parser.add_argument('--zeroMeanWeights', action = 'store_true',help="use zero-mean normalization?")
parser.add_argument('--rbf_start', type = int, default = -100, help="The lower bound of the interval where the RBF centers will be placed.")
parser.add_argument('--rbf_end', type = int, default = 100, help="The upper bound of the interval where the RBF centers will be placed.")
parser.add_argument('--data_min', type = int, default = -100, help="The minimum value of the data used to create the LUT for the computation of the RBF mixture response.")
parser.add_argument('--data_max', type = int, default = 100, help="The maximum value of the data used to create the LUT for the computation of the RBF mixture response.")
parser.add_argument('--data_step', type = float, default = 0.1, help="The step_size to be used for sampling uniformly the data in the range [data_min, data_max].")
parser.add_argument('--alpha', action = 'store_true', help="learn a scaling for the projection threshold?")
parser.add_argument('--clb', type = int, default = 0, help="The minimum valid intensity value of the output of the network.")
parser.add_argument('--cub', type = int, default = 255, help="The maximum valid intensity value of the output of the network.")
# Training parameters
parser.add_argument('--batchSize', type = int, default = 64, help='training batch size.')
parser.add_argument('--testBatchSize', type = int, default = 100, help='testing batch size.')
parser.add_argument('--nEpochs', type = int, default = 100, help='number of epochs to train for.')
parser.add_argument('--lr', type = float, default = 1e-3, help='learning rate. Default=1e-3.')
parser.add_argument('--lr_milestones', type = int, nargs = '+', default = [100], help="Scheduler's learning rate milestones. Default=[100].")
parser.add_argument('--lr_gamma', type = float, default = 0.1, help="multiplicative factor of learning rate decay.")
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type = int, default = 4, help='number of threads for data loader to use.')
parser.add_argument('--seed', type = int, default = 123, help='random seed to use. Default=123.')
parser.add_argument('--stdn', type = tupleOfFloats, default='(5,9,13,17,21,25,29)', help=" Number of noise levels (standard deviation) for which the network will be trained.")
# DataSet Parameters
parser.add_argument('--imdbPath', type = str, default = '', help='location of the dataset.')
parser.add_argument('--data_seed', type = int, default = 20180102, help='random seed for data generation. Default=20180102')
opt = parser.parse_args()

print(opt)


#class Namespace:
#    def __init__(self, **kwargs):
#        self.__dict__.update(kwargs)
#
#opt = 