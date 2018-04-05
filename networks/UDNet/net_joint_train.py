#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 22:32:58 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmatis@skoltech.ru
"""

import argparse
from net import UDNet

#import os.path
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from ... datasets.BSDS import BSDS
from math import log10


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
parser.add_argument('--kernel_size', type=tupleOfInts, default = '(5,5)', help="The spatial support of the filters in the network.")
parser.add_argument('--color', action='store_true', help="Type of images used to train the network.")
parser.add_argument('--num_filters', type == int, default = 74, help="Number of filters used in the convolution layer.")
parser.add_argument('--rbf_mixtures', type == int, default = 51, help="Number of RBF mixtures.")
parser.add_argument('--rbf_precision', type == int, default = 4, help="The precision for the RBF mixtures.")
parser.add_argument('--stages', type=int, default = 5, help="How many stages the network will consist of.")
parser.add_argument('--pad', type=tupleOfIntsorString, default = 'same', help="amount of padding of the input")
parser.add_argument('--padType', type=str, default = 'symmetric', help="The type of padding used before convolutions.")
parser.add_argument('--convWeightSharing', action='store_true',help="use shared weights for the convolution layers?")
parser.add_argument('--scale_f', action='store_true', help="use scaling for the convolution weights?")
parser.add_argument('--scale_t', action='store_true', help="use scaling for the transpose convolution weights?")
parser.add_argument('--normalizedWeights', action='store_true',help="use weightNormalization?")
parser.add_argument('--zeroMeanWeights', action='store_true',help="use zero-mean normalization?")
parser.add_argument('--rbf_start', type=int, default = -100, help="The lower bound of the interval where the RBF centers will be placed.")
parser.add_argument('--rbf_end', type=int, default = 100, help="The upper bound of the interval where the RBF centers will be placed.")
parser.add_argument('--data_min', type=int, default = -100, help="The minimum value of the data used to create the LUT for the computation of the RBF mixture response.")
parser.add_argument('--data_max', type=int, default = 100, help="The maximum value of the data used to create the LUT for the computation of the RBF mixture response.")
parser.add_argument('--data_step', type=int, default = 100, help="The step_size to be used for sampling uniformly the data in the range [data_min, data_max].")
parser.add_argument('--alpha', action='store_true', help="learn a scaling for the projection threshold?")
parser.add_argument('--clb', type=int, default = 0, help="The minimum valid intensity value of the output of the network.")
parser.add_argument('--cub', type=int, default = 255, help="The maximum valid intensity value of the output of the network.")

# Training parameters
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--stdn', type=tupleOfInts, default='(5,9,13,17,21,25,29)', help=" Number of noise levels (standard deviation) for which the network will be trained.")
# DataSet Parameters
parser.add_argument('--imdbPath', type=str, default='', help='random seed for data generation. Default=20180102')
parser.add_argument('--data_seed', type=int, default=20180102, help='random seed for data generation. Default=20180102')
opt = parser.parse_args()

print(opt)

input_channels = 3 if opt.color else 1
output_features = opt.num_filters

if opt.cuda and not th.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

th.manual_seed(opt.seed)
if opt.cuda:
    th.cuda.manual_seed(opt.seed)

print('===> Loading datasets')

#if len(opt.dataPath) == 0:
#    currentPath = os.path.dirname(os.path.realpath(__file__))
#    imdbPath = os.path.join(currentPath,"../../../datasets/BSDS500/")
#else:
#    imdbPath = opt.imdbPath



train_set = BSDS(opt.stdn,random_seed=opt.data_seed,filepath=opt.imdbPath,train=True,color=opt.color,shape=(180,180),im2Tensor=True)
test_set = BSDS(opt.stdn,random_seed=opt.data_seed,filepath=opt.imdbPath,train=False,color=opt.color,shape=(180,180),im2Tensor=True)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = UDNet(opt.kernel_size,input_channels,output_features,opt.rbf_mixtures,\
             opt.rbf_precision,opt.stages,opt.pad,opt.padType,opt.convWeightSharing,\
             opt.scale_f,opt.scale_t,opt.normalizedWeights,opt.zeroMeanWeights,\
             opt.rbf_start,opt.rbf_end,opt.data_min,opt.data_max,opt.data_step,\
             opt.alpha,opt.clb,opt.cub)
criterion = nn.MSELoss()

if opt.cuda:
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, sigma = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            sigma = sigma.cuda()

        optimizer.zero_grad()
        loss = criterion(model(input,sigma), target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target, sigma = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            sigma = sigma.cuda()

        prediction = model(input,sigma)
        mse = criterion(prediction, target)
        psnr = 20 * log10(255 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    th.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test()
    checkpoint(epoch)