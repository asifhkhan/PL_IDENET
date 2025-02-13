import argparse
import logging
import os
import os.path as osp
import sys
import time
from collections import OrderedDict
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr

#### options
parser = argparse.ArgumentParser()
parser.add_argument(
    "-opt",
    type=str,
    default="options/setting1/test/test_setting1_x4.yml",
    help="Path to options YMAL file.",
)
parser.add_argument("-input_dir", type=str, default="../../../data_samples/LR")
parser.add_argument("-output_dir", type=str, default="../../../data_samples/DANv1_SR")
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)

opt = option.dict_to_nonedict(opt)

model = create_model(opt)

if not osp.exists(args.output_dir):
    os.makedirs(args.output_dir)

img_t = torch.randn(100,3,32,32)

# Initialize variables
total_inference_time = 0.0

# Start iterating over patches
for i, img_t_patch in tqdm(enumerate(img_t, 1)):
    start_time = time.time()

    model.feed_data(img_t_patch.unsqueeze(0))
    # Measure inference time
    model.test(img_t_patch.unsqueeze(0))
    end_time = time.time()
    inference_time = end_time - start_time
    total_inference_time += inference_time

# Compute average inference time
num_patches = img_t.size(0)
average_inference_time = total_inference_time / num_patches
print(f'Average Inference time for {num_patches} patches: {average_inference_time:.5f} seconds')
