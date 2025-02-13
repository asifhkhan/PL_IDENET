import argparse
import logging
import os
import os.path as osp
import sys
import time
from collections import OrderedDict
from glob import glob
# from piq import niqe

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
parser.add_argument("-input_dir", type=str, default="./sample_lr/")
parser.add_argument("-output_dir", type=str, default="./output/")
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)

opt = option.dict_to_nonedict(opt)


model = create_model(opt)

if not osp.exists(args.output_dir):
    os.makedirs(args.output_dir)

test_files = glob(osp.join(args.input_dir, "*bmp"))
total_inference_time = 0.0
for inx, path in tqdm(enumerate(test_files)):
    name = path.split("/")[-1].split('.bmp')[0]

    img = cv2.imread(path)[:, :, [2, 1, 0]]
    img = img.transpose(2, 0, 1)[None] / 255
    img_t = torch.as_tensor(np.ascontiguousarray(img)).float()

    model.feed_data(img_t)
    start_time = time.time()
    model.test(img_t)
    end_time = time.time()
    inference_time = end_time - start_time
    total_inference_time += inference_time

    sr = model.fake_SR.detach().float().cpu()[0]
    sr = sr*255.0
    sr_im = util.tensor2img(sr)

    save_path = osp.join(args.output_dir, "{}.bmp".format(name))
    cv2.imwrite(save_path, sr_im)

num_images = len(test_files)
average_inference_time = total_inference_time / num_images
print(f'Per-image inference time: {average_inference_time / num_images:.5f} seconds')
print(f'Average inference time: {average_inference_time:.5f} seconds')