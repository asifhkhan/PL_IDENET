import argparse

import torch
from torchsummaryX import summary
from ptflops import get_model_complexity_info

import options as option
from models import create_model

parser = argparse.ArgumentParser()
parser.add_argument(
    "-opt",
    type=str,
    default="options/setting1/test/test_setting1_x4.yml",
    help="Path to option YMAL file of Predictor.",
)
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)

opt = option.dict_to_nonedict(opt)
model = create_model(opt)
# macs, params = get_model_complexity_info(model.netG, (3, 270, 180), as_strings=True,
#                                            print_per_layer_stat=True, verbose=True)

# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))
test_tensor = torch.randn(1, 3, 16, 16).cuda()
summary(model.netG, x=test_tensor)

