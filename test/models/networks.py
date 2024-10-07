import logging

import torch

from models import modules as M

logger = logging.getLogger("base")

# Generator
def define_G(opt):
    opt_net = opt["network_G"]
    which_model = opt_net["which_model_G"]   # string path of model
    setting = opt_net["setting"]
    netG = getattr(M, which_model)(**setting)
    # print(netG)
    return netG


