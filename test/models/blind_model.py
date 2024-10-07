import logging
from collections import OrderedDict
import torchvision
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parallel import DataParallel, DistributedDataParallel
import numpy as np
# import models.lr_scheduler as lr_scheduler
import models.networks as networks
# from models.modules.loss import CharbonnierLoss
import math
from .base_model import BaseModel
from torch.nn import functional as F
from torchsummaryX import summary
from torch.cuda import Event

start_event = Event(enable_timing=True)
end_event = Event(enable_timing=True)
logger = logging.getLogger("base")
def crop_forward(model, x, sf, shave=10, min_size=100000, bic=None):
    """
    chop for less memory consumption during test
    """
    n_GPUs = 1
    scale = sf
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if bic is not None:
        bic_h_size = h_size*scale
        bic_w_size = w_size*scale
        bic_h = h*scale
        bic_w = w*scale

        bic_list = [
            bic[:, :, 0:bic_h_size, 0:bic_w_size],
            bic[:, :, 0:bic_h_size, (bic_w - bic_w_size):bic_w],
            bic[:, :, (bic_h - bic_h_size):bic_h, 0:bic_w_size],
            bic[:, :, (bic_h - bic_h_size):bic_h, (bic_w - bic_w_size):bic_w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            if bic is not None:
                bic_batch = torch.cat(bic_list[i:(i + n_GPUs)], dim=0)

            sr_batch_temp = model(lr_batch)

            if isinstance(sr_batch_temp, list):
                sr_batch = sr_batch_temp[-1]
            else:
                sr_batch = sr_batch_temp

            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            crop_forward(model, x=patch, stdn=stdn, shave=shave, min_size=min_size) \
            for patch in lr_list
            ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output



class B_Model(BaseModel):
    def __init__(self, opt):
        super(B_Model, self).__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt["dist"]:
            self.netG = DistributedDataParallel(
                self.netG, device_ids=[torch.cuda.current_device()]
            )
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            train_opt = opt["train"]
            # self.init_model() # Not use init is OK, since Pytorch has its owen init (by default)
            self.netG.train()

    def feed_data(self, LR_img, GT_img=None, ker_map=None):
        # window_size=8
        # _, _, h_old, w_old = LR_img.size()
        # h_pad = (h_old // window_size + 1) * window_size - h_old
        # w_pad = (w_old // window_size + 1) * window_size - w_old
        # img_lq = torch.cat([LR_img, torch.flip(LR_img, [2])], 2)[:, :, :h_old + h_pad, :]
        # LR_img = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        self.var_L = LR_img.to(self.device)
        if not (GT_img is None):
            self.real_H = GT_img.to(self.device)
        if not (ker_map is None):
            self.real_ker = ker_map.to(self.device)

    def test(self,LR_img ):
        self.netG.eval()
        with torch.no_grad():

            _, _, h_old, w_old = LR_img.size()
            output, kernel = self.netG(self.var_L)
            self.fake_SR = output
            self.fake_ker = kernel
            self.fake_SR = self.fake_SR[..., :h_old * 4, :w_old * 4]
            #self.fake_ker = kermaps[-1]
            self.netG.train()
    #
    # def test(self, LR_img):
    #     self.netG.eval()
    #
    #     self.img = self.var_L
    #     self.scale = 4
    #     self.tile_size = 256
    #     self.tile_pad = 24
    #     self.pre_pad = 0
    #     self.mod_scale = 4
    #     self.half = False
    #     if self.half:
    #         self.img = self.img.half()
    #
    #         # pre_pad
    #     if self.pre_pad != 0:
    #         self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
    #         # mod pad for divisible borders
    #     if self.scale == 2:
    #         self.mod_scale = 2
    #     elif self.scale == 1:
    #         self.mod_scale = 4
    #     if self.mod_scale is not None:
    #         self.mod_pad_h, self.mod_pad_w = 0, 0
    #         _, _, h, w = self.img.size()
    #         if (h % self.mod_scale != 0):
    #             self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
    #         if (w % self.mod_scale != 0):
    #             self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
    #         self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')
    #     batch, channel, height, width = self.img.shape
    #     output_height = height * self.scale
    #     output_width = width * self.scale
    #     output_shape = (batch, channel, output_height, output_width)
    #
    #     # start with black image
    #     self.output = self.img.new_zeros(output_shape)
    #     tiles_x = math.ceil(width / self.tile_size)
    #     tiles_y = math.ceil(height / self.tile_size)
    #
    #     # loop over all tiles
    #     for y in range(tiles_y):
    #         for x in range(tiles_x):
    #             # extract tile from input image
    #             ofs_x = x * self.tile_size
    #             ofs_y = y * self.tile_size
    #             # input tile area on total image
    #             input_start_x = ofs_x
    #             input_end_x = min(ofs_x + self.tile_size, width)
    #             input_start_y = ofs_y
    #             input_end_y = min(ofs_y + self.tile_size, height)
    #
    #             # input tile area on total image with padding
    #             input_start_x_pad = max(input_start_x - self.tile_pad, 0)
    #             input_end_x_pad = min(input_end_x + self.tile_pad, width)
    #             input_start_y_pad = max(input_start_y - self.tile_pad, 0)
    #             input_end_y_pad = min(input_end_y + self.tile_pad, height)
    #
    #             # input tile dimensions
    #             input_tile_width = input_end_x - input_start_x
    #             input_tile_height = input_end_y - input_start_y
    #             tile_idx = y * tiles_x + x + 1
    #             input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
    #
    #             # upscale tile
    #             try:
    #                 with torch.no_grad():
    #                     output_tile, ker = self.netG(input_tile)
    #             except RuntimeError as error:
    #                 print('Error', error)
    #             print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')
    #
    #             # output tile area on total image
    #             output_start_x = input_start_x * self.scale
    #             output_end_x = input_end_x * self.scale
    #             output_start_y = input_start_y * self.scale
    #             output_end_y = input_end_y * self.scale
    #
    #             # output tile area without padding
    #             output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
    #             output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
    #             output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
    #             output_end_y_tile = output_start_y_tile + input_tile_height * self.scale
    #
    #             # put tile into output image
    #             self.output[:, :, output_start_y:output_end_y,
    #             output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
    #                                            output_start_x_tile:output_end_x_tile]
    #
    #             torchvision.utils.save_image( self.output, 'out.png')
    #
    #     if self.mod_scale is not None:
    #         _, _, h, w = self.output.size()
    #         self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
    #         # remove prepad
    #     if self.pre_pad != 0:
    #         _, _, h, w = self.output.size()
    #         self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
    #     self.fake_SR =  self.output
    #     self.fake_ker = ker
    #     torchvision.utils.save_image(self.fake_SR, 'SR.png')
    #     torchvision.utils.save_image(self.fake_SR, 'SR.png')



    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["LQ"] = self.var_L.detach()[0].float().cpu()
        out_dict["SR"] = self.fake_SR.detach()[0].float().cpu()
        out_dict["GT"] = self.real_H.detach()[0].float().cpu()
        out_dict["ker"] = self.fake_ker.detach()[0].float().cpu()
        # out_dict["GT"] =  (self.real_H.detach()[0].float().cpu())
        out_dict["Batch_SR"] =(self.fake_SR.detach()[0].float().cpu())  # Batch SR, for train
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(
            self.netG, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.netG.__class__.__name__, self.netG.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt["path"]["strict_load"])


