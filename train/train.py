################################
## Imports
################################
import os
import math
import torch
import numpy as np
import logging
from collections import OrderedDict
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import utils as util
import argparse, random
from ski_blur import BatchBlur
import utils.file_utils
from imageio import imwrite as save
from models.ResDNet import ResDNet
from models.RM_LFWM import SRResDNet
from models.IDENet import DEblurSRResDNet
import models.discriminator_arch as SRGAN_arch
# from models.deblur import DEBLUR
# import models.discriminator_vgg_arch as SRGAN_arch
from modules.loss import GANLoss, TV_L2Loss, TV_L1LOSS
from modules.loss import CharbonnierLoss

from modules import filters, io, initialization, module_info, lr_scheduler, PerceptualLoss
import torch.nn.functional as F
from models.kernel_prediction_module import KernelEstimator
import options as option
import torch
from data import util as ut
from pydl.networks.WienerDeblurNet.net import WienerDeblurNet

import wandb
from build import losses as losses_builder
from build import optimizers as optimizers_builder
from build import datasets as dataset_builder
from evaluation import evaluator

from models.plrefinement_module import SwinIR


import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

torch.backends.cudnn.benchmark = True


# def findLastCheckpoint(expDir):
#     """Finds the latest checkpoint."""
#     lfiles = os.listdir(expDir)
#     lfiles = [i for i in lfiles if pattern_match('model_epoch',i)]
#     if len(lfiles) == 0 or (len(lfiles) == 1 and lfiles[0].split('epoch_')[-1].split('.')[0] == 'best'):
#         return 0
#     else:
#         lfiles = [lfiles[i].split('epoch_')[-1].split('.')[0] for i in range(len(lfiles))]
#         return max(int(i) for i in lfiles if i != 'best')

def tupleOfData(s, dtype):
    if s.find('(', 0, 1) > -1:  # If the first character of the string is '(' then
        # this is a tuple and we keep only the substring with the values
        # separated by commas, i.e., s[1:-1]. Then we create a list that holds
        # the characters which corresponds to the entries of the tuple by using
        # s[1:-1].split(',')
        s = tuple(dtype(i) for i in s[1:-1].replace(" ", "").split(',') if i != "")
    else:
        s = dtype(s)
    return s


# def tupleOfBools(s):
#     if s.find('(',0,1) > -1: # If the first character of the string is '(' then
#         # this is a tuple and we keep only the substring with the values
#         # separated by commas, i.e., s[1:-1]. Then we create a list that holds
#         # the characters which corresponds to the entries of the tuple by using
#         # s[1:-1].split(',')
#         s = tuple(i == 'True' for i in s[1:-1].replace(" ","").split(',') if i!="")
#     else:
#         s = (s == 'True')
#     return s

tupleOfInts = lambda s: tupleOfData(s, int)
tupleOfFloats = lambda s: tupleOfData(s, float)


def tupleOfIntsorString(s):
    if s == "same":
        return s
    elif s.find('(', 0, 1) > -1:  # If the first character of the string is '(' then
        # this is a tuple and we keep only the substring with the values
        # separated by commas, i.e., s[1:-1]. Then we create a list that holds
        # the characters which corresponds to the entries of the tuple by using
        # s[1:-1].split(',')
        s = tuple(int(i) for i in s[1:-1].split(','))
    else:
        s = int(s)
    return s


def main():
    wandb.login()

    ################################
    ##  Network parameters
    ################################

    parser = argparse.ArgumentParser(description='Image Super-resolution with prompt_Swinir')
    # read yml file
    parser.add_argument("-opt", type=str, default=
    '/home/asif/Documents/kernel_est/experiment5/pl_idenet/setting1/train/train_setting1_x4.yml',
                        help="Path to option YMAL file.")
    # parser.add_argument("-opt", type=str,
    #                     default='/home/asif/Documents/kernel_est/experiment4/SRResCGAN_master/training_codes/setting1/train/train_setting1_x2.yml',help="Path to option YMAL file.")

    # Debluring sub-network parameters
    parser.add_argument('--color', default=True, action='store_true', help="Type of images used to train the network.")
    parser.add_argument('--wiener_kernel_size', type=tupleOfInts, default='(5,5)',
                        help="The spatial support of the filters used in the Wiener debluring filter.")
    parser.add_argument('--wiener_output_features', type=int, default=24,
                        help="Number of filters used in the Wiener debluring layer.")
    parser.add_argument('--numWienerFilters', type=int, default=4, help="Number of Wiener debluring layers.")
    parser.add_argument('--wienerWeightSharing', default=True, action='store_true',
                        help="use shared weights for the Wiener debluring layers?")
    parser.add_argument('--wienerChannelSharing', default=True, action='store_true',
                        help="use shared weights for the different image channels in the Wiener debluring layers?")
    parser.add_argument('--alphaChannelSharing', default=True, action='store_true',
                        help="use shared alpha weights for the different image channels in the Wiener debluring layers?")
    parser.add_argument('--alpha_update', default=True, action='store_true',
                        help="Learn the alpha weights for the Wiener debluring layers?")
    parser.add_argument('--lb', type=float, default=1e-5, help="The minimum value of the alpha parameter.")
    parser.add_argument('--ub', type=float, default=1e-2, help="The maximum value of the alpha parameter.")
    parser.add_argument('--wiener_pad', default=True, action='store_true', help="Pad the input image before debluring?")
    parser.add_argument('--wiener_padType', type=str, default='symmetric', help="What padding to use?")
    parser.add_argument('--edgeTaper', default=True, action='store_true', help="Use edge tapering?")
    parser.add_argument('--wiener_scale', action='store_true',
                        help="use scaling for the Wiener debluring layer weights?")
    parser.add_argument('--wiener_normalizedWeights', default=True, action='store_true',
                        help="use weightNormalization?")
    parser.add_argument('--wiener_zeroMeanWeights', default=True, action='store_true',
                        help="use zero-mean normalization?")

    parser.add_argument('--kernel_size', type=tupleOfInts, default='(5,5)',
                        help="The spatial support of the filters in the network.")
    parser.add_argument('--conv_init', type=str, default='dct',
                        help='type of initialization for the convolutional layers.')
    parser.add_argument('--pad', type=tupleOfIntsorString, default='same', help="amount of padding of the input")
    parser.add_argument('--padType', type=str, default='symmetric',
                        help="The type of padding used before convolutions.")

    # SRResdnet Parameters
    parser.add_argument("--in_nc", type=int, default=3, help='no. of in_chs for D')
    parser.add_argument("--nf", type=int, default=64, help='no. of feat. maps for D')
    parser.add_argument("--resdnet_depth", type=int, default=5, help='no. of resblocks for resdnet')
    # SwinIR parameters
    parser.add_argument('--patch', type=int, default=1, help='patch size for training. ')
    parser.add_argument('--embed_dim', type=int, default=96, help='embded dimension. ')
    parser.add_argument('--depths', type=list, default=[4,4,4,4], help='depth. ')
    parser.add_argument('--num_heads', type=list, default=[4,4,4,4], help='number of heads. ')
    parser.add_argument('--window_size', type=int, default=8, help='window size. ')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='window size. ')
    parser.add_argument('--qkv_bias', type=bool, default=True, help='query key value. ')
    parser.add_argument('--qk_scale', default=None,  help='query key scale. ')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='drop ratio. ')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, help='attention drop ratio. ')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help='drop ratio. ')
    parser.add_argument('--ape', type=bool, default=False, help='ape. ')
    parser.add_argument('--patch_norm', type=bool, default=True, help='patch norm. ')
    parser.add_argument('--use_checkpoint', type=bool, default=False, help='use_checkpoint. ')
    parser.add_argument('--upscale', type=int, default=4, help='patch norm. ')
    parser.add_argument('--img_range', type=float, default=1., help='image range. ')
    parser.add_argument('--upsampler', type=str, default='', help='upsampler ')
    parser.add_argument('--resi_connection', type=str, default='1conv', help='upsampler ')
    # Training Parameters
    parser.add_argument('--train_stdn', type=list, default=[0.0],
                        help=" Number of noise levels (standard deviation) for which the network will be trained.")  # [1.0,2.0,2.55,3.0,4.,5.0,5.10,6.0,7.,7.65,8.0,9.0,10.,11.0,12.0,12.75,13.0,14.0,15.0]
    parser.add_argument('--test_stdn', type=list, default=[0.0],
                        help=" Number of noise levels (standard deviation) for testing.")  # [2.55, 5.10, 7.65, 12.75]
    parser.add_argument('--upscale_factor', type=int, default=4, help='scaling factor.')  # [2, 3, 4]
    parser.add_argument('--trainBatchSize', type=int, default=2, help='training batch size.')
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size.')
    parser.add_argument('--niter', type=int, default=500000, help='number of iters to train for.')
    parser.add_argument('--use_bn', type=bool, default=False, help='use Batch-Norm?')
    # parser.add_argument('--gpu',type = int, default = [1], nargs ='+', help ='used gpu')
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
    parser.add_argument('--gpu_ids', type=list, default=[0], help='use number of gpus for training')
    # parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123.')
    parser.add_argument('--use_filters', type=bool, default=False, help='use Filters: LP, HP?')
    parser.add_argument('--resume', type=bool, default=True, help='resume training?')
    parser.add_argument('--resume_start_epoch', type=int, default=0, help='Where to resume training?.')
    parser.add_argument('--pretrainedModelPath', type=str, default='pretrained_nets/idenet/2853_final_G.pth',
                        help='location of pretrained model.')
    parser.add_argument('--pretrain', type=bool, default=False, help='Initialize the model paramaters from a '
                                                                     'pretrained net.')
    parser.add_argument('--tile', type=int, default=128,help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--whole', type=bool, default=True, help='test image as a whole')
    # DataSet Parameters
    parser.add_argument('--imdbTrainPath', type=str, default='datasets/', help='location of the training dataset.')
    parser.add_argument('--imdbTestPath', type=str, default='datasets/', help='location of the testing dataset.')
    parser.add_argument('--patch_size', type=int, default=32, help='patch size for training. [x2-->64,x3-->42,'
                                                                   'x4-->32]')
    parser.add_argument('--rgb_range', type=int, default=1, help='data range of the training images.')
    parser.add_argument('--is_train', type=bool, default=True, help=' True for training phase')
    parser.add_argument('--is_mixup', type=bool, default=True, help=' mixup_data augmentation for training data')
    parser.add_argument('--use_chop', type=bool, default=False, help=' chop for less memory consumption during test.')
    parser.add_argument('--alpha', type=float, default=1.2, help='alpha for data mixup (uniform=1., ERM=0.)')
    parser.add_argument('--numWorkers', type=int, default=4, help='number of threads for data loader to use.')
    # Optimizer Parameters
    parser.add_argument('--lr_G', type=float, default=1e-4, help='learning rate for G.')
    parser.add_argument('--beta1_G', type=float, default=0.9, help='learning rate. Default=0.9.')
    parser.add_argument('--beta2_G', type=float, default=0.999, help='learning rate. Default=0.999.')
    parser.add_argument('--eps_G', type=float, default=1e-8, help='learning rate. Default=1e-8.')
    parser.add_argument('--weightdecay_G', type=float, default=0, help='learning rate. Default=0.')
    parser.add_argument('--lr_D', type=float, default=1e-4, help='learning rate for D.')
    parser.add_argument('--beta1_D', type=float, default=0.9, help='learning rate. Default=0.9.')
    parser.add_argument('--beta2_D', type=float, default=0.999, help='learning rate. Default=0.999.')
    parser.add_argument('--eps_D', type=float, default=1e-8, help='learning rate. Default=1e-8.')
    parser.add_argument('--weightdecay_D', type=float, default=0, help='learning rate. Default=0.')
    parser.add_argument('--amsgrad', type=bool, default=False, help='Use the fix for Adam?')
    parser.add_argument('--lr_milestones', type=list, default=[250000, 400000, 450000, 475000],
                        help="Scheduler's learning rate milestones.")
    parser.add_argument('--lr_gamma', type=float, default=0.5, help="multiplicative factor of learning rate decay.")
    parser.add_argument('--lr_restart', default=None, help='lr restart.')
    parser.add_argument('--lr_restart_weights', default=None, help='lr restart weights.')
    parser.add_argument('--warmup_iter', type=int, default=-1, help='warmup iter.')
    parser.add_argument('--D_update_ratio', type=int, default=1, help='D_update_ratio.')
    parser.add_argument('--D_init_iters', type=int, default=0, help='D_init_iters.')
    # losses Parameters
    parser.add_argument('--pixel_criterion', type=str, default='l1', help='pixel-wise criteria.')
    parser.add_argument('--pixel_criterion_char', type=str, default='charbonnair', help='pixel-wise criteria.')
    parser.add_argument('--pixel_criterion_kernel', type=str, default='l1', help='pixel-wise criteria.')
    parser.add_argument('--pixel_criterion_GT_kernel', type=str, default='l1', help='pixel-wise criteria.')
    parser.add_argument('--feature_criterion', type=str, default='l1', help='feature criteria.')
    parser.add_argument('--tv_criterion', type=str, default='l1', help='TV criteria.')
    parser.add_argument('--gan_type', type=str, default='gan', help='gan type. default: gan | ragan')
    parser.add_argument('--pixel_weight', type=float, default=1., help='weight for pixel-wise criteria. default: 1e-2')
    parser.add_argument('--pixel_kernel_weight', type=float, default=1.,
                        help='weight for pixel-wise criteria for kernel. default: 1e-2')
    parser.add_argument('--pixel_weight_char', type=float, default=1.,
                        help='weight for char pixel-wise criteria. default: 1e-2')
    parser.add_argument('--pixel_kernel_gt_weight', type=float, default=1.,
                        help='weight for pixel-wise criteria for kernel. default: 1e-2')
    parser.add_argument('--feature_weight', type=float, default=1., help='weight for feature criteria.')
    parser.add_argument('--layer_weights', type=dict, default={ 'conv1_2': 0.1,  'conv2_2': 0.1, 'conv3_4': 1, 'conv4_4': 1,  'conv5_4': 1 }, help='weight for feature criteria.')
    parser.add_argument('--tv_weight', type=float, default=1., help='weight for TV criteria.')
    parser.add_argument('--gan_weight', type=float, default=0.1, help='weight for gan | ragan criteria. default: 5e-3')
    # Results Output Parameters
    parser.add_argument('--saveTrainedModelsPath', type=str, default='trained_nets', help='location of trained models.')
    parser.add_argument('--save_path_training_states', type=str, default='/training_states/',
                        help='location of training states.')
    parser.add_argument('--save_path_netG', type=str, default='/netG/', help='location of trained netG.')
    parser.add_argument('--save_path_netD', type=str, default='/netD/', help='location of trained netD.')
    parser.add_argument('--save_path_best_psnr', type=str, default='/best_psnr/',
                        help='location of trained model best PSNR.')
    parser.add_argument('--save_path_best_lpips', type=str, default='/best_lpips/',
                        help='location of trained model best LPIPS.')
    parser.add_argument('--saveImgsPath', type=str, default='results',
                        help='location of saved images on training validation.')
    parser.add_argument('--saveLogsPath', type=str, default='logs', help='location of training logs.')
    parser.add_argument('--save_checkpoint_freq', type=float, default=50,
                        help='Every how many iters we save the model parameters, default:5e3.')
    parser.add_argument('--saveBest', type=bool, default=True, help='save the best model parameters?')

    opt = parser.parse_args()

    run = wandb.init(
        # Set the project where this run will be logged
        project="wiener_filter_SR",
        # Track hyperparameters and run metadata
        config=vars(opt)
    )

    # os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(x) for x in opt.gpu)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    # shave boader to calculate PSNR and SSIM
    # border = opt.upscale_factor
    border = 0
    opt_pca = option.parse(opt.opt)
    # load PCA matrix of enough kernel
    print("load PCA matrix")
    pca_matrix = torch.load(opt_pca["pca_matrix_path"], map_location=lambda storage, loc: storage)
    print("PCA matrix shape: {}".format(pca_matrix.shape))

    # Setup output folders
    logs_save_path, training_states_save_path, save_test_imgs_path,best_psnr_save_path, best_lpips_save_path, netG_save_path,netD_save_path, save_train_imgs_path  = utils.file_utils.create_output_folders(opt)

    # setup logging 
    logger = util.utils_logger.logger_info('train_prompt_Swinir', log_path=os.path.join(logs_save_path, 'train_prompt_Swinir.log'))

    # save the training arguments
    torch.save(opt, os.path.join(logs_save_path, "args.pth"))

    logger.info('===================== Selected training parameters =====================')
    logger.info('{:s}.'.format(str(opt)))

    ################################
    ## datasets preparation 
    ################################
    train_dataset, trainset_loader, test_dataset, testset_loader= dataset_builder.build(opt, logger)

    ################################
    ## Functions
    ################################


    def train_generator(lr_image, hr_gt_image,gt_blur_kernel, netG, netD, optimizer_G,
                        cri_pix, cri_pix_k,cri_pix_gt_k,l_pix_k_gt_w, cri_fea, cri_tv, cri_gan, cri_pix_char, l_pix_w_char, opt):

        netG.train()

        # Freeze discriminator
        for p in netD.parameters():
            p.requires_grad = False

        optimizer_G.zero_grad()

        # Forward generator
        hr_image_super_resolved, estimated_kernel = netG(lr_image)


        ##############################GT_kernel Loss######################
        #est_kernel_img = estimated_kernel[0]
        #save_image(est_kernel_img, "estimated_k.png")
        # est_kernel = kernel_output
        ##################################################################

        # lr_image_estimated_kernel = None
        GEN_SKI = BatchBlur(21, opt.upscale_factor)

        lr_image_estimated_kernel = GEN_SKI.gen_ski(hr_gt_image, estimated_kernel)

        l_g_total = 0
        # if step % opt.D_update_ratio == 0 and step > opt.D_init_iters:
        # l_g_pix = 0

        if cri_pix:  # pixel loss

            # l_g_pix_f = l_pix_w * cri_pix(filter_low(fake_H), filter_low(var_H))

            l_g_pix_f = l_pix_w * cri_pix(hr_image_super_resolved, hr_gt_image)
            l_g_pix_nf = l_pix_w * cri_pix(hr_image_super_resolved, hr_gt_image)

            if opt.use_filters:
                l_g_pix = l_g_pix_f
            else:
                l_g_pix = l_g_pix_nf

            l_g_total += l_g_pix

        # l_g_pix_k = 0


        if cri_pix_k:  # pixel kernel loss
        
            # l_g_pix_k = l_pix_k_w * cri_pix_k(F.interpolate(F.conv2d(fake_H, fake_K, padding='same'),scale_factor=1/opt.upscale_factor, mode='bilinear', align_corners=False ), var_L)  #B, 3, 32, 32, var_L = B, 3, 32 32
            l_g_pix_k = l_pix_k_w * cri_pix_k(lr_image_estimated_kernel, lr_image)  # B, 3, 32, 32, var_L = B, 3, 32 32
            l_g_total += l_g_pix_k
            


        l_g_pix_char = 0
        """"
        if cri_pix_char:
            l_g_pix_char = l_pix_w_char * cri_pix_char(hr_image_super_resolved, hr_gt_image)
            l_g_total += l_g_pix_char
            
        """

        l_g_pix_gt_k = 0
        """"
        ###############################GT_kernel#######################################
        if cri_pix_gt_k:
           l_g_pix_gt_k = l_pix_k_gt_w * cri_pix_gt_k(estimated_kernel, gt_blur_kernel)
           l_g_total += l_g_pix_gt_k
        ###############################################################################
        """
        l_g_fea = 0

        '''
        if cri_fea:  # feature loss
            # real_fea = netF(hr_gt_image).detach()
            # fake_fea = netF(hr_image_super_resolved)
            l_g_fea = netF(hr_image_super_resolved, hr_gt_image)
            l_g_total += l_g_fea
        '''
        # l_g_tv = 0


        if cri_tv:  # TV loss
            l_g_tv = l_tv_w * cri_tv(hr_image_super_resolved, hr_gt_image)
            l_g_total += l_g_tv
            



        '''
        # Optional high pass filter, return x if not enabled
        optional_filter_func = lambda x: filter_high(x) if opt.use_filters else x

        # Discriminator predictions
        pred_g_fake = netD(optional_filter_func(hr_image_super_resolved))

        if opt.gan_type == 'gan':
            l_g_gan = l_gan_w * cri_gan(pred_g_fake, True)
        elif opt.gan_type == 'ragan':
            pred_d_real = netD(optional_filter_func(hr_gt_image)).detach()
            l_g_gan = l_gan_w * (
                    cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                    cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
        l_g_total += l_g_gan
        '''

        l_g_gan = 0

        l_g_total.backward()
        optimizer_G.step()

        return hr_image_super_resolved, estimated_kernel, gt_blur_kernel, lr_image_estimated_kernel, \
               {'l_g_pix': l_g_pix, 'l_g_pix_k': l_g_pix_k, 'l_g_pix_char':l_g_pix_char, 'l_g_pix_gt_k':l_g_pix_gt_k, 'l_g_fea': l_g_fea,
                'l_g_tv': l_g_tv, 'l_g_gan': l_g_gan, 'l_g_total': l_g_total}

    def train_discriminator(hr_image_super_resolved, hr_gt_image, netD, optimizer_D, cri_gan):
        # D
        netD.train()
        for p in netD.parameters():
            p.requires_grad = True

        optimizer_D.zero_grad()
        l_d_total = 0

        # Optional high pass filter, return x if not enabled
        optional_filter_func = lambda x: filter_high(x) if opt.use_filters else x

        # Predictions on real and fake HR images
        # detach to avoid BP to G
        pred_d_real = netD(optional_filter_func(hr_gt_image))
        pred_d_fake = netD(optional_filter_func(hr_image_super_resolved.detach()))

        if opt.gan_type == 'gan':
            l_d_real = cri_gan(pred_d_real, True)
            l_d_fake = cri_gan(pred_d_fake, False)
            l_d_total = l_d_real + l_d_fake
        elif opt.gan_type == 'ragan':
            l_d_real = cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            l_d_fake = cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            l_d_total = (l_d_real + l_d_fake) / 2

        l_d_total.backward()
        optimizer_D.step()

        # l_d_real_batch = l_d_real.item()
        # l_d_fake_batch = l_d_fake.item()
        # D_real_batch = torch.mean(pred_d_real.detach())
        # D_fake_batch = torch.mean(pred_d_fake.detach())
        # logger.info("===> train:: l_d_real Loss:{:.6f}".format(l_d_real_batch))
        # logger.info("===> train:: l_d_fake Loss:{:.6f}".format(l_d_fake_batch))
        # logger.info("===> train:: l_d_total Loss:{:.6f}".format(l_d_total.item()))
        #
        # logger.info("===> train:: D_real output:{:.6f}".format(D_real_batch))
        # logger.info("===> train:: D_fake output:{:.6f}".format(D_fake_batch))

        return {'l_d_fake': l_d_fake, 'l_d_real': l_d_real, 'l_d_total': l_d_total}

    def train(hr_gt_image, lr_image, sigma, gt_blur_kernel,
              netG, netD, netF,
              optimizers,
              cri_pix, l_pix_w,
              cri_pix_k, cri_pix_gt_k, l_pix_k_w, l_pix_k_gt_w,
              cri_fea, l_fea_w,
              cri_tv, l_tv_w,
              cri_gan, l_gan_w,
             cri_pix_char, l_pix_w_char,
              filter_low, filter_high,
              step):


        optimizer_G, optimizer_D = optimizers[0], optimizers[1]

        # set log
        # if step % opt.D_update_ratio == 0 and step > opt.D_init_iters:
        #     if cri_pix:
        #         l_g_pix_batch = l_g_pix.item()
        #     if cri_pix_k:
        #         l_g_pix_batch_k = l_g_pix_k.item()
        #     if cri_pix_gt:
        #         l_g_pix_batch_gt_k = l_g_pix_gt_k.item()
        #     if cri_fea:
        #         l_g_fea_batch = l_g_fea.item()
        #     if cri_tv:
        #         l_g_tv_batch = l_g_tv.item()
        #     l_g_gan_batch = l_g_gan.item()

        # logger.info("===> train:: l_g_pix Loss:{:.6f}".format(l_g_pix_batch))
        # logger.info("===> train:: l_g_pix_k Loss:{:.6f}".format(l_g_pix_batch_k))
        # logger.info("===> train:: l_g_pix_gt_k Loss:{:.6f}".format(l_g_pix_batch_gt_k))
        # logger.info("===> train:: l_g_fea Loss:{:.6f}".format(l_g_fea_batch))
        # logger.info("===> train:: l_g_tv Loss:{:.6f}".format(l_g_tv_batch))
        # logger.info("===> train:: l_g_gan Loss:{:.6f}".format(l_g_gan_batch))

        hr_image_super_resolved, estimated_kernel,gt_blur_kernel, lr_image_estimated_kernel, g_losses = \
            train_generator(lr_image, hr_gt_image,gt_blur_kernel, netG, netD, optimizer_G,
                            cri_pix, cri_pix_k,cri_pix_gt_k,l_pix_k_gt_w, cri_fea, cri_tv, cri_gan,cri_pix_char, l_pix_w_char, opt)

        # Unlock netD training after "opt.D_init_iters" steps
        d_losses = {}

        # if step > opt.D_init_iters:
            # d_losses = train_discriminator(hr_image_super_resolved, hr_gt_image, netD, optimizer_D, cri_gan)

        psnr_batch = util.psnr(hr_image_super_resolved*255.0, hr_gt_image*255.0, border=border)
        # psnr_batch =0

        wandb.log(g_losses | d_losses | {"psnr_batch": psnr_batch})

        return hr_image_super_resolved, estimated_kernel,gt_blur_kernel, lr_image_estimated_kernel, \
               g_losses['l_g_total'].item(), psnr_batch, \
               netG, netD

    ################################
    ## NN Architecture
    ################################
    logger.info('===================== Building model =====================')

    # Parameters that we need to specify in order to initialize our model
    params = OrderedDict(input_channels=opt.in_nc, wiener_kernel_size=opt.wiener_kernel_size, \
                         wiener_output_features=opt.wiener_output_features, numWienerFilters=opt.numWienerFilters, \
                         wienerWeightSharing=opt.wienerWeightSharing, wienerChannelSharing= \
                             opt.wienerChannelSharing, alphaChannelSharing=opt.alphaChannelSharing, \
                         alpha_update=opt.alpha_update, lb=opt.lb, ub=opt.ub, wiener_pad=opt.wiener_pad, \
                         wiener_padType=opt.wiener_padType, edgeTaper=opt.edgeTaper, \
                         wiener_scale=opt.wiener_scale, wiener_normalizedWeights=opt.wiener_normalizedWeights, \
                         wiener_zeroMeanWeights=opt.wiener_zeroMeanWeights, )

    # Discriminator net
    # netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt.in_nc, nf=opt.nf)
    netD = SRGAN_arch.UNetDiscriminatorSN(num_in_ch=opt.in_nc, num_feat=opt.nf, skip_connection=True)


    # debulr net
    wienernet = WienerDeblurNet(*params.values())


    ### Kernel Prediction module###
    netK = KernelEstimator()

    norm_layer = nn.LayerNorm
    #########Refinement Module###########
    netT = SwinIR(opt.patch_size, opt.patch,opt.in_nc, opt.embed_dim, opt.depths, opt.num_heads,opt.window_size, opt.mlp_ratio, opt.qkv_bias, opt.qk_scale, opt.drop_rate, opt.attn_drop_rate, opt.drop_path_rate, norm_layer, opt.ape,opt.patch_norm, opt.use_checkpoint, opt.upscale_factor, opt.img_range, opt.upsampler, opt.resi_connection)

    #########Refinement and Learnbale WienerFilter Module###########
    netDe = SRResDNet(netT,wienernet, scale=opt.upscale_factor)  # stdn=torch.FloatTensor(opt.train_stdn).cuda()

    #####netG############
    netG = DEblurSRResDNet(netDe, netK)


    # initialize model with pertrained network
    if opt.pretrain:
        logger.info('Initialized model with pretrained net from {:s}.'.format(opt.pretrainedModelPath))
        netG = io.load_network(opt.pretrainedModelPath, netG)
    # else:
        # netG = initialization.init_weights(netG, init_type='kaiming')
        # netD = initialization.init_weights(netD, init_type='kaiming')
        # netK = _net_init(netK, init_type='kaiming')

    # Filters: low-pass: W_L, high-pass: W_H
    filter_low = filters.FilterLow()
    filter_high = filters.FilterHigh()
    if opt.cuda:
        netG = nn.DataParallel(netG, device_ids=opt.gpu_ids).cuda()
        netD = nn.DataParallel(netD, device_ids=opt.gpu_ids).cuda()
        filter_low = filter_low.cuda()
        filter_high = filter_high.cuda()

    # optimizers
    optimizers, schedulers = optimizers_builder.build(netG, netD, opt)

    # losses criteria
    cri_pix, l_pix_w, cri_pix_k, l_pix_k_w, cri_pix_gt_k, l_pix_k_gt_w, cri_tv, l_tv_w, \
    cri_fea, l_fea_w, cri_gan, l_gan_w, netF, cri_pix_char, l_pix_w_char = losses_builder.build(opt)

    # lpips distance metric for test
    lpips = PerceptualLoss(model='net-lin', net='alex', use_gpu=opt.cuda)  # alex, squeeze, vgg

    # print networks
    module_info.print_network(netG, netD, netF, netK, logger)

    ################################
    ## Main
    ################################
    # start training
    logger.info('===================== start training =====================')
    # resume training
    if opt.resume:
        logger.info('===================== resume training =====================')
        resume_path = training_states_save_path
        if not os.listdir(resume_path):
            logger.info('===> No saved training states to resume.')
            current_step = 0
            start_epoch = 0
            epoch_psnr_old = -float('inf')
            epoch_lpips_old = float('inf')
            logger.info('===> start training from epoch: {}, iter: {}.'.format(start_epoch, current_step))
        else:
            resume_start_epoch = opt.resume_start_epoch
            resume_path = training_states_save_path + str(resume_start_epoch) + '_tr_states.pth'
            start_epoch, current_step, epoch_psnr_old, epoch_lpips_old, optimizers, schedulers = io.resume_training(
                resume_path, optimizers, schedulers, opt)
            logger.info('===> loading pretrained models: G,k,D.')
            load_path_G = netG_save_path + str(start_epoch) + '_G.pth'
            load_path_D = netD_save_path + str(start_epoch) + '_D.pth'
            netG, netD = io.custom_load(netG, netD, load_path_G, load_path_D, logger)
            logger.info('===> Resuming training from epoch: {}, iter: {}.'.format(start_epoch, current_step))

    # training loop
    t = util.timer()
#    t.tic()
    total_iters = int(opt.niter)
    train_size = int(math.ceil(len(train_dataset) / opt.trainBatchSize))
    total_epochs = int(math.ceil(total_iters / train_size))
    logger.info('Total # of epochs for training: {}.'.format(total_epochs))

    prepro = util.SRMDPreprocessing(scale=opt_pca["scale"], pca_matrix=pca_matrix, cuda=True,
                                    **opt_pca["degradation"])


    for epoch in range(start_epoch + 1, total_epochs + 1):
        # print('===> Epoch %d' % epoch)
        logger.info("===> train:: Epoch[{}]".format(epoch))
        epoch_loss = 0
        epoch_psnr = 0

        # Force pre-run empty cache
        torch.cuda.empty_cache()

        for _, data in enumerate(trainset_loader):
            current_step += 1
            if current_step > total_iters:
                break

            #### update learning rate
            lr_scheduler.update_learning_rate(optimizers, schedulers, cur_iter=current_step, warmup_iter=opt.warmup_iter)

            #logger.info("===> train:: Epoch[{}] \t Iter-step[{}]".format(epoch, current_step))
            hr_gt_image, sigma = data['HR'], data['sigma']

            #####image blur
            lr_image, ker_map, b_kernel = prepro(hr_gt_image)
            lr_image = (lr_image * 255).round() / 255

            b_kernel = b_kernel.unsqueeze(1)


            if opt.cuda:
                lr_image = lr_image.cuda()
                hr_gt_image = hr_gt_image.cuda()
                # y_ref = y_ref.cuda()
                sigma = sigma.cuda()
                b = b_kernel.cuda()

            # generate mixed inputs, targets and mixing coefficient
            #if (opt.is_mixup and random.random() < 0.5):
            #    hr_gt_image, lr_image, sigma, lam = util.utils_common.mixup_data(hr_gt_image, lr_image, sigma, alpha=opt.alpha, use_cuda=opt.cuda)

            hr_gt_image = hr_gt_image.float()
            lr_image = lr_image.float()
            sigma = sigma.float()
            gt_blur_kernel = b.float()

            # print('train x:', x.shape, x.min(), x.max())
            # print('train y:', y.shape, y.min(), y.max())
            # print('train sigma:', sigma.shape, sigma.min(), sigma.max())

            # t = timer()
            # t.tic()
            xhat, est_kernel,gt_kernel, lr_image_kernel,loss, psnr_batch, netG, netD = train(hr_gt_image, lr_image, sigma, gt_blur_kernel,
                                                                         netG, netD, netF,
                                                                         optimizers,
                                                                         cri_pix, l_pix_w,
                                                                         cri_pix_k, cri_pix_gt_k, l_pix_k_w, l_pix_k_gt_w,
                                                                         cri_fea, l_fea_w,
                                                                         cri_tv, l_tv_w,
                                                                         cri_gan, l_gan_w,
                                                                         cri_pix_char, l_pix_w_char,
                                                                         filter_low, filter_high,
                                                                         current_step)
            # est_kernel = kernel_k
            # b = b

            # Accumulate loss and psnr values
            epoch_loss += loss
            epoch_psnr += psnr_batch

            # save train images

            # kernel_k = (kernel_k)*255
            gt = (hr_gt_image*255.0).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
            LR = (lr_image*255.0).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
            est_kernel= (est_kernel*255.0).permute(0,2,3,1).detach().cpu().numpy().astype(np.uint8)
            gt_kernel = (gt_kernel*255.0).permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
            LR_SKI = (lr_image_kernel*255.0).permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
            output = (xhat*255.0).permute(0,2,3,1).detach().cpu().numpy().astype(np.uint8)

            idx = 1

            for j in range(gt.shape[0]):
            #
            #
                if current_step % 10 == 0:

                    save(save_train_imgs_path+'img'+repr(idx)+'_GT'+'.png', gt[j])
                    save(save_train_imgs_path+'img'+repr(idx)+'_LR'+'.png', LR[j])
                    save(save_train_imgs_path + 'img' + repr(idx) + 'est_kernel' + '.png', est_kernel[j])
                    save(save_train_imgs_path + 'img' + repr(idx) + 'gt_Kernel' + '.png', gt_kernel[j])
                    save(save_train_imgs_path + 'img' + repr(idx) + 'LR_SKI' + '.png', LR_SKI[j])
                    save(save_train_imgs_path+'img'+repr(idx)+'_SR'+'.png', output[j])
            #
            #
            #     # psnr and ssim
            #     psnr_SR = util.utils_common.calculate_psnr(output[j], gt[j], border=border)
            #     ssim_SR = util.utils_common.calculate_ssim(output[j], gt[j], border=border)
            #
            #     logger.info('===> train:: {:->4d}--> {:>10s}, SR psnr:{:.2f}dB'.format(idx, save_train_imgs_path+'img'+repr(idx)+'_sr'+'.png', psnr_SR))
            #     logger.info('===> train:: {:->4d}--> {:>10s}, SR ssim:{:.4f}'.format(idx, save_train_imgs_path+'img'+repr(idx)+'_sr'+'.png', ssim_SR))
            #
            #
                # idx += 1

            del hr_gt_image
            del lr_image
            del sigma
            del gt_blur_kernel
            del xhat
            del gt
            del LR
            del output
            del est_kernel
            del gt_kernel
            del LR_SKI
            # del y_ref


           # del LR_SKI
            torch.cuda.empty_cache()
            # break

        #logger.info("train:: Epoch[{}] Complete: Avg. Loss: {:.6f}".format(epoch, epoch_loss / len(trainset_loader)))
        #logger.info("train:: Epoch[{}] Complete: Avg. PSNR: {:.2f} dB".format(epoch, epoch_psnr / len(trainset_loader)))

        # Force empty cache after each epoch is completed
        torch.cuda.empty_cache()

        # current learning rate (lr)
        #logger.info("train:: current lr[{:.8f}]".format(lr_scheduler.get_current_learning_rate(optimizers[0])))
        wandb.log({'g_lr': lr_scheduler.get_current_learning_rate(optimizers[0])})


        if epoch / 1 == 1 or epoch % 20 == 0 :

            # testing
            epoch_psnr_new, epoch_ssim_new, epoch_lpips_new = evaluator.evaluate(netG, testset_loader, lpips, epoch, opt, logger, save_test_imgs_path, border,lpips)

            # saving the best model w.r.t. psnr
            if opt.saveBest:
                if epoch_psnr_new > epoch_psnr_old:
                    epoch_psnr_old = epoch_psnr_new
                    io.save_checkpoint_best(best_psnr_save_path,
                                         epoch, current_step,
                                         optimizers, schedulers,
                                         {'epoch_psnr_old': epoch_psnr_old,
                                          'epoch_lpips_old': epoch_lpips_old},
                                         "psnr", logger)
                    io.save_network(best_psnr_save_path, netG, 'G_best_psnr', epoch)
                    io.save_network(best_psnr_save_path, netD, 'D_best_psnr', epoch)

                # saving the best model w.r.t. lpips
                if epoch_lpips_new < epoch_lpips_old:
                    epoch_lpips_old = epoch_lpips_new
                    io.save_checkpoint_best(best_lpips_save_path,
                                         epoch, current_step,
                                         optimizers, schedulers,
                                         {'epoch_psnr_old': epoch_psnr_old,
                                          'epoch_lpips_old': epoch_lpips_old},
                                         "lpips", logger)
                    io.save_network(best_lpips_save_path, netG, 'G_best_lpips', epoch)
                    io.save_network(best_lpips_save_path, netD, 'D_best_lpips', epoch)

        # save models and training states
        if epoch % opt.save_checkpoint_freq == 0:
            logger.info('Saving models and training states.')
            io.save_network(netG_save_path, netG, 'G', epoch)
            io.save_network(netD_save_path, netD, 'D', epoch)
            io.save_training_state(epoch, current_step, optimizers, schedulers,
                                {'epoch_psnr_old': epoch_psnr_old,
                                 'epoch_lpips_old': epoch_lpips_old},training_states_save_path)
        # break

    # save final network states
    logger.info('===================== Saving the final model =====================')
    io.save_network(netG_save_path, netG, 'final_G', epoch)
    io.save_network(netD_save_path, netD, 'final_D', epoch)
    logger.info('===================== end training =====================')


    logger.info('===================== Training completed in {:.4f} seconds ====================='.format(t.toc()))

    ###############################
    # End
    ###############################


if __name__ == '__main__':
    main()
