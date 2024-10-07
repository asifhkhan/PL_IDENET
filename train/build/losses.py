import torch.nn as nn
from modules.loss import GANLoss, TV_L2Loss, TV_L1LOSS
from models import discriminator_vgg_arch as SRGAN_arch
from modules.loss import CharbonnierLoss
from .basic_loss import PerceptualLoss


def build(opt):
    # G pixel loss
    l_pix_type_char = opt.pixel_criterion_char
    if l_pix_type_char == 'charbonnair':
        cri_pix_char = CharbonnierLoss(eps=1e-3)
    # elif l_pix_type == 'l2':
    #     cri_pix = nn.MSELoss()
    else:
        raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type_char))
    l_pix_w_char = opt.pixel_weight_char

    # G pixel loss
    l_pix_type = opt.pixel_criterion
    if l_pix_type == 'l1':
        cri_pix = nn.L1Loss()
    elif l_pix_type == 'l2':
        cri_pix = nn.MSELoss()
    else:
        raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
    l_pix_w = opt.pixel_weight

    # Kernel pixel loss
    l_pix_type_k = opt.pixel_criterion_kernel
    if l_pix_type_k == 'l1':
        cri_pix_k = nn.L1Loss()
    elif l_pix_type_k == 'l2':
        cri_pix_k = nn.MSELoss()
    else:
        raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type_k))
    l_pix_k_w = opt.pixel_kernel_weight

    # Kernel gt pixel loss
    l_pix_type_gt_k = opt.pixel_criterion_GT_kernel
    if l_pix_type_gt_k == 'l1':
        cri_pix_gt_k = nn.L1Loss()
    elif l_pix_type_gt_k == 'l2':
        cri_pix_gt_k = nn.MSELoss()
    else:
        raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type_k))
    l_pix_k_gt_w = opt.pixel_kernel_gt_weight

    # G TV loss
    l_tv_type = opt.tv_criterion
    if l_tv_type == 'l1':
        cri_tv = TV_L1LOSS()
    elif l_tv_type == 'l2':
        cri_tv = TV_L2Loss()
    else:
        raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_tv_type))
    l_tv_w = opt.tv_weight

    # G feature loss
    l_fea_type = opt.feature_criterion
    if l_fea_type == 'l1':
        cri_fea = nn.L1Loss()
    elif l_fea_type == 'l2':
        cri_fea = nn.MSELoss()
    else:
        raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
    l_fea_w = opt.feature_weight

    if cri_fea:  # load VGG perceptual loss
        # PyTorch pretrained VGG19-54, before ReLU.
        # if opt.use_bn:
        #     feature_layer = 49
        # else:
        #     feature_layer = 34
        # netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer,
        #                                       use_bn=opt.use_bn,
        #                                       use_input_norm=True,
        #                                       device='cuda')
        # netF.eval()  # No need to train
        # netF = netF.cuda()

    # GD gan loss

        netF = PerceptualLoss(layer_weights=opt.layer_weights,vgg_type='vgg19',
                     use_input_norm=True,
                     range_norm=False,
                     perceptual_weight=0.5,
                     style_weight=0.,
                     criterion=opt.feature_criterion )
    cri_gan = GANLoss(opt.gan_type, 1.0, 0.0)
    l_gan_w = opt.gan_weight

    if opt.cuda:
        cri_pix = cri_pix.cuda()
        cri_fea = cri_fea.cuda()
        cri_gan = cri_gan.cuda()
        cri_pix_k = cri_pix_k.cuda()
        cri_pix_gt_k = cri_pix_gt_k.cuda()
        cri_pix_char = cri_pix_char.cuda()

    return cri_pix, l_pix_w, \
           cri_pix_k, l_pix_k_w, \
           cri_pix_gt_k, l_pix_k_gt_w, \
           cri_tv, l_tv_w, \
           cri_fea, l_fea_w, \
           cri_gan, l_gan_w, \
           netF, cri_pix_char, l_pix_w_char
