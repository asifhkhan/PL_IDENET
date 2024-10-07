# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from models.pydl.nnLayers import init
from models.pydl.nnLayers.functional.functional import Pad2D, Crop2D,\
WeightNormalization, WeightNormalization5D, EdgeTaper, WienerFilter
from models.pydl.utils import formatInput2Tuple, getPad2RetainShape
from math import log10
from basicsr.archs.arch_util import default_init_weights, make_layer, pixel_unshuffle
from models.common_deblur import Conv , Deconv , ResBlock
import models.utils_deblur as utils_deblur
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numbers
from einops import rearrange
from einops.layers.torch import Rearrange
import time
# from torch.nn import functional as F
from torchsummaryX import summary
from torch.cuda import Event
start_event = Event(enable_timing=True)
end_event = Event(enable_timing=True)

class Wmad_estimator(nn.Module):
    """
    Standard deviation estimator using MAD upon wavelet coefficients
    """
    def __init__(self):
        super(Wmad_estimator, self).__init__()

        # DB7 high pass decomposition filter
        #self.db7_decomp_high = torch.Tensor([-0.48296291314469025, 0.836516303737469, -0.22414386804185735,
        #                                     -0.12940952255092145])[:,None]
        self.db7_decomp_high = torch.Tensor([-0.07785205408506236, 0.39653931948230575, -0.7291320908465551,
                                                      0.4697822874053586, 0.14390600392910627, -0.22403618499416572,
                                                      -0.07130921926705004, 0.0806126091510659, 0.03802993693503463,
                                                      -0.01657454163101562,-0.012550998556013784, 0.00042957797300470274,
                                                      0.0018016407039998328,0.0003537138000010399])[:,None]
        self.db7_decomp_high = self.db7_decomp_high[None, None, :]
        #self.db7_decomp_high = torch.stack([self.db7_decomp_high,self.db7_decomp_high,self.db7_decomp_high])


    def forward(self, x):
        if x.max() > 1:
            x  = x/255
        db7_decomp_high = self.db7_decomp_high
        if x.shape[1] > 1:
            db7_decomp_high = torch.cat([self.db7_decomp_high]*x.shape[1], dim=0)

        if x.is_cuda:
            db7_decomp_high = db7_decomp_high.cuda()

        #print('wmad x:', x.shape, x.min(), x.max())
        diagonal = F.pad(x, (0,0,self.db7_decomp_high.shape[2]//2,self.db7_decomp_high.shape[2]//2), mode='reflect')
        diagonal = F.conv2d(diagonal, db7_decomp_high, stride=(2,1), groups=x.shape[1])
        diagonal = F.pad(diagonal, (self.db7_decomp_high.shape[2]//2,self.db7_decomp_high.shape[2]//2,0,0), mode='reflect')
        diagonal = F.conv2d(diagonal.transpose(2,3), db7_decomp_high, stride=(2,1), groups=x.shape[1])
        #diagonal = diagonal.transpose(2,3)
        sigma = 0
        diagonal = diagonal.view(diagonal.shape[0],diagonal.shape[1],-1)
        for c in range(diagonal.shape[1]):
            d = diagonal[:,c]
            sigma += torch.median(torch.abs(d), dim=1)[0] / 0.6745
        sigma = sigma / diagonal.shape[1]
        sigma = sigma.detach()
        del db7_decomp_high
        return sigma


class WienerDeblurNet(nn.Module):

    def __init__(self, input_channels=3, \
                 wiener_kernel_size=(5, 5), \
                 wiener_output_features=24, \
                 numWienerFilters=4, \
                 wienerWeightSharing=True, \
                 wienerChannelSharing=True, \
                 alphaChannelSharing=True, \
                 alpha_update=True, \
                 lb=1e-3, \
                 ub=1e-1, \
                 wiener_pad=True, \
                 wiener_padType='symmetric', \
                 edgeTaper=True, \
                 wiener_scale=True, \
                 wiener_normalizedWeights=True, \
                 wiener_zeroMeanWeights=True,
                 output_features=64,
                 bias_f=True,
                 scale_f=True,
                 normalizedWeights=True, ):

        super(WienerDeblurNet, self).__init__()

        # self.l2proj = l2proj.L2Proj()
        # Initialize the Wiener filters used for deconvolution
        # self.model = model
        self.wiener_pad = wiener_pad
        self.wiener_padType = wiener_padType
        self.edgetaper = edgeTaper
        self.wienerWeightSharing = wienerWeightSharing
        self.wiener_normalizedWeights = wiener_normalizedWeights
        self.wiener_zeroMeanWeights = wiener_zeroMeanWeights
        self.alpha_update = alpha_update
        # self.alpha_res = nn.Parameter(torch.Tensor(np.linspace(np.log(2), np.log(1), 1)))
        self.weights = nn.Parameter(torch.Tensor(1, numWienerFilters, 1, 1, 1).fill_(1 / numWienerFilters))
        # self.bbproj = nn.Hardtanh(min_val=0., max_val=255.)

        assert (numWienerFilters >= 1), "More than one Wiener filter is expected."

        wchannels = 1 if wienerChannelSharing else input_channels

        wiener_kernel_size = formatInput2Tuple(wiener_kernel_size, int, 2)

        if self.wienerWeightSharing:
            shape = (wiener_output_features, wchannels) + wiener_kernel_size
        else:
            shape = (numWienerFilters, wiener_output_features, wchannels) + wiener_kernel_size

        self.wiener_conv_weights = nn.Parameter(torch.Tensor(torch.Size(shape)))
        init.dctMultiWiener(self.wiener_conv_weights)

        if wiener_scale and wiener_normalizedWeights:
            if self.wienerWeightSharing:
                self.wiener_scale = nn.Parameter(torch.Tensor(wiener_output_features).fill_(0.1))
            else:
                self.wiener_scale = nn.Parameter(torch.Tensor(numWienerFilters, wiener_output_features).fill_(0.1))
        else:
            self.register_parameter('wiener_scale', None)

        assert (lb > 0 and ub > 0), "Lower (lb) and upper (ub) bounds of the " \
                                    + "beta parameter must be positive numbers."
        alpha = torch.logspace(log10(lb), log10(ub), numWienerFilters).unsqueeze(-1).log()
        if alphaChannelSharing:
            shape = (numWienerFilters, 1)
        else:
            alpha = alpha.repeat(1, input_channels)
            shape = (numWienerFilters, input_channels)

        if self.alpha_update:
            self.alpha = nn.Parameter(torch.Tensor(torch.Size(shape)))
            self.alpha.data.copy_(alpha)
        else:
            self.alpha = alpha

        if scale_f and normalizedWeights:
            self.scale_f = nn.Parameter(torch.Tensor(output_features).fill_(1))
        else:
            self.register_parameter('scale_f', None)
        #
        #         # Initialize the bias for the conv layer
        if bias_f:
            self.bias_f = nn.Parameter(torch.Tensor(output_features).fill_(0))
        else:
            self.register_parameter('bias_f', None)

    #
    #
    def forward(self, image, blurKernel, stdn):
        # blurKernel = th.randn(1,1,21,21).cuda()
        images = []
        cstdns = []
        for i in range(image.size(0)):
            inputs = image[i]
            blurKernels = blurKernel[i]
            # inputs, blurKernels = input[i], blurKernel[i]
            inputs, blurKernels = inputs.unsqueeze(0), blurKernels.unsqueeze(0)

            # global padding
            blurKernel_size = (blurKernels.size(2), blurKernels.size(3))
            if self.wiener_pad:
                padding = getPad2RetainShape(blurKernel_size)
                input = Pad2D.apply(inputs, padding, self.wiener_padType)

            if self.edgetaper:
                input = EdgeTaper.apply(input, blurKernels.squeeze(0).squeeze(0))  ##1,3,148,148

            if self.wienerWeightSharing:
                wiener_conv_weights = WeightNormalization.apply(self.wiener_conv_weights, \
                                                                self.wiener_scale, self.wiener_normalizedWeights,
                                                                self.wiener_zeroMeanWeights)
            else:
                wiener_conv_weights = WeightNormalization5D.apply(self.wiener_conv_weights, \
                                                                  self.wiener_scale, self.wiener_normalizedWeights,
                                                                  self.wiener_zeroMeanWeights)

            if not self.alpha_update:
                self.alpha = self.alpha.type_as(wiener_conv_weights)
            output, cstdn = WienerFilter.apply(input, blurKernels, wiener_conv_weights, \
                                               self.alpha)
            images.append(output)
            cstdns.append(cstdn)

        input = torch.cat(images, dim=0)
        cstdn = torch.cat(cstdns, dim=0)

        # compute the variance of the remaining colored noise in the output
        # cstdn is of size batch x numWienerFilters
        cstdn = torch.sqrt(stdn.type_as(cstdn).unsqueeze(-1).pow(2).mul(cstdn.mean(dim=2)))

        batch, numWienerFilters = input.shape[0:2]

        cstdn = cstdn.view(-1)  # size: batch*numWienerFilters

        # input has size batch*numWienerFilters x C x H x W
        input = input.view(batch * numWienerFilters, *input.shape[2:])
        # output = self.model(input, cstdn, self.alpha_res)
        # output = input-output
        # cropping layer
        output = Crop2D.apply(input, padding)
        # output = self.bbproj(output)

        output = output.view(batch, numWienerFilters, *output.shape[1:])
        output = output.mul(self.weights).sum(dim=1)

        return output

#
# class Upsample(nn.Sequential):
#     """Upsample module.
#
#     Args:
#         scale (int): Scale factor. Supported scales: 2^n and 3.
#         num_feat (int): Channel number of intermediate features.
#     """
#
#     def __init__(self, scale, num_feat):
#         m = []
#         if (scale & (scale - 1)) == 0:  # scale = 2^n
#             for _ in range(int(math.log(scale, 2))):
#                 m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
#                 m.append(nn.PixelShuffle(2))
#         elif scale == 3:
#             m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
#             m.append(nn.PixelShuffle(3))
#         else:
#             raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
#         super(Upsample, self).__init__(*m)

# class DEBLUR(nn.Module):
#     def __init__(self):
#         super(DEBLUR, self).__init__()
#
#         n_resblock = 3
#         n_feats1 = 96
#         n_feats = 32
#         kernel_size = 5
#         self.n_colors = 3
#
#         FeatureBlock = [Conv(self.n_colors, n_feats1, kernel_size, padding=2, act=True),
#                         ResBlock(Conv, n_feats1, kernel_size, padding=2),
#                         ResBlock(Conv, n_feats1, kernel_size, padding=2),
#                         ResBlock(Conv, n_feats1, kernel_size, padding=2)]
#
#         InBlock1 = [Conv(n_feats1, n_feats, kernel_size, padding=2, act=True),
#                    ResBlock(Conv, n_feats, kernel_size, padding=2),
#                    ResBlock(Conv, n_feats, kernel_size, padding=2),
#                    ResBlock(Conv, n_feats, kernel_size, padding=2)]
#         InBlock2 = [Conv(n_feats1 + n_feats, n_feats, kernel_size, padding=2, act=True),
#                    ResBlock(Conv, n_feats, kernel_size, padding=2),
#                    ResBlock(Conv, n_feats, kernel_size, padding=2),
#                    ResBlock(Conv, n_feats, kernel_size, padding=2)]
#
#         # encoder1
#         Encoder_first= [Conv(n_feats , n_feats*2 , kernel_size , padding = 2 ,stride=2 , act=True),
#                         ResBlock(Conv , n_feats*2 , kernel_size ,padding=2),
#                         ResBlock(Conv , n_feats*2 , kernel_size ,padding=2),
#                         ResBlock(Conv , n_feats*2 , kernel_size ,padding=2)]
#         # encoder2
#         Encoder_second = [Conv(n_feats*2 , n_feats*4 , kernel_size , padding=2 , stride=2 , act=True),
#                           ResBlock(Conv , n_feats*4 , kernel_size , padding=2),
#                           ResBlock(Conv , n_feats*4 , kernel_size , padding=2),
#                           ResBlock(Conv , n_feats*4 , kernel_size , padding=2)]
#         # decoder2
#         Decoder_second = [ResBlock(Conv , n_feats*4 , kernel_size , padding=2) for _ in range(n_resblock)]
#         Decoder_second.append(Deconv(n_feats*4 , n_feats*2 ,kernel_size=3 , padding=1 , output_padding=1 , act=True))
#         # decoder1
#         Decoder_first = [ResBlock(Conv , n_feats*2 , kernel_size , padding=2) for _ in range(n_resblock)]
#         Decoder_first.append(Deconv(n_feats*2 , n_feats , kernel_size=3 , padding=1, output_padding=1 , act=True))
#
#         OutBlock = [ResBlock(Conv , n_feats , kernel_size , padding=2) for _ in range(n_resblock)]
#
#         OutBlock2 = [Conv(n_feats , self.n_colors, kernel_size , padding=2)]
#
#         self.FeatureBlock = nn.Sequential(*FeatureBlock)
#         self.inBlock1 = nn.Sequential(*InBlock1)
#         self.inBlock2 = nn.Sequential(*InBlock2)
#         self.encoder_first = nn.Sequential(*Encoder_first)
#         self.encoder_second = nn.Sequential(*Encoder_second)
#         self.decoder_second = nn.Sequential(*Decoder_second)
#         self.decoder_first = nn.Sequential(*Decoder_first)
#         self.outBlock = nn.Sequential(*OutBlock)
#         self.outBlock2 = nn.Sequential(*OutBlock2)
#
#     def forward(self, input, kernel):
#
#         for jj in range(kernel.shape[0]):
#             kernel[jj:jj+1,:,:,:] = torch.div(kernel[jj:jj+1,:,:,:], torch.sum(kernel[jj:jj+1,:,:,:]))
#         feature_out = self.FeatureBlock(input)
#         clear_features = torch.zeros_like(feature_out)
#         ks = kernel.shape[2]
#         dim = (ks, ks, ks, ks)
#         first_scale_inblock_pad = F.pad(feature_out, dim, "replicate")
#         for i in range(first_scale_inblock_pad.shape[1]):
#             blur_feature_ch = first_scale_inblock_pad[:, i:i + 1, :, :]
#             clear_feature_ch = utils_deblur.get_uperleft_denominator(blur_feature_ch, kernel)
#             clear_features[:, i:i + 1, :, :] = clear_feature_ch[:, :, ks:-ks, ks:-ks]
#         return clear_features

class M_kemb(nn.Module):
    def __init__(self, in_dim=21 * 21, out_dim=128):
        super(M_kemb, self).__init__()

        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.act = nn.ReLU()

    def forward(self, k):
        # Reshape the input tensor to a flat
        k_flat = k.view(k.size(0), -1)

        # Apply the linear transformation
        k_emb = self.linear(k_flat)

        # Apply ReLU activation
        k_emb = self.act(k_emb)

        return k_emb



class KernelEstimator(nn.Module):
    def __init__(self):
        super(KernelEstimator, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),nn.ReLU())

        self.res_block_1 = nn.Sequential(nn.Conv2d(64, 32, 3, 1,1))
        self.max_pool_1=nn.Sequential(nn.MaxPool2d(2,2))
        self.conv_2=nn.Sequential(nn.Conv2d(32, 64, 3, 1,1),nn.ReLU())
        self.res_block_2 = nn.Sequential(nn.Conv2d(64, 32, 3, 1,1))
        self.maxpool_2=nn.Sequential(nn.MaxPool2d(2, 2))
        self.conv_3 =nn.Sequential(nn.Conv2d(32, 64, 3, 1,1),nn.ReLU())
        self.transpose_1 =nn.Sequential(nn.ConvTranspose2d(64, 64, 2,2,0))
        self.conv_4 =nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),nn.ReLU())
        self.res_block_3 =nn.Sequential(nn.Conv2d(64, 32, 3,1,1))
        self.transpose_2 = nn.Sequential(nn.ConvTranspose2d(64, 64, 2,2,0))
        self.conv_5 =nn.Sequential(nn.Conv2d(64, 64, 3, 1,1),nn.ReLU())
        self.res_block_4 =nn.Sequential(nn.Conv2d(64, 32, 3, 1,1))
        self.conv_6 =nn.Sequential(nn.Conv2d(64, 128, 3),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(128, 441, 1), nn.Softmax())



    def forward(self, x):

        b, c, h, w = x.size()

        x = self.conv_1(x)
        x = self.res_block_1(x)
        res1 = x
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.res_block_2(x) #29
        res2 = x
        x = self.maxpool_2(x)
        x = self.conv_3(x)
        x = self.transpose_1(x)
        x = self.conv_4(x)
        x = self.res_block_3(x)
        x = F.interpolate(x, size=(res2.shape[2], res2.shape[3]), mode="bilinear")
        x = torch.cat([x, res2], 1) #concatenation resblock_2 and res_block_3 #14
        x = self.transpose_2(x)
        x = self.conv_5(x)
        x = self.res_block_4(x)
        x = F.interpolate(x, size=(res1.shape[2], res1.shape[3]), mode="bilinear")
        x = torch.cat([x, res1], 1) #concatenation res_block_1 and res_block_4
        x = self.conv_6(x)
        x = x.view(b, 1, 21,21)
        return x


    #################################kernel_prompt_learning_swinir##################

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim=192, num_heads=1, ffn_expansion_factor=2.0, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # x = self.norm1(x)
        # x = self.attn(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
##---------- Prompt Gen Module (PGM)-----------------------
class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=96, prompt_len=128, lin_dim=96):
        super(PromptGenBlock, self).__init__()

        self.prompt_dim = prompt_dim
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x,k_embd):
        B, C, H, W = x.shape  # 1st it = _x96x128x128
        emb = x.mean(dim=(-2, -1))  # 4,96
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)  # 4,256
        # prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * k_embd #4,256,1,4,256
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * k_embd.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.prompt_dim, 1, 1)  # 4,256,32,1,1
        prompt = torch.sum(prompt, dim=1)  # 4,32,1,1
        # prompt = prompt.expand(-1, self.prompt_dim ,-1,-1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")  # 4,32,128,128
        prompt = self.conv3x3(prompt)

        return prompt


##---------- Prompt Interaction Block (PIM) -----------------------

class PromptInteractionBlock(nn.Module):
    def __init__(self, dim=192, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(PromptInteractionBlock, self).__init__()
        self.transformerblock = TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)


    def forward(self, x, prompt):
        out = torch.cat([x, prompt], 1) #1, 128, 128, 128 #1, 160, 128, 128
        out = self.transformerblock(out)
        #out = self.conv_1(out)
        return out


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        # x = to_3d(x)
        # x = self.body(x)
        # x = to_4d(x,h,w)
        return to_4d(self.body(to_3d(x)), h, w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


#########################################################################################################################


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])


        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv',
                 ffn_expansion_factor=1.0,
                 bias=False, LayerNorm_type='WithBias',
                 prompt_scale_factor: int = 1,
                 prompt_num_heads: int = 1):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))
        elif resi_connection == '1conv+prompt':
            self.conv = nn.Conv2d(dim * 2 + (32 * prompt_scale_factor), dim, 3, 1, 1)


        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.pgm = PromptGenBlock(prompt_dim=32 * prompt_scale_factor,
                                  prompt_len=128,
                                  lin_dim=dim) # * prompt_scale_factor
        # self.pgm = PromptGenBlock()

        self.pim = PromptInteractionBlock(dim=dim + (32 * prompt_scale_factor), num_heads=prompt_num_heads,#dim=dim+32, num_heads=prompt_num_heads,
                                          ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                          LayerNorm_type=LayerNorm_type)


    def forward(self, x, x_size, k_embd):  # , x_fea):

        # Go back to image by tiling patches then
        # run PGM and PIM -> prompt features
        x_fea = self.patch_unembed(x, x_size) # OUT = 1, dim, 128, 128
        #x_project = self.conv_2(x_fea) #added to check
        prompt = self.pgm(x_fea, k_embd) # OUT = 1, prompt_dim, 128, 128  #1, prompt_dim, 128, 128
        f_hat = self.pim(x_fea, prompt) # IN = _, dim, 128, 128 | _, prompt_dim, 128, 128

        # Go back from patches to image
        out = self.patch_unembed(self.residual_group(x, x_size), x_size)
        out = torch.cat([f_hat, out], 1)
        # x_fea_out = self.conv_1(out)

        # Go back to patches
        # out = self.conv(out)
        out = self.patch_embed(self.conv(out)) + x

        # OLD/Original RSTB block
        # a = self.patch_unembed(self.residual_group(x, x_size), x_size)
        # self.patch_embed(self.conv(a)) + x

        return out  # , x_fea_out

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class SwinIR(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=32, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[4, 4, 4, 4], num_heads=[4, 4, 4, 4],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=4, img_range=1., upsampler='', resi_connection='1conv',
                 ffn_expansion_factor=1.0,
                 bias=False,
                 LayerNorm_type='WithBias',
                 prompt_num_heads: list = [1, 2, 4, 8],
                 **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)


        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        prompt_scale_factors = [1,2,2,2]
        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection='1conv+prompt',
                         prompt_scale_factor=prompt_scale_factors[i_layer],
                         prompt_num_heads=prompt_num_heads[i_layer]
                         )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.upsample = None
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x,k_embd):
        # x_fea = x # bs, c, 128, 128
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)  # b, d, c
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size,k_embd)
            # x, x_fea = layer(x, x_size, x_fea) opt1

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x, k_embd):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            # x_first = x_clear
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first,k_embd)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        return x[:, :, :H * self.upscale, :W * self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        if self.upsample is not None or self.upsample == '':
            flops += self.upsample.flops()
        return flops


class SRResDNet(nn.Module):
    def __init__(self,weinernet, netT,scale):
        super(SRResDNet, self).__init__()
        self.netT = netT
        self.upscale_factor = scale
        self.bbproj = nn.Hardtanh(min_val=0., max_val=1.)
        self.noise_estimator = Wmad_estimator()
        self.deblur = weinernet
        self.ch_in = 3
        self.num_out_ch = 3
        self.num_feat = 64
        self.k_embd = M_kemb()
        self.bbproj = nn.Hardtanh(min_val=0., max_val=1.)
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(self.ch_in, self.num_feat, 3, 1, 1),
                                                  nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(scale, self.num_feat)
        self.conv_last = nn.Conv2d(self.num_feat, self.num_out_ch, 3, 1, 1)


    def forward(self, input, kernel):
        # reconstruction block
        x = self.conv_before_upsample(input)
        x_up = self.conv_last(self.upsample(x))
        sigma = self.noise_estimator(x)
        sigma *= 255.
        w_output = self.deblur(x_up, kernel, sigma)
        k_embd = self.k_embd(kernel)
        output = self.netT(w_output, k_embd)
        return output

class DEblurSRResDNet(nn.Module):
    def __init__(self, scale):
        super(DEblurSRResDNet, self).__init__()
        self.scale = scale
        # self.deblur  = DEBLUR()
        self.deblur = WienerDeblurNet()
        self.netT = SwinIR()
        self.netDe = SRResDNet( self.deblur, self.netT, self.scale)
        self.netK = KernelEstimator()


    def forward(self, input):

        K = self.netK(input)
        sr = self.netDe(input, K.detach())

        return sr, K

if __name__ == "__main__":
    input = torch.randn(2,3,50,50).type(torch.FloatTensor)
    sf = 4

    model = DEblurSRResDNet(sf)
    print(model)

    s = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Number of model params: %d' % s)

    output = model(input)
    print('output:', output.shape)