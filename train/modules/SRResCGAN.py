import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from pydl.nnLayers import modules
from pydl.nnLayers import init
from pydl.nnLayers.cascades import nconv2D, nconv_transpose2D
from pydl.nnLayers.functional.functional import Pad2D, Crop2D,\
WeightNormalization, WeightNormalization5D, EdgeTaper, WienerFilter
from pydl.utils import formatInput2Tuple, getPad2RetainShape
from math import log10

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, bias=True)


class KernelEstimator(nn.Module):
    def __init__(self):
        super(KernelEstimator, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),nn.ReLU())

        self.res_block_1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1))
        self.max_pool_1=nn.Sequential(nn.MaxPool2d(2,2))
        self.conv_2=nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),nn.ReLU())
        self.res_block_2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1))
        self.maxpool_2=nn.Sequential(nn.MaxPool2d(2, 2))
        self.conv_3 =nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),nn.ReLU())
        self.transpose_1 =nn.Sequential(nn.ConvTranspose2d(64, 64, 3,2,1,1))
        self.conv_4 =nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),nn.ReLU())
        self.res_block_3 =nn.Sequential(nn.Conv2d(64, 64, 3, padding=1))
        self.transpose_2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 3,2,1,1))
        self.conv_5 =nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),nn.ReLU())
        self.res_block_4 =nn.Sequential(nn.Conv2d(64, 64, 3, padding=1))
        self.conv_6 =nn.Sequential(nn.Conv2d(128, 256, 3),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(256, 441, 1), nn.Softmax())



    def forward(self, x):

        b, c, h, w = x.size()

        x = self.conv_1(x)
        # print("conc_1",x.shape)   #64
        x = self.res_block_1(x)
        # print("resblock_1 :", x.shape)    #62
        res1 = x
        x = self.max_pool_1(x)
        # print("maxpool_1", x.shape)
        x = self.conv_2(x)
        # print("conv_2",x.shape)  #62
        x = self.res_block_2(x) #29
        # print("resblock_2 :" ,x.shape)  #60
        res2 = x
        x = self.maxpool_2(x)
        # print("maxpool_2", x.shape)
        x = self.conv_3(x)
        # print("conv_3",x.shape) #58
        x = self.transpose_1(x)
        # print("transpose_1",x.shape) #60
        x = self.conv_4(x)
        # print("conv_4",x.shape) #60
        x = self.res_block_3(x)
        x = F.interpolate(x, size=(res2.shape[2], res2.shape[3]), mode="bilinear")
        # print("resblock_3 :",x.shape) #60
        x = torch.cat([x, res2], 1) #concatenation resblock_2 and res_block_3 #14
        # print("concate_res_2 and res_3",x.shape) #60
        x = self.transpose_2(x)
        # print("transpose_2",x.shape) #62
        x = self.conv_5(x)
        # print("conv_5",x.shape) #62
        x = self.res_block_4(x)
        # print("resblock_4 :", x.shape) #62
        x = F.interpolate(x, size=(res1.shape[2], res1.shape[3]), mode="bilinear")
        x = torch.cat([x, res1], 1) #concatenation res_block_1 and res_block_4
        # print("concate_res_1 and res_4",x.shape) #62
        x = self.conv_6(x)
        # print("conv_6", x.shape)
        x = x.view(b, 1, 21,21)
        # print("est", x.min())
        # print("est", x.max())
        # print(x.shape) #1
        return x


class WienerDeblurNet(nn.Module):

    def __init__(self, input_channels, \
                 wiener_kernel_size=(5, 5), \
                 wiener_output_features=24, \
                 numWienerFilters=3, \
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
        # self.bbproj = nn.Hardtanh(min_val=0., max_val=255.)
        scale = 1
        self.pixel_shuffle = nn.PixelShuffle(scale)

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
        input = torch.cat(images, dim=0)
        # compute the variance of the remaining colored noise in the output
        # cstdn is of size batch x numWienerFilters
        cstdn = torch.sqrt(stdn.type_as(cstdn).unsqueeze(-1).pow(2).mul(cstdn.mean(dim=2)))

        batch, numWienerFilters = input.shape[0:2]

        cstdn = cstdn.view(-1)  # size: batch*numWienerFilters

        # input has size batch*numWienerFilters x C x H x W
        output = input.view(batch * numWienerFilters, *input.shape[2:])
        # cropping layer
        output = Crop2D.apply(output, padding)
        output = output.view(batch, numWienerFilters, *output.shape[1:])
        output = output.sum(dim=1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + ', wiener_kernel_size = ' + str(tuple(self.wiener_conv_weights.shape[-2:])) \
               + ', wiener_output_features = ' + str(self.wiener_conv_weights.size(-4)) \
               + ', wienerWeightSharing = ' + str(self.wienerWeightSharing) \
               + ', edgeTaper = ' + str(self.edgetaper)
class BasicBlock(nn.Module):
    """
    Residual BasicBlock 
    """
    def __init__(self, inplanes, planes, stride=1, weightnorm=None, shortcut=True):
        super(BasicBlock, self).__init__()
        self.shortcut = shortcut
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.relu1 = nn.PReLU(num_parameters=planes,init=0.1)
        self.relu2 = nn.PReLU(num_parameters=planes, init=0.1)
        self.conv2 = conv3x3(inplanes, planes, stride)
        if weightnorm:
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)

    def forward(self, x):
        out = self.relu1(x)
        out = F.pad(out,(1,1,1,1),'reflect')
        out = self.conv1(out)
        out = out[:,:, :x.shape[2], :x.shape[3]]
        out = self.relu2(out)
        out = F.pad(out,(1,1,1,1),'reflect')
        out = self.conv2(out)
        out = out[:,:, :x.shape[2], :x.shape[3]]
        if self.shortcut:
            out = x + out
        return out

class L2Proj(nn.Module):
    """
    L2Proj layer
    source link: https://github.com/cig-skoltech/deep_demosaick/blob/master/l2proj.py
    """
    def __init__(self):
        super(L2Proj, self).__init__()

    def forward(self, x, stdn, alpha):
        if x.is_cuda:
            x_size = torch.cuda.FloatTensor(1).fill_(x.shape[1] * x.shape[2] * x.shape[3])
        else:
            x_size = torch.Tensor([x.shape[1] * x.shape[2] * x.shape[3]])
        numX = torch.sqrt(x_size-1)
        if x.is_cuda:
            epsilon = torch.cuda.FloatTensor(x.shape[0],1,1,1).fill_(1) * (torch.exp(alpha) * stdn * numX)[:,None,None,None]
        else:
            epsilon = torch.zeros(x.size(0),1,1,1).fill_(1) * (torch.exp(alpha) *  stdn * numX)[:,None,None,None]
        x_resized = x.view(x.shape[0], -1)
        x_norm = torch.norm(x_resized, 2, dim=1).reshape(x.size(0),1,1,1)
        max_norm = torch.max(x_norm, epsilon)
        result = x * (epsilon / max_norm)
        
        return result

class Wmad_estimator(nn.Module):
    """
    Standard deviation estimator using MAD upon wavelet coefficients
    source link: https://github.com/cig-skoltech/deep_demosaick/blob/master/modules/wmad_estimator.py
    """
    def __init__(self):
        super(Wmad_estimator, self).__init__()

        # DB7 high pass decomposition filter
        self.db7_decomp_high = torch.Tensor([-0.07785205408506236, 0.39653931948230575, -0.7291320908465551,
                                                      0.4697822874053586, 0.14390600392910627, -0.22403618499416572,
                                                      -0.07130921926705004, 0.0806126091510659, 0.03802993693503463,
                                                      -0.01657454163101562,-0.012550998556013784, 0.00042957797300470274,
                                                      0.0018016407039998328,0.0003537138000010399])[:,None]
        self.db7_decomp_high = self.db7_decomp_high[None, None, :]


    def forward(self, x):
        if x.max() > 1:
            x  = x/255
        db7_decomp_high = self.db7_decomp_high
        if x.shape[1] > 1:
            db7_decomp_high = torch.cat([self.db7_decomp_high]*x.shape[1], dim=0)

        if x.is_cuda:
            db7_decomp_high = db7_decomp_high.cuda()

        diagonal = F.pad(x, (0,0,self.db7_decomp_high.shape[2]//2,self.db7_decomp_high.shape[2]//2), mode='reflect')
        diagonal = F.conv2d(diagonal, db7_decomp_high, stride=(2,1), groups=x.shape[1])
        diagonal = F.pad(diagonal, (self.db7_decomp_high.shape[2]//2,self.db7_decomp_high.shape[2]//2,0,0), mode='reflect')
        diagonal = F.conv2d(diagonal.transpose(2,3), db7_decomp_high, stride=(2,1), groups=x.shape[1])
        
        sigma = 0
        diagonal = diagonal.view(diagonal.shape[0],diagonal.shape[1],-1)
        for c in range(diagonal.shape[1]):
            d = diagonal[:,c]
            sigma += torch.median(torch.abs(d), dim=1)[0] / 0.6745
        sigma = sigma / diagonal.shape[1]
        sigma = sigma.detach()
        del db7_decomp_high
        
        return sigma

class ResCNet(nn.Module):
    """
    Residual Convolutional Net
    """
    def __init__(self, depth=5, color=True, weightnorm=True):
        self.inplanes = 64
        super(ResCNet, self).__init__()
        if color:
            in_channels = 3
        else:
            in_channels = 1

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=0,
                               bias=True)
        if weightnorm:
            self.conv1 = weight_norm(self.conv1)

        # Resnet blocks layer
        self.layer1 = self._make_layer(BasicBlock, 64, depth)
        self.conv_out = nn.ConvTranspose2d(64, in_channels, kernel_size=5, stride=1, padding=2, bias=True)
        if weightnorm:
            self.conv_out = weight_norm(self.conv_out)

        self.l2proj = L2Proj()
        self.zeromean()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride, weightnorm=True, shortcut=False))
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, weightnorm=True, shortcut=True))
        return nn.Sequential(*layers)

    def zeromean(self):
        # Function zeromean subtracts the mean E(f) from filters f
        # in order to create zero mean filters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data - torch.mean(m.weight.data)

    def forward(self, x, stdn, alpha):
        self.zeromean()
        out = F.pad(x,(2,2,2,2),'reflect')
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.conv_out(out)
        out = self.l2proj(out, stdn, alpha)
        return out


class SRResDNet(nn.Module):
    def __init__(self, winernet, model, scale, stdn):
        super(SRResDNet, self).__init__()

        self.deblur = winernet
        self.model = model
        self.upscale_factor = scale
        self.stdn = stdn
        self.noise_estimator = Wmad_estimator()
        self.alpha = nn.Parameter(torch.Tensor(np.linspace(np.log(2), np.log(1), 1)))
        self.bbproj = nn.Hardtanh(min_val=0., max_val=255.)

    def forward(self, input, kernel):
        # reconstruction block

        input = F.interpolate(input, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)

        output = self.deblur(input, kernel, self.stdn)
        sigma = self.noise_estimator(output)
        sigma *= 255.
        #
        output = self.model(output, sigma, self.alpha)
        #
        output = input - output
        #
        output = self.bbproj(output)

        # print('clipping output:', output.shape, output.min(), output.max())

        return output

class DEblurSRResDNet(nn.Module):
    def __init__(self,netG, netK):
        super(DEblurSRResDNet, self).__init__()

        self.netG = netG
        self.netK = netK


    def forward(self, input):


        # reconstruction block


        K = self.netK(input)

        sr = self.netG(input,  K)


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