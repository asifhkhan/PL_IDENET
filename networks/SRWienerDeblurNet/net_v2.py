import torch as th
from torch import nn
from pydl.nnLayers import modules
from pydl.nnLayers import init
from pydl.nnLayers.cascades import nconv2D, nconv_transpose2D
from pydl.nnLayers.functional.functional import Pad2D, Crop2D,\
WeightNormalization, WeightNormalization5D, EdgeTaper, WienerFilter
from pydl.utils import formatInput2Tuple, getPad2RetainShape
from math import log10
from pydl.nnLayers import l2proj

class SRWienerDeblurNet(nn.Module):
    
    def __init__(self, input_channels,\
                 sf = 16,\
                 wiener_kernel_size = (5,5),\
                 wiener_output_features = 24,\
                 numWienerFilters = 3,\
                 wienerWeightSharing = True,\
                 wienerChannelSharing = True,\
                 alphaChannelSharing = True,\
                 alpha_update = True,\
                 lb = 1e-3,\
                 ub = 1e-1,\
                 wiener_pad = True,\
                 wiener_padType = 'symmetric',\
                 edgeTaper = True,\
                 wiener_scale = True,\
                 wiener_normalizedWeights = True,\
                 wiener_zeroMeanWeights = True,\
                 kernel_size = (7,7),\
                 output_features = 64,\
                 convWeightSharing = True,\
                 pad = 'same',\
                 padType = 'symmetric',\
                 conv_init = 'dct',\
                 bias_f= True,\
                 bias_t = True,\
                 scale_f = True,\
                 scale_t = True,\
                 normalizedWeights = True,\
                 zeroMeanWeights = True,\
                 alpha_proj = True,\
                 rpa_depth = 5,\
                 rpa_kernel_size1 = (3,3),\
                 rpa_kernel_size2 = (3,3),\
                 rpa_output_features = 64,\
                 rpa_init = 'msra',\
                 rpa_bias1 = True,\
                 rpa_bias2 = True,\
                 rpa_prelu1_mc = True,\
                 rpa_prelu2_mc = True,\
                 prelu_init = 0.1,\
                 rpa_scale1 = True,\
                 rpa_scale2 = True,\
                 rpa_normalizedWeights = True,\
                 rpa_zeroMeanWeights = True,\
                 shortcut = (True),\
                 clb = 0,\
                 cub = 255):

        super(SRWienerDeblurNet, self).__init__()
        
        self.l2proj = l2proj.L2Proj()
        self.upscale_factor = sf
        # Initialize the Wiener filters used for deconvolution
        self.wiener_pad = wiener_pad
        self.wiener_padType = wiener_padType
        self.edgetaper = edgeTaper
        self.wienerWeightSharing = wienerWeightSharing
        self.wiener_normalizedWeights = wiener_normalizedWeights
        self.wiener_zeroMeanWeights = wiener_zeroMeanWeights
        self.alpha_update = alpha_update
        
        assert(numWienerFilters > 1),"More than one Wiener filter is expected."
        
        wchannels = 1 if wienerChannelSharing else input_channels
        
        wiener_kernel_size = formatInput2Tuple(wiener_kernel_size,int,2)
        
        if self.wienerWeightSharing:
            shape = (wiener_output_features,wchannels)+wiener_kernel_size
        else:
            shape = (numWienerFilters,wiener_output_features,wchannels)+wiener_kernel_size       
        
        self.wiener_conv_weights = nn.Parameter(th.Tensor(th.Size(shape)))
        init.dctMultiWiener(self.wiener_conv_weights)
        
        if wiener_scale and wiener_normalizedWeights:
            if self.wienerWeightSharing:
                self.wiener_scale = nn.Parameter(th.Tensor(wiener_output_features).fill_(0.1))
            else:
                self.wiener_scale = nn.Parameter(th.Tensor(numWienerFilters,wiener_output_features).fill_(0.1))
        else:
            self.register_parameter('wiener_scale',None)
                
        assert(lb > 0 and ub > 0),"Lower (lb) and upper (ub) bounds of the "\
        +"beta parameter must be positive numbers."
        alpha = th.logspace(log10(lb),log10(ub),numWienerFilters).unsqueeze(-1).log()
        if alphaChannelSharing:            
            shape = (numWienerFilters,1)
        else:
            alpha = alpha.repeat(1,input_channels)
            shape = (numWienerFilters,input_channels)
        
        if self.alpha_update:
            self.alpha = nn.Parameter(th.Tensor(th.Size(shape)))
            self.alpha.data.copy_(alpha)
        else:
            self.alpha = alpha
       
        # Initialize the Residual Denoising Network       
        kernel_size = formatInput2Tuple(kernel_size,int,2)
        
        if isinstance(pad,str) and pad == 'same':
            pad = getPad2RetainShape(kernel_size)
#            Kc = th.Tensor(kernel_size).add(1).div(2).floor()
#            pad = (int(Kc[0])-1, kernel_size[0]-int(Kc[0]),\
#                   int(Kc[1])-1,kernel_size[1]-int(Kc[1]))              
        
        self.pad = formatInput2Tuple(pad,int,4)
        self.padType = padType
        self.normalizedWeights = normalizedWeights
        self.zeroMeanWeights = zeroMeanWeights
        self.convWeightSharing = convWeightSharing

        # Initialize conv weights
        shape_input_conv = (output_features,input_channels)+kernel_size
        self.conv_weights = nn.Parameter(th.Tensor(th.Size(shape_input_conv)))
        init.convWeights(self.conv_weights,conv_init)

        # Initialize the scaling coefficients for the conv weight normalization
        if scale_f and normalizedWeights:
            self.scale_f = nn.Parameter(th.Tensor(output_features).fill_(1))
        else:
            self.register_parameter('scale_f', None)       
        
        # Initialize the bias for the conv layer
        if bias_f:
            self.bias_f = nn.Parameter(th.Tensor(output_features).fill_(0))
        else:
            self.register_parameter('bias_f', None)            

        # Initialize tconv weights
        shape_input_tconv = (output_features, input_channels*self.upscale_factor**2)+kernel_size
        
        # Initialize the bias for the transpose conv layer
        if bias_t:
            self.bias_t = nn.Parameter(th.Tensor(input_channels*self.upscale_factor**2).fill_(0))
        else:
            self.register_parameter('bias_t', None)                      

        if not self.convWeightSharing:
            self.convt_weights = nn.Parameter(th.Tensor(th.Size(shape_input_tconv)))
            init.convWeights(self.convt_weights,conv_init)      

            if scale_t and normalizedWeights:
                self.scale_t = nn.Parameter(th.Tensor(output_features).fill_(1))
            else:
                self.register_parameter('scale_t', None)           
        
        numparams_prelu1 = output_features if rpa_prelu1_mc else 1
        numparams_prelu2 = rpa_output_features if rpa_prelu2_mc else 1
        
        self.rpa_depth = rpa_depth
        self.shortcut = formatInput2Tuple(shortcut,bool,rpa_depth,strict = False)
        self.resPA = nn.ModuleList([modules.ResidualPreActivationLayer(\
                        rpa_kernel_size1,rpa_kernel_size2,output_features,\
                        rpa_output_features,rpa_bias1,rpa_bias2,1,1,\
                        numparams_prelu1,numparams_prelu2,prelu_init,padType,\
                        rpa_scale1,rpa_scale2,rpa_normalizedWeights,\
                        rpa_zeroMeanWeights,rpa_init,self.shortcut[i]) \
                        for i in range(self.rpa_depth)]) 
        
        self.bbproj = nn.Hardtanh(min_val = clb, max_val = cub)  
        
        # Initialize the parameter for the L2Proj layer
        if alpha_proj:
            self.alpha_proj = nn.Parameter(th.Tensor(1).fill_(0))
        else:
            self.register_parameter('alpha_proj',None)
        
        # Initialize the parameter for weighting the outputs of each Residual
        # Denoising Network
        self.weights = nn.Parameter(th.Tensor(1,numWienerFilters,1,1,1).fill_(1/numWienerFilters))
        
        # upscaling layer
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)
        
    def forward(self,input,blurKernel,stdn):
        print('input:', input.shape)
        #global padding
        blurKernel_size = (blurKernel.size(2), blurKernel.size(3))
        if self.wiener_pad:
            padding = getPad2RetainShape(blurKernel_size)
            input = Pad2D.apply(input,padding,self.wiener_padType)
            print('wiener pad input:', input.shape)
        
        if self.edgetaper:
            input = EdgeTaper.apply(input,blurKernel.squeeze(0).squeeze(0))
            print('edgetaper input:', input.shape)
        
        if self.wienerWeightSharing:
            wiener_conv_weights = WeightNormalization.apply(self.wiener_conv_weights,\
                self.wiener_scale,self.wiener_normalizedWeights,self.wiener_zeroMeanWeights)
        else:
            wiener_conv_weights = WeightNormalization5D.apply(self.wiener_conv_weights,\
                self.wiener_scale,self.wiener_normalizedWeights,self.wiener_zeroMeanWeights)
        
        if not self.alpha_update:
            self.alpha = self.alpha.type_as(wiener_conv_weights)
        input, cstdn = WienerFilter.apply(input,blurKernel,wiener_conv_weights,\
                                           self.alpha)
        print('wiener_output:', input.shape)
        print('wiener_output_csdtn:', cstdn.shape)
        # compute the variance of the remaining colored noise in the output
        # cstdn is of size batch x numWienerFilters
        cstdn = th.sqrt(stdn.type_as(cstdn).unsqueeze(-1).pow(2).mul(cstdn.mean(dim=2)))            
        
        batch, numWienerFilters = input.shape[0:2]
        
        cstdn = cstdn.view(-1) # size: batch*numWienerFilters
        print('wiener_output_csdtn:', cstdn.shape)
        # input has size batch*numWienerFilters x C x H x W
        input = input.view(batch*numWienerFilters,*input.shape[2:])
        print('wiener_output_reshape:', input.shape)            
        output = nconv2D(input,self.conv_weights,bias=self.bias_f,stride=1,\
                     pad=self.pad,padType=self.padType,dilation=1,\
                     scale=self.scale_f,normalizedWeights=self.normalizedWeights,
                     zeroMeanWeights=self.zeroMeanWeights)
        print('conv_output:', output.shape)
        
        for m in self.resPA:
            output = m(output)
            print('resdnet_output:', output.shape)
            
        if self.convWeightSharing:
            output = nconv_transpose2D(output,self.conv_weights,bias=self.bias_t,\
                     stride=1,pad=self.pad,padType=self.padType,dilation=1,\
                     scale=self.scale_f,normalizedWeights=self.normalizedWeights,
                     zeroMeanWeights=self.zeroMeanWeights)
            print('tconv_output:', output.shape)
        else:
            output = nconv_transpose2D(output,self.convt_weights,bias=self.bias_t,\
                     stride=1,pad=self.pad,padType=self.padType,dilation=1,\
                     scale=self.scale_t,normalizedWeights=self.normalizedWeights,
                     zeroMeanWeights=self.zeroMeanWeights)            
            print('tconv_output:', output.shape)
        
        # projection layer
        output = self.l2proj(output,self.alpha_proj,cstdn)
        
        print('proj_output:', output.shape)
        print('proj_input:', input.shape)
        # clipping layer
        output = self.bbproj(input-output)
        print('clip_output:', output.shape)
        # cropping layer
        output = Crop2D.apply(output,padding)
        print('crop_output:', output.shape)
        
        # upscaling layer
        output = self.pixel_shuffle(output)
        print('upscaling output:', output.shape)
        
        # size of batch x numWienerFilters x C x H x W
        output = output.view(batch,numWienerFilters,*output.shape[1:])
        print('final_output1:', output.shape)
        output = output.mul(self.weights).sum(dim=1)
        print('final_output2:', output.shape)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'input_channels = ' + str(self.conv_weights.size(-3)) \
            + ', wiener_kernel_size = ' + str(tuple(self.wiener_conv_weights.shape[-2:])) \
            + ', wiener_output_features = ' + str(self.wiener_conv_weights.size(-4)) \
            + ', wienerWidth = ' + str(self.weights.size(1)) \
            + ', wienerWeightSharing = ' + str(self.wienerWeightSharing)\
            + ', edgeTaper = ' + str(self.edgetaper)\
            + ', ResDNet_depth = ' + str(self.rpa_depth) \
            + ', convWeightSharing = ' + str(self.convWeightSharing)\
            + ', shortcut = ' + str(self.shortcut) + ')' 

def loadModel(filePath,location='gpu',gpu_device=0):
    """Loads a trained model.
    
    filePath : the path of the file containing the parameters and the architecture
               of the model.
    location : Where to load the model. (Default: cpu)
    """
    if location == 'gpu' and th.cuda.is_available():
        if gpu_device != th.cuda.current_device()\
            and not (gpu_device >= 0 and gpu_device < th.cuda.device_count()):
                gpu_device = th.cuda.current_device()
                
        state = th.load(filePath,map_location=lambda storage,loc:storage.cuda(gpu_device))
    elif location == 'cpu':
        state =  th.load(filePath,map_location=lambda storage,loc:storage)
    else:
        raise Exception("Unknown device to load the model.")
    
    # state['params'] is an Ordered dictionary with the following keys:
    #    odict_keys(['kernel_size', 'input_channels', 'output_features', \
    #   'rbf_mixtures', 'rbf_precision', 'stages', 'pad', 'padType', \
    #  'convWeightSharing', 'scale_f', 'scale_t', 'normalizedWeights', \
    #   'zeroMeanWeights', 'rbf_start', 'rbf_end', 'data_min', 'data_max', \
    #   'data_step', 'alpha', 'clb', 'cub'])
    
    model = SRWienerDeblurNet(*state['params'].values())
    model.load_state_dict(state['model_state_dict'])

    return model

def SRWienerDeblurNet_deblur(y,psf,stdn):
       
    assert(isinstance(stdn,(float,int)) or \
           (th.is_tensor(stdn) and stdn.numel()==y.size(0))),\
           "The second argument must be a tensor or an int or a float."
    
    if not th.is_tensor(stdn):
        stdn = th.Tensor([stdn]).type_as(y)
    
    while y.dim() < 4:
        y = y.unsqueeze(0)
    
    mpath = 'trained_nets/gwdnet_trained/'
    
    batch,channels,H,W = y.shape
    
    if channels == 1:
        mpath += "GWResNet_2.55/model_epoch_best.pth"
    elif channels == 3:
        mpath += "color/model_epoch_best.pth"        
    else: 
        raise ValueError("Input tensor must have either one or three channels.")
    
    model = loadModel(mpath)
    if y.is_cuda:
        model = model.cuda()
        model.eval()
    
    with th.no_grad(): 
        out = model(y,psf,stdn,sf)
    
    return out

if __name__ == '__main__':
    y = th.randn(1,1,128,128)
    psf = th.randn(1,1,11,11)
    stdn = th.tensor([2.55]).type(th.FloatTensor)
    sf = 4
    model = SRWienerDeblurNet(input_channels=1, sf=sf)
    out = model(y,psf,stdn)
    
    print("SRWDNet:", out.shape)
    
#    import numpy as np
#    s = sum([np.prod(list(p.size())) for p in model.parameters()]);
#    print('Number of params: %d' % s)
#    
#    print('out', out.shape)
#    print(model)
    
    