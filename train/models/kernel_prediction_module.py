import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from einops import rearrange



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
        elif isinstance(m, nn.ConvTranspose2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.ConvTranspose2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



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
        # x = F.interpolate(x, size=(res2.shape[2], res2.shape[3]), mode="bilinear")
        x = torch.cat([x, res2], 1) #concatenation resblock_2 and res_block_3 #14
        x = self.transpose_2(x)
        x = self.conv_5(x)
        x = self.res_block_4(x)
        # x = F.interpolate(x, size=(res1.shape[2], res1.shape[3]), mode="bilinear")
        x = torch.cat([x, res1], 1) #concatenation res_block_1 and res_block_4
        x = self.conv_6(x)
        x = x.view(b, 1, 21,21)
        return x






