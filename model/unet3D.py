import torch
import torch.nn.functional as F
from scipy import ndimage
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import models
import math


# fcn = models.segmentation.fcn_resnet101().eval()




class UnetGenerator_3d(nn.Module):

    def __init__(self, in_dim, out_dim, num_filter):
        super(UnetGenerator_3d, self).__init__()
        self.in_dim = in_dim
        print("indim", self.in_dim)
        self.out_dim = out_dim
        print("outdim", self.out_dim)
        self.num_filter = num_filter
        print("num_filter", self.num_filter)
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        print("\n--------------Initiating U-Net---------\n")



        #上采样初始化
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = maxpool_3d()
        self.bridege1 = conv_block_3d(5, 1, act_fn)

        self.down_2 = conv_block_2_3d(self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2 = maxpool_3d()
        self.bridege2 = conv_block_3d(12, 8, act_fn)

        self.down_3 = conv_block_2_3d(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.pool_3 = maxpool_3d()
        self.bridege3 = conv_block_3d(8, 4, act_fn)
        self.se = SELayer_3D_Avg(12)
        self.bridge = conv_block_2_3d(self.num_filter * 4, self.num_filter * 8, act_fn)


        self.trans_1 = conv_trans_block_3d(self.num_filter * 8, self.num_filter * 8, act_fn)
        self.up_1 = conv_block_2_3d(self.num_filter * 12, self.num_filter * 4, act_fn)
        self.trans_2 = conv_trans_block_3d(self.num_filter * 4, self.num_filter * 4, act_fn)

        self.up_2 = conv_block_2_3d(self.num_filter * 6, self.num_filter * 2, act_fn)
        self.trans_3 = conv_trans_block_3d(self.num_filter * 2, self.num_filter * 2, act_fn)


        self.up_3 = conv_block_2_3d(self.num_filter * 3, self.num_filter * 1, act_fn)

        act_fn = nn.Softmax(dim=1)  # 多分类

        self.out = conv_block_3d(self.num_filter*3, out_dim, act_fn)



    def forward(self, x):

        down_1 = self.down_1(x)   #4*32*64*64
        pool_1 = self.pool_1(down_1)   #4*16*32*32
        down_2 = self.down_2(pool_1)   #8*16*32*32

        pool_2 = self.pool_2(down_2)  #8*8*16*16
        down_3 = self.down_3(pool_2)   #16*8*16*16
        pool_3 = self.pool_3(down_3)  #16*4*8*8

        bridge = self.bridge(pool_3)     #桥接模块  32*4*8*8

        trans_1 = self.trans_1(bridge)  #32*8*16*16
        concat_1 = torch.cat([trans_1, down_3], dim=1)  #48*8*16*16
        up_1 = self.up_1(concat_1)    #16*8*16*16

        trans_2 = self.trans_2(up_1)   #16*16*32*32
        concat_2 = torch.cat([trans_2, down_2], dim=1)  #24*16*32*32
        up_2 = self.up_2(concat_2)  #8*16*32*32

        trans_3 = self.trans_3(up_2)   #8*32*64*64
        concat_3 = torch.cat([trans_3, down_1], dim=1)  #12*32*64*64
        up_3 = self.up_3(concat_3)  #4*32*64*64



        up_6 = F.interpolate(up_2, (32, 64, 64), mode='trilinear', align_corners=True)  # 8 32 64 64
        concat_4 = torch.cat([up_3, up_6], dim=1)  # 60 32 64 64

        se_1 = self.se(concat_4)
        out = self.out(se_1)

        return out,se_1













def conv_block_3d(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(num_groups=1,num_channels=out_dim),
        act_fn,
    )
    return model


def conv_trans_block_3d(in_dim, out_dim, act_fn):        #反卷积
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.GroupNorm(num_groups=1,num_channels=out_dim),
        act_fn,
    )
    return model


def maxpool_3d():
    pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_2_3d(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block_3d(in_dim, out_dim, act_fn),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(num_groups=1,num_channels=out_dim),
    )
    return model




def conv_block_3_3d(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block_3d(in_dim, out_dim, act_fn),
        conv_block_3d(out_dim, out_dim, act_fn),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(num_groups=1,num_channels=out_dim),
    )
    return model

#SE通道注意力
class SELayer_3D_Avg(nn.Module):
    def __init__(self, channel, reduction=4, L=4):
        super(SELayer_3D_Avg, self).__init__()
        self.fbap = nn.AdaptiveAvgPool3d(1)  # 三维自适应pool到指定维度    这里指定为1，实现 三维GAP
        d = max(channel // reduction, L)  # 计算向量Z 的长度d   下限为L
        self.fc = nn.Sequential(
            nn.Linear(channel, d, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channle, _, _, _ = x.size()
        y = self.fbap(x).view(batch, channle)
        y = self.fc(y).view(batch, channle, 1, 1, 1)
        return x * y.expand_as(x)

