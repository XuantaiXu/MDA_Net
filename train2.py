"""
    修改模型
    focalloss损失函数训练
    多分类做分割模型
"""

# from unet3D import UnetGenerator_3d, loss_function
# from model import unet3D
# from model import qingliangKi_AAM
# from model import KI_ASPP
# from model import KI_ASPP_middle
# from model import Ki_AGs
# from model import Ki_ASPP_chuan
# from model import Res_KI
# from model import qingliangKiUnet
from model import unet3D
from model import discriminator1
from model import discriminator
#from unet3D import
from torch.autograd import Variable
from model.discriminator import FCDiscriminator
from  model import Double_Unet
#from  model import doubleunet
from model import deeplabv3_plus
#from model import SEUnet
# from model import KiUnet
# from model import SE3DUnet1
import numpy as np
import pandas as pd
import openpyxl
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
# from logger import Logger
from torch import optim
import csv
import os
import random
import losses
import time
from util import *
from scipy import ndimage
from torch.autograd import Function
from torch.optim import lr_scheduler
from scipy import ndimage
from scipy.ndimage.interpolation import zoom

# from tensorboardX import SummaryWriter
from scipy.ndimage.interpolation import zoom
from sklearn.cluster import KMeans

# train_path = "/FeaturedData/Public/zhangjianmin/BYLW/train_data/data1"        # 训练集
# dir_checkpoint = "Z:/FeaturedData/Public/zhangjianmin/BYLW/train_data/data1/model"            # change  模型保存路径
train_path = "/mnt/data/xxt/chapter2/3D_unet_GAN/dataset/64_64_32"  # 训练集
# dir_checkpoint = "/mnt/data/zq/zq/3D-Unet/shougong/model_64_64_32_qingliangKi_AAM_toadd" #6.20没测精度
dir_checkpoint = "/mnt/data/xxt/chapter2/3D_unet_GAN/dataset/adam_model_64_64_32_GAN2"

modelPath = "/mnt/data/xxt/chapter2/3D_unet_GAN/dataset/adam_model_64_64_32_GAN/CP55.pth"

lr = [0.01, 0.001]  # 学习率

seed=42

def init_random_seed(manual_seed):
    seed=None
    if manual_seed is None:
        seed=random.randint(1,10000)
    else:
        seed=manual_seed
    print("use random seed : {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

# lr = [0.05, 0.01, 0.001]      # 学习率

def recall(predict, target):  # 召回率
    if t.is_tensor(predict):
        predict = t.sigmoid(predict).data.cpu().numpy()
    if t.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = np.atleast_1d(predict.astype(np.bool))
    target = np.atleast_1d(target.astype(np.bool))

    tp = np.count_nonzero(predict & target)
    fn = np.count_nonzero(~predict & target)
    tn = np.count_nonzero(~predict & ~target)
    fp = np.count_nonzero(predict & ~target)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    # print("recall",recall)
    return recall


def precision(predict, target):  # 精确度
    if t.is_tensor(predict):
        predict = t.sigmoid(predict).data.cpu().numpy()
    if t.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = np.atleast_1d(predict.astype(np.bool))
    target = np.atleast_1d(target.astype(np.bool))

    tp = np.count_nonzero(predict & target)
    fn = np.count_nonzero(~predict & target)
    tn = np.count_nonzero(~predict & ~target)
    fp = np.count_nonzero(predict & ~target)

    pre = tp / float(tp + fp + 1e-10)
    # print("pre",pre)
    return pre


def F1(predict, target):  # F1分数
    # print("predict",predict)
    if t.is_tensor(predict):
        predict = t.sigmoid(predict).data.cpu().numpy()
    if t.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = np.atleast_1d(predict.astype(np.bool))
    # print("predict",predict)
    # np.save('predict128.npy',predict)
    target = np.atleast_1d(target.astype(np.bool))
    # np.save('target128.npy', target)
    # print("target128", target)

    tp = np.count_nonzero(predict & target)
    fn = np.count_nonzero(~predict & target)
    tn = np.count_nonzero(~predict & ~target)
    fp = np.count_nonzero(predict & ~target)

    pre = tp / float(tp + fp + 1e-10)
    recall = tp / float(tp + fn)
    F1 = (2 * pre * recall) / (pre + recall + 1e-10)

    # print("F1",F1)

    return F1

def eightwayASCLoss(probs, size=1):  #实际上是十个方向
    _, _,d, h, w = probs.size()
   # softmax = F.softmax(probs, dim=1)
    p = size
    softmax_pad = F.pad(probs, [p]*6, mode='replicate')
    affinity_group = []
    bot_epsilon = 1e-4
    top_epsilon = 1.0
    #se=SELayer_3D_Avg(20).cuda()

    softmax_pad = torch.clamp(softmax_pad, bot_epsilon, top_epsilon)
    i=0
    for st_z in range(0, 2*size+1, size):
        for st_y in range(0, 2 * size + 1, size):
            for st_x in range(0, 2 * size + 1, size):
                if st_y == size and st_x == size  :
                    if st_z!=size:

                        affinity_paired = t.sum(0.5*(t.log( 2*probs/(softmax_pad[:, :, st_z: st_z+d, st_y:st_y + h, st_x:st_x + w]+
                                probs))* probs+t.log( 2*softmax_pad[:, :, st_z: st_z+d, st_y:st_y + h, st_x:st_x + w]/(softmax_pad[:, :, st_z: st_z+d, st_y:st_y + h, st_x:st_x + w]+ probs))*
                                softmax_pad[:, :, st_z: st_z+d, st_y:st_y + h, st_x:st_x + w]),dim=1)
                        affinity_group.append(affinity_paired.unsqueeze(1))
                elif st_z==size:

                    affinity_paired = t.sum(0.5 * (t.log(2 * probs / (softmax_pad[:, :, st_z: st_z + d, st_y:st_y + h, st_x:st_x + w] +
                                                  probs)) * probs + t.log(
                            2 * softmax_pad[:, :, st_z: st_z + d, st_y:st_y + h, st_x:st_x + w] / (
                                        softmax_pad[:, :, st_z: st_z + d, st_y:st_y + h, st_x:st_x + w] + probs)) *
                               softmax_pad[:, :, st_z: st_z + d, st_y:st_y + h, st_x:st_x + w]),dim=1)
                    affinity_group.append(affinity_paired.unsqueeze(1))
    affinity = torch.cat(affinity_group, dim=1)
  #  print(affinity.shape)
   # affinity = torch.mean(affinity, dim=1)
    affinity = affinity.reshape(1,10,32,64,64)
    #affinity = torch.mean(affinity, dim=1)

   # loss = 1.0 - affinity
   # print(loss.mean())
    return affinity

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1e-5

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss




class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss/C

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)


    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def Loss_ma(data, output):
    """
    :param data: 原始数据
    :param gt: 原始数据对应标签
    :param output: 网络输出
    :return: 流形约束损失
    """
    #data = torch.cat((data[0], data[1]), 0)
    #output = torch.cat((output[0], output[1]), 0)
    data=data.reshape(32*64*64,1)
    #newdata=random.sample(data,2048)
    output=output.reshape(2,32,64,64)
    output=output.permute(1,2,3,0)
    output=output.reshape(32*64*64,2)

    #随机选择16384个点
    # 生成下标
    index = np.arange(32*64*64)
    np.random.shuffle(index)

    # 随机选择2048个元素
    select_index = index[:8192]

    # 从x和y中选择相应的元素
    newdata = data[select_index]
    newoutput = output[select_index]



   # [d1, d2, d3, d4,d5] = data.size()  # 原始数据维度，d1：BATCH_SIZE=16, d2：通道数=103, [d3,d4]:WINDOW_SIZE=[64,64]
   # [_, d6] = output.size()  # 网络输出特征维度，d5：通道数=9

    w = EuclideanDistance(newdata, newdata)  # 计算原始数据每个点之间的欧氏距离
    _, indices = torch.sort(w)  # 把像素点之间的欧氏距离排序
    d = 5  # 取的近邻个数
    y = indices[:, d:]  # 取距离大于d的所有点的位置，用于构建稀疏矩阵
    y1 = indices[:, 0:d]  # 取距离小于d的所有点位置，用于计算sigma
    # 求sigma
    x1 = torch.arange(0, y1.size(0)).expand(y1.t().size()).t()  # 求y1（距离小于7的所有点）对应的横坐标
    sigma = w[x1.long(), y1].mean()  # 求近邻点的均值作为sigma

    x = torch.arange(0, y.size(0)).expand(y.t().size()).t()  # 求y（距离大于7的所有点）对应的横坐标
    w = torch.exp(-w / sigma)  # 计算W矩阵
    w[x.long(), y] = 0  # 把距离大于7的所有点的值归零，构造稀疏矩阵

    # 计算
    z = EuclideanDistance(newoutput, newoutput)  # zz为网络输出结果各像素点之间的欧氏距离，

    # loss = torch.mean(torch.mm(zz, w.t()))  # 先让zz和w的转置两个矩阵相乘，然后返回所有元素的平均值
    aa = torch.sum(torch.mul(z, w), 1)  # 流形正则化约束公式，z矩阵乘w矩阵，再求和
    loss = torch.mean(aa) / d  # 取平均再/近邻个数，避免近邻个数影响loss的值
    # print(loss)

    return loss


def EuclideanDistance(a, b):
    """
    :param a: 输入矩阵A
    :param b: 输入矩阵B
    :return: 两矩阵的欧氏距离
    """
    a = a.view(a.shape[0], -1)
    b = b.view(b.shape[0], -1)
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = ((a - b) ** 2).sum(dim=2)
    return logits


def get_image_block(block_num, img, block_size=[64, 64, 64]):
    a, b, c = img.GetSize()
    x, y, z = int(a / block_size[0]), int(b / block_size[1]), int(c / block_size[2])
    z_num, xy_num = int(block_num / (x * y)), int(block_num % (x * y))
    x_num, y_num = int(xy_num % x), int(xy_num / x)
    img_block = img[x_num * block_size[0]: (x_num + 1) * block_size[0],
                y_num * block_size[1]: (y_num + 1) * block_size[1], z_num * block_size[2]: (z_num + 1) * block_size[2]]
    img_arr = sitk.GetArrayFromImage(img_block)
    img_arr = img_arr.astype(np.uint8)
    img_arr = t.from_numpy(img_arr)
    # 将图像数据转换为张量
    img_arr_1 = t.zeros(1, 1, block_size[0], block_size[1], block_size[2])
    img_arr_1[0, 0, :, :, :] = img_arr

    return img_arr_1


def img2tensor1(img_path, mantage=False):
    img = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img).astype(np.float64)
    # 原图归一化 Z-score标准化方法
    if mantage:
        mean = ndimage.mean(img_arr)
        var = ndimage.variance(img_arr)
        img_arr = (img_arr - mean) / var
    img_arr = t.from_numpy(img_arr)
    # 将图像数据转换为张量
    img_arr_1 = t.zeros(1, 1, 32, 64, 64)  # change 根据数据块的大小调整
    img_arr_1[0, 0, :, :, :] = img_arr
   # Sigmoid=nn.Sigmoid()
    #img_arr_1=Sigmoid(img_arr_1)

    return img_arr_1

def img2tensor2(img_path, mantage=False):
    img = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img).astype(np.float64)
    # 原图归一化 Z-score标准化方法
    if mantage:
        mean = ndimage.mean(img_arr)
        var = ndimage.variance(img_arr)
        img_arr = (img_arr - mean) / var
    img_arr = t.from_numpy(img_arr)
    # 将图像数据转换为张量
    img_arr_1 = t.zeros(1, 1, 32, 64, 64)  # change 根据数据块的大小调整
    img_arr_1[0, 0, :, :, :] = img_arr
   # Sigmoid=nn.Sigmoid()
    #img_arr_1=Sigmoid(img_arr_1)

    return img_arr_1


def ReadImage(imgPath):
    img = sitk.ReadImage(imgPath)
    img_arr = sitk.GetArrayFromImage(img).astype(np.float64)

    return img_arr


def GetLabel(imgPath, N):
    """

    :param imgPath:
    :param N:   标签类别个数
    :return: eg: [1,2,128,128,128]
    """
    img = ReadImage(imgPath)

    img[img == 255] = 1
    # k = np.unique(img)
    # print("k1", k)
    # print(img.size())
    img = t.LongTensor(img)

    # k = np.unique(img)
    # print("k1", k)
    # print("img:", img)
    # img = np.clip(img, 0, 1)
    hot_label = getOneHot(img, N)
    hot_label = hot_label.permute(0, 4, 1, 2, 3)

    return hot_label



def train_net(model, epochs, mantage1,mantage2,mantage3, foreground_label1, background_label1,foreground_label2 ,background_label2,foreground_label3 ,background_label3):
    # 设置模型保存的位置

    #criterion = nn.BCELoss()  # 二进制交叉损失函数
    #global predict_y_s
    criterion = nn.BCELoss()                   # 二进制交叉损失函数
    criterion1 = nn.CrossEntropyLoss(ignore_index=2)

    #criterion = nn.CrossEntropyLoss()
    # criterion = losses.FocalLoss()
    # criterion = losses.CELDice()
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    model_D=discriminator.FCDiscriminator()
    model_D1=discriminator1.FCDiscriminator()
    model_D.cuda()
    model_D1.cuda()

    # optimizer1 = optim.SGD(model.parameters(), lr=lr[0], momentum=0.9, weight_decay=0.0005)
    optimizer1 = optim.Adam(model.parameters(), lr=lr[0], weight_decay=0.0005)
    # optimizer2 = optim.SGD(model.parameters(), lr=lr[1], momentumnewdata=0.9, weight_decay=0.0005)
    optimizer2 = optim.Adam(model_D.parameters(), lr=lr[0], weight_decay=0.0005)
    optimizer3 = optim.Adam(model_D1.parameters(), lr=lr[0], weight_decay=0.0005)
    #optimizer3 = optim.Adam(ConGen.parameters(), lr=lr[1], weight_decay=0.0005)
    # optimizer3 = optim.SGD(model.parameters(), lr=lr[2], momentum=0.9, weight_decay=0.0005)
    #optimizer_D = optim.Adam(model_D.parameters(),  lr=lr[1], weight_decay=0.0005)

    scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=[30], gamma=0.1)
 #
    #
    scheduler2 = lr_scheduler.MultiStepLR(optimizer2, milestones=[30], gamma=0.1)
    scheduler3 = lr_scheduler.MultiStepLR(optimizer3, milestones=[30], gamma=0.1)

    train_loss = []
    val_loss = []

    loss_summury = np.zeros((epochs, 2))
    for epoch in range(epochs):
        epoch_loss = 0
        #aepoch_loss = 0
        #depoch_loss = 0
        img_dir1 = os.listdir(mantage1)
        #img_dir2 = os.listdir(mantage2)
        # print("img_dir",img_dir)
        # random.shuffle(img_dir)
        # print("random.shuffle(img_dir)",img_dir)
        img_num = len(img_dir1)
        #img_num2 = len(img_dir2)
        loss_batch_D = 0
        loss_batch_G=0
        loss_batch=0
        #loss_batch_adv = 0
        #aloss_batch = 0
        #dloss_batch=0
        gt1 = []
        gt2=[]
        predict_y_np_s = []
        predict_y_np_s1 = []
        predict_y_np_t = []


        #soure和target命名要一致
        for i, name in enumerate(img_dir1):
            if(epoch==0):
                temp=np.full((32, 4096), 2)
                data1 = pd.DataFrame(temp)
                data1.to_csv('/mnt/data/xxt/chapter2/3D_unet_GAN/dataset/64_64_32/target/clu_label_0.01/弱标签5t' + str(i + 1) + '.csv')
            
            if epoch>=50:
                #计算距离图
                with open(str('/mnt/data/xxt/chapter2/3D_unet_GAN/dataset/64_64_32/target/clu_label_0.01/弱标签5t') + str(i + 1) + '.csv') as file_name:
                    file_read = csv.reader(file_name)
                    array1 = list(file_read)
                array1 = np.array(array1)
                    # b = np.zeros(256)
                wt = array1[1:, 1:]
                wt = wt.reshape(1,131072).astype(np.float64)
                wt = t.LongTensor(wt).cuda(0)

          #  print(name)
            p = float(i + epoch * 128) / 1500 / 128
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            mantage_path1 = mantage1 + '/' + name
            foreground_label_path1 = foreground_label1 + '/' + name
            mantage_path2 = mantage2 + '/' + name
            foreground_label_path2 = foreground_label2 + '/' + name

        #######
            # 读取图像和label  source
            mantage_img_block1 = img2tensor2(mantage_path1, mantage=True).cuda()
        
            # print("xingzhuang",mantage_img_block.shape)
            true_y = GetLabel(foreground_label_path1, 2)
            true_y_qu1 = t.squeeze(true_y)
            true_y_suoyin = t.argmax(true_y_qu1, dim=0)
            gt1 = np.append(gt1, true_y_suoyin)
            true_y = true_y.cuda()




          ########
            # 读取图像和label  target
            mantage_img_block2 = img2tensor1(mantage_path2, mantage=True).cuda()
            # print("xingzhuang",mantage_img_block.shape)
            true_y_2 = GetLabel(foreground_label_path2, 2)
            true_y_qu2 = t.squeeze(true_y_2)
            true_y_suoyin2 = t.argmax(true_y_qu2, dim=0)
            gt2 = np.append(gt2, true_y_suoyin2)
            true_y_2 = true_y_2.cuda()


              # 伪标签的选择



          #source predict
        # 距离图的拼接
            '''
            with open(str('距离s') + str(i + 1) + '.csv') as file_name:
                file_read = csv.reader(file_name)
                array1 = list(file_read)
            array1 = np.array(array1)
            # b = np.zeros(256)
            b = array1[1:, 1:]
            b = b.reshape(1,1,32, 64, 64).astype(np.float64)
            b = t.Tensor(b).cuda(3)
            mantage_img_block1 = t.cat((mantage_img_block1,b),dim=1)
            '''



          #  print(mantage_img_block2.shape)

            predict_y_s,adv_s= model(mantage_img_block1)
            A_s = eightwayASCLoss(predict_y_s)
           # I_s = -predict_y_s*t.log(predict_y_s)
           # w_s = t.mean(I_s,dim=1)
           # predict_y_s_2=(1-w_s)*predict_y_s

          #  A_s =w_s*A_s
            predict_y_s_d=ReverseLayerF.apply(adv_s, alpha)
          #  A_s_d=ReverseLayerF.apply(A_s, alpha)
            predict_qu1_s = t.squeeze(predict_y_s)
            pre_suoyin_s = t.argmax(predict_qu1_s, dim=0)
            pre_suoyin_s = pre_suoyin_s.data.cpu()
            predict_y_np_s = np.append(predict_y_np_s, pre_suoyin_s)

            '''
            predict_y_s_d1=ReverseLayerF.apply(adv_s1, alpha)
            predict_qu1_s1 = t.squeeze(predict_y_s1)
            pre_suoyin_s1 = t.argmax(predict_qu1_s1, dim=0)
            pre_suoyin_s1 = pre_suoyin_s1.data.cpu()
            predict_y_np_s1 = np.append(predict_y_np_s1, pre_suoyin_s1)

            #adv_s3=(adv_s+adv_s1)/2
            '''

            '''
            with open(str('距离t') + str(i + 1) + '.csv') as file_name:
                file_read = csv.reader(file_name)
                array1 = list(file_read)
            array1 = np.array(array1)
            # b = np.zeros(256)
            b = array1[1:, 1:]
            b = b.reshape(1, 1, 32, 64, 64).astype(np.float64)
            b = t.Tensor(b).cuda(3)
            mantage_img_block2 = t.cat((mantage_img_block2, b), dim=1)
            '''


            #target predict
            predict_y_t,adv_t = model(mantage_img_block2)
           # I_t = -predict_y_t * t.log(predict_y_t)
           # w_t = t.mean(I_t, dim=1)
           # predict_y_t_2 = (1 - w_t) * predict_y_t
            A_t = eightwayASCLoss(predict_y_t)
          #  I_t = -predict_y_t * t.log(predict_y_t)
          #  w_t = t.mean(I_t, dim=1)
           # A_t = A_t*w_t
            #print(A_t.shape)
            predict_y_t_d = ReverseLayerF.apply(adv_t, alpha)
         #   A_t_d = ReverseLayerF.apply(A_t, alpha)

         #   predict_y_t_d1=ReverseLayerF.apply(adv_t1, alpha)

         #   adv_t3=(adv_t+adv_t1)/2
            
            if epoch>=50 and epoch<55:
                
                #将目标域的特征图进行聚类
                clu_fea=adv_t
                clu_fea=clu_fea.reshape(12,32,64,64)
                clu_fea_temp=clu_fea.permute(1,2,3,0)
                clu_fea_temp=clu_fea_temp.reshape(32*64*64,12)
                clu_fea_temp=clu_fea_temp.data.cpu().data.numpy()
                
                classifier = KMeans(n_clusters=2, max_iter=100,
                        init='k-means++')  # 建立分类器
                kmeans = classifier.fit(clu_fea_temp)  # 进行分类
                labels = kmeans.labels_
                labels=labels.reshape(1,64*64*32)

                _,index_0=np.where(labels==0)
                #print(index_0.shape)
                _,index_1=np.where(labels==1)
                #print(index_1.shape)
                
                clu_fea=clu_fea.reshape(12,32*64*64)
                select_0=clu_fea[:,index_0]
               # print(select_0.shape)
                select_1=clu_fea[:,index_1]
                #print(select_1.shape)
                
                fea_0=t.mean(select_0, dim=1)
                fea_0=fea_0.reshape(12,1)
               # print(fea_0.shape)
                fea_1=t.mean(select_1, dim=1)
                fea_1=fea_1.reshape(12,1)
                #print(fea_1.shape)
                
                labels=labels.astype(np.float64)
                labels = t.LongTensor(labels).cuda(3)
                                # 对于 label 为 0 的位置，计算 a - b 的平均
                mask_0 = (labels == 0).float()  # 创建掩码，将 label 为 0 的位置标记为 1，其余为 0
               
                #count_0=t.sum(mask_0, dim=1).data.cpu().data.numpy()
                
                diff_0 = (clu_fea - fea_0) * mask_0  # 应用掩码，只保留 label 为 0 的位置的计算结果
                mean_0 = t.norm(diff_0, p=2,dim=0)   # 沿着第一个维度（即 12 那个维度）求和
                mean_0 = mean_0.reshape(1,131072)
                
                
                # 对于 label 为 1 的位置，计算 a - c 的平均
                mask_1 = (labels == 1).float()  # 创建掩码，将 label 为 1 的位置标记为 1，其余为 0
                
                diff_1 = (clu_fea - fea_1) * mask_1  # 应用掩码，只保留 label 为 1 的位置的计算结果
                #count_1=t.sum(mask_1, dim=1).data.cpu().data.numpy()
        
                mean_1 = t.norm(diff_1, p=2,dim=0)   # 沿着第一个维度求和
             
                # 将 mean_0 和 mean_1 组合成最终的结果数组 d
                # 这里假设 label 中只包含 0 和 1，且每个位置只有一个标签
                dis = mean_0 + mean_1
                dis=dis.reshape(1,131072)
                
                temp_index=t.where(wt!=2)
                #print()
                target_true=true_y_suoyin2
                target_true=target_true.reshape(1,131072).cuda()
                #计算1%像素点的个数/5
                
                top_k_0=int(66/5)
                dis_0=dis
                dis_0[temp_index]=100000
                dis_0[0,index_1]=100000
                neg_dis_0=-dis_0
                _,top_indices=t.topk(neg_dis_0,top_k_0,largest=True, sorted=True)
                
                wt[0,top_indices]=target_true[0,top_indices]
                #print(top_indices)
                
                top_k_1=int(66/5)
                dis_1=dis
                dis_1[temp_index]=100000
                dis_1[0,index_0]=100000
                neg_dis_1=-dis_1
                _,top_indices=t.topk(neg_dis_1,top_k_1,largest=True, sorted=True)
                
                wt[0,top_indices]=target_true[0,top_indices]
                
                
                wt2=wt
                
                wt2 = wt2.reshape(32, 4096).data.cpu().data.numpy()
                data1 = pd.DataFrame(wt2)
                data1.to_csv('/mnt/data/xxt/chapter2/3D_unet_GAN/dataset/64_64_32/target/clu_label_0.01/弱标签4t' + str(i + 1) + '.csv')
            
            if epoch >=50:
                wt = wt.reshape(1,32, 64, 64)
            
            
            











            for param in model_D1.parameters():
                param.requires_grad = False


            source_D=t.ones( 1 , 1 , 1 , 2 , 2 ).detach().cuda(0)   #源域判别器标签 1
            target_D = t.zeros( 1 , 1, 1, 2 , 2 ).detach().cuda(0)  #目标域判别器标签 0
            #print(source_D)

            D_out_s = model_D(predict_y_s_d)
            D_out_as = model_D1(A_s)

            D_out_t = model_D(predict_y_t_d)
            D_out_at = model_D1(A_t)

            if epoch<50:
                loss_1 = criterion(predict_y_s, true_y) #+criterion1(predict_y_t, wt)
            else:
                loss_1 = criterion(predict_y_s, true_y) +criterion1(predict_y_t, wt) 
            
           # loss_3 =abs( 0.5*(t.mean(abs(adv_s_ture-adv_s))+t.mean(abs(adv_t_ture-adv_t)))-t.sum(abs(mean_1-mean_0)))
            loss = loss_1#+0.01*eightwayASCLoss(predict_y_t)
            loss_batch_G += loss
            D_loss1 = criterion(D_out_s, source_D)+ criterion(D_out_t, target_D)
            loss_adv =  criterion(D_out_at, source_D)
            D_loss =D_loss1+0.05*loss_adv
            loss_batch+=loss+D_loss

            epoch_loss += loss.data


           # D_out_t_1 = model_D(adv_t)
            #print(D_out_t.shape)
           # D_out_t = model_D(adv_t)


            if i % 8 == 0:
                optimizer1.zero_grad()  # 手动将梯度缓存区设置为0，因为梯度是反向传播中的说明是累加
                optimizer2.zero_grad()  # 手动将梯度缓存区设置为0，因为梯度是反向传播中的说明是累加的
                #optimizer3.zero_grad()  # 手动将梯度缓存区设置为0，因为梯度是反向传播中的说明是累加的
                loss_batch.backward()
                optimizer1.step()
                optimizer2.step()
              #  optimizer3.step()
                # optimizer2.step()
                print('======> Epoch_train: {}, iter_train: {}/{}, loss_G: {:.10f}'.format(epoch, i, img_num,
                                                                                           loss.data ))
                print('======> Epoch_train: {}, iter_train: {}/{}, loss_DF: {:.10f}'.format(epoch, i, img_num,
                                                                                      D_loss1.data))
                loss_batch_G = 0
               # loss_batch_D=0  #减少ASA对模型的影响
                loss_batch=0

            for param in model_D1.parameters():
                param.requires_grad = True
            A_s=A_s.detach()
            A_t=A_t.detach()
            D_out_as = model_D1(A_s)
            D_out_at = model_D1(A_t)
            D_loss2 = criterion(D_out_as, source_D)+ criterion(D_out_at, target_D)
            loss_batch_D += D_loss2

            if i % 8 == 0:
                optimizer3.zero_grad()  # 手动将梯度缓存区设置为0，因为梯度是反向传播中的说明是累加的
                loss_batch_D.backward()

                optimizer3.step()
                # optimizer2.step()
                print('======> Epoch_train: {}, iter_train: {}/{}, loss_DA: {:.10f}'.format(epoch, i, img_num,
                                                                                      D_loss2.data))
                loss_batch_D=0









            #loss_batch+=loss_batch_G+loss_batch_D

           # loss_batch+=loss_batch_G+loss_batch_D








            #D_loss.backward(retain_graph=True)


           # epoch_loss += D_loss.data

            '''
            #更新
            if i % 8 == 0:
                optimizer2.zero_grad()  # 手动将梯度缓存区设置为0，因为梯度是反向传播中的说明是累加的
                loss_batch.backward()
                optimizer2.step()
                print('======> Epoch_train: {}, iter_train: {}/{}, loss_D: {:.10f}'.format(epoch, i, img_num,
                                                                                           D_loss.data ))
                loss_batch_D = 0
            '''

        scheduler1.step()
        scheduler2.step()
        scheduler3.step()


        #源域 指标
        pret_s = precision(predict_y_np_s, gt1)
        rect_s = recall(predict_y_np_s, gt1)
        f1t_s = F1(predict_y_np_s, gt1)






        epoch_loss /= img_num  # 平均loss
        train_loss.append(epoch_loss)
        loss_summury[epoch][0] = epoch_loss * 10000





        ## 验证
        gt = []
        val_predict_y_np = []
        val_predict_y_np1 = []
        val_list = os.listdir(mantage3)
      #  random.shuffle(val_list)
        val_num = len(val_list)
        val_epoch_loss = 0
        for i, name in enumerate(val_list):
            print('============val image name{}:============'.format(name))
           


            val_path = mantage3 + '/' + name
            val_foreground_label_path = foreground_label3 + '/' + name

            val_img_block = img2tensor1(val_path, mantage=True).cuda()

            val_true_y = GetLabel(val_foreground_label_path, 2)

            val_true_y_qu1 = t.squeeze(val_true_y)
            val_true_y_suoyin = t.argmax(val_true_y_qu1, dim=0)
            gt = np.append(gt, val_true_y_suoyin)

            val_true_y = val_true_y.cuda()
            '''
            with open(str('距离t') + str(i + 1) + '.csv') as file_name:
                file_read = csv.reader(file_name)
                array1 = list(file_read)
            array1 = np.array(array1)
            # b = np.zeros(256)
            b = array1[1:, 1:]
            b = b.reshape(1, 1, 32, 64, 64).astype(np.float64)
            b = t.Tensor(b).cuda(3)
            val_img_block = t.cat((val_img_block, b), dim=1)
            '''



            val_predict_y,_= model(val_img_block)

            val_predict_qu1 = t.squeeze(val_predict_y)
            val_pre_suoyin = t.argmax(val_predict_qu1, dim=0)
            val_pre_suoyin = val_pre_suoyin.data.cpu()
            val_predict_y_np = np.append(val_predict_y_np, val_pre_suoyin)
            '''
            val_predict_qu11 = t.squeeze(val_predict_y1)
            val_pre_suoyin1 = t.argmax(val_predict_qu11, dim=0)
            val_pre_suoyin1 = val_pre_suoyin1.data.cpu()
            val_predict_y_np1 = np.append(val_predict_y_np1, val_pre_suoyin1)
            '''

            # loss = F.cross_entropy(val_predict_y, val_true_y)
            loss = criterion(val_predict_y, val_true_y)
            val_epoch_loss += loss.data
            print('======> Epoch_train: {}, iter_val: {}/{}, loss_train: {:.10f}'.format(epoch, i, img_num,
                                                                                         loss.data * 10000))

        pre_v = precision(val_predict_y_np, gt)
        rec_v = recall(val_predict_y_np, gt)
        f1_v = F1(val_predict_y_np, gt)
        val_epoch_loss /= val_num
        print('val loss:', val_epoch_loss.data)
        val_loss.append(val_epoch_loss)
        loss_summury[epoch][1] = val_epoch_loss * 10000

        s = '======> Epoch_train: {}, iter_train: {}/{}, loss_train: {:.10f}, pre: {:.10f}, recall: {:.10f}, F1: {:.10f}'.format(
            epoch, epoch, epochs,
            epoch_loss * 10000, pret_s, rect_s, f1t_s)
        f = open(r'train_processing\646432_unet3D_source.txt', 'a')
        print(s, file=f)
        f.close()

        stistic_data = pd.DataFrame(loss_summury)
        writer = pd.ExcelWriter(r'losses\model_193742_1128.xls')  # 写入Excel文件
        stistic_data.to_excel(writer, float_format='%.3f')  # ‘page_1’是写入excel的sheet名
        writer.save()
        writer.close()






        s = '======> Epoch_train: {}, iter_train: {}/{}, loss_train: {:.10f}, pre: {:.10f}, recall: {:.10f}, F1: {:.10f}'.format(
            epoch, epoch, epochs,
            val_epoch_loss * 10000, pre_v, rec_v, f1_v)
        f = open(r'train_processing\646432_unet3D_val2.txt', 'a')
        print(s, file=f)
        f.close()



        t.save(model.state_dict(), dir_checkpoint + '/' + 'CP{}.pth'.format(epoch))  # change
        print('Checkpoint {} saved !'.format(epoch))

    return train_loss, val_loss


if __name__ == '__main__':
    #  源域
    mantage_path1 = train_path + '/' + 'source' + '/' + 'mont'
    foreground_label_path1 = train_path + '/' + 'source' + '/' + 'label'
    background_label_path1 = train_path + '/' + 'source' + '/' + 'back128'
    #  目标域
    mantage_path2 = train_path + '/' + 'target' + '/' + 'mont'
    foreground_label_path2 = train_path + '/' + 'target' + '/' + 'label'
    background_label_path2 = train_path + '/' + 'target' + '/' + 'back128'

    #验证集adam_model_64_64_32_GAN2adam_model_64_64_32_GAN2
    mantage_path3 = train_path + '/' + 'target' + '/' + 'mont'
    foreground_label_path3 = train_path + '/' + 'target' + '/' + 'label'
    background_label_path3 = train_path + '/' + 'target' + '/' + 'back128'

    seed=init_random_seed(seed)

    t.cuda.set_device(0)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    # print(torch.cuda.get_device_name(0))
    # model = KiUnet.KiUNet_min(in_dim=1, out_dim=2, num_filter=4).cuda()
    # model = ResUnet.ResUnet(in_dim=1, out_dim=2, num_filter=4).cuda()
    # model = SE3DUnet1.SE3DUnet1(in_dim=1, out_dim=2, num_filter=4).cuda()
    model = unet3D.UnetGenerator_3d(in_dim=1, out_dim=2, num_filter=4).cuda()
    #model = doubleunet.doubleunet_3d(in_dim=1, out_dim=2, num_filter=4).cuda()
    #model = deeplabv3_plus.DeepLab().cuda()
    # model = qingliangKi_AAM.KiUnet_AAM(in_dim=1, out_dim=2, num_filter=4).cuda()
    # model = KI_ASPP_middle.Ki_ASPP(in_dim=1, out_dim=2, num_filter=4).cuda()np.save(f'confusion_matrix/DCFSL+ME+ST+CAM-INP-{iDataSet}.npy', C)
    # model = Ki_AGs.KiUnet_3D(in_dim=1, out_dim=2, num_filter=4).cuda()
    # model = Ki_ASPP_chuan.KiUnet_3D(in_dim=1, out_dim=2, num_filter=4).cuda()
    # model = KI_ASPP.Ki_ASPP(in_dim=1, out_dim=2, num_filter=4).cuda()
    # model = KI_ASPP.Ki_ASPP(in_dim=1, out_dim=2, num_filter=4).cuda()
    # model = qingliangKiUnet.KiUnet_3D(in_dim=1, out_dim=2, num_filter=4).cuda()
    # load 预训练好的模型
    # model.load_state_dict(t.load(r'G:\ZJM\project\3D_unet_pytorch\3D_unet_pytorch\checkpoint\model_190420\CP50.pth', map_location=lambda storage, loc: storage))      # change
   # model.load_state_dict(t.load(modelPath, map_location=lambda storage, loc: storage))  # change

    # 训练模型
    train_loss, val_loss = train_net(model, epochs=1500, mantage1=mantage_path1,mantage2=mantage_path2,mantage3=mantage_path3, foreground_label1=foreground_label_path1,
                                     background_label1=background_label_path1, foreground_label2=foreground_label_path2,background_label2=background_label_path2,foreground_label3=foreground_label_path3,background_label3=background_label_path3,
                                     )




