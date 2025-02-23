"""
    适用于带粗亮纤维的胞体识别
    作者：张建敏
    日期：2021.03.18
"""

import torch as t
import math
from torch import nn

from model import unet3D
import csv

import numpy as np
import SimpleITK as sitk
import os
import time

import torch.nn.functional as F
from torch.autograd import Variable
from util import *
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt

imgPath = "/mnt/data/xxt/3D_unet_GAN/dataset/64_64_32/newdata/mont"     # 需要预测的数据块路径
outputPath = "/mnt/data/xxt/3D_unet_GAN/dataset"      # 结果保存路径

modelPath = "/mnt/data/xxt/3D_unet_GAN/dataset/adam_model_64_64_32_GAN/CP447.pth"      # 训练好的模型
block_size = [64, 64, 32]                       # 不需要修改                数值：x,y,z 实际输入网络的数据块大小为【128，128，128】
z_pixel = 1                                       # change 图像z向分辨率
CUDA_ID = 1                                       # 可以选择GPU

def mkdir_file():
    # 创建predict的子文件夹
    predict_img = outputPath + '/' + 'image'
    if not os.path.exists(predict_img):
        os.makedirs(predict_img)
    predict_swc = outputPath + '/' + 'swc'
    if not os.path.exists(predict_swc):
        os.makedirs(predict_swc)
        print('创建文件夹完成')

def get_image_block(block_num, img, block_size=[128, 128, 128]):
    a, b, c = img.GetSize()
    x, y, z = int(a/block_size[0]), int(b/block_size[1]), int(c/block_size[2])
    z_num, xy_num = int(block_num/(x*y)), int(block_num%(x*y))
    x_num, y_num = int(xy_num%x), int(xy_num/x)
    img_block = img[x_num*block_size[0]: (x_num+1)*block_size[0], y_num*block_size[1]: (y_num+1)*block_size[1], z_num*block_size[2]: (z_num+1)*block_size[2]]
    img_arr = sitk.GetArrayFromImage(img_block)
    img_arr = img_arr.astype(np.uint8)
    img_arr = t.from_numpy(img_arr)
    print(img_arr.shape)
    # 将图像数据转换为张量
    img_arr_1 = t.zeros(1, 1, block_size[0], block_size[1], block_size[2])
    print(img_arr_1.shape)
    img_arr_1[0, 0, :, :, :] = img_arr

    return img_arr_1

class test(object):
    def __init__(self, model, img,h):
        self.model = model
        self.img = img

        self.zPixel, self.xPixel, self.yPixel = img.shape          # a, b, c
        self.pred = np.zeros((self.zPixel, self.xPixel, self.yPixel))  # 网络输出图像
        self.model = model
        self.h=h

    def eval(self):
        if self.zPixel == 128:
            num = 1
        else:
            num = self.numOfSegment()        # 分割数据块个数

        for id in range(num):
            h=self.h
            img, region = self.regionOfSegment(id)
            '''
            with open(str('距离t') + str(h) + '.csv') as file_name:
                file_read = csv.reader(file_name)
                array1 = list(file_read)
            array1 = np.array(array1)
            # b = np.zeros(256)
            b = array1[1:, 1:]
            b = b.reshape(1, 1, 32, 64, 64).astype(np.float64)
            b = t.Tensor(b).cuda(2)
            img = t.cat((img, b), dim=1)
            '''

            output,fea= self.model(img)

            '''
            output1 = (output + output) / 2
            output2 = (output + output) / 2
            output3 = (output + output) / 2
            output4 = (output + output) / 2
            # print( output1[:,1,:,:,:])
            #  output2 = (output1 + output1) / 2
            label_t_1 = output1[:, 1, :, :, :]  # 第一个通道下就是胞体通道下的预测结果
            # print(label_t_1)
            # label_t_1=label_t_1.reshape()
            # label_t_1[(label_t_1 < 0.5) | (label_t_1 >= 0.9)] = 0
            # label_t_1[(label_t_1 >= 0.5) & (label_t_1 < 0.9)] = 1  # 将预测结果大于0.5小于0.9的特征保存下来，去掉其余的预测结果
            label_t_1[label_t_1 >= 0.5] = 1  # 将预测结果大于0.9的特征保存下来，去掉其余的预测结果
            label_t_1[label_t_1 < 0.5] = 0
            fea_1 = label_t_1 * fea  # 1 12 32 64 64
            fea_1 = fea_1.reshape(12, 32 * 64 * 64)
            mean_1 = t.mean(fea_1, dim=1)  # 12 1    得到平均胞体的特征值
            mean_1 = mean_1.reshape(1, 12, 1, 1, 1)

            label_t1_1 = output2[:, 1, :, :, :]  # 第一个通道下就是胞体通道下的预测结果
            label_t1_1[(label_t1_1 < 0.8)] = 0
            label_t1_1[(label_t1_1 >= 0.8)] = 1  # 将预测结果大于0.5小于0.9的特征保存下来，去掉其余的预测结果

            fea1_1 = label_t1_1 * fea  # 1 12 32 64 64  #胞体预测结果大于0.5小于0.9的特征图
            # fea1_1 = t.mean(fea1_1, dim=1)  # 1 1 32 64 64
            # fea1_1 = fea1_1.reshape(1, 1, 32, 64, 64)

            label_t_0 = output3[:, 0, :, :, :]  # 第0个通道下就是背景通道下的预测结果
            # print(label_t_0)
            #  label_t_0[(label_t_0 < 0.5) | (label_t_0 >= 0.9)] = 0
            # label_t_0[(label_t_0 >= 0.5) & (label_t_0 < 0.9)] = 1  # 将预测结果大于0.5小于0.9的特征保存下来，去掉其余的预测结果
            label_t_0[label_t_0 >= 0.5] = 1  # 将预测结果大于0.9的特征保存下来，去掉其余的预测结果
            label_t_0[label_t_0 < 0.5] = 0
            fea_0 = label_t_0 * fea  # 1 12 32 64 64
            fea_0 = fea_0.reshape(12, 32 * 64 * 64)
            mean_0 = t.mean(fea_0, dim=1)  # 12 1    得到平均胞体的特征值
            mean_0 = mean_0.reshape(1, 12, 1, 1, 1)

            label_t1_0 = output4[:, 0, :, :, :]  # 第一个通道下就是背景通道下的预测结果
            # print(label_t_0)
            label_t1_0[(label_t1_0 < 0.8)] = 0
            label_t1_0[(label_t1_0 >= 0.8)] = 1  # 将预测结果大于0.5小于0.9的特征保存下来，去掉其余的预测结果

            fea1_0 = label_t1_0 * fea  # 1 12 32 64 64  #背景预测结果大于0.5小于0.9的特征图
            # fea1_0 = t.mean(fea1_0, dim=1)  # 1 1 32 64 64
            # fea1_0 = fea1_0.reshape(1, 1, 32, 64, 64)

            dis_11 = abs(fea1_1 - mean_1) ** 2  # 1 12 32 64 64
            dis_11 = t.sum(dis_11, dim=1)
            dis_11 = t.sqrt(dis_11)
            dis_11 = dis_11.reshape(1, 1, 32, 64, 64)

            dis_10 = abs(fea1_1 - mean_0) ** 2  # 1 12 32 64 64
            dis_10 = t.sum(dis_10, dim=1)
            dis_10 = t.sqrt(dis_10)
            dis_10 = dis_10.reshape(1, 1, 32, 64, 64)

            dis_00 = abs(fea1_0 - mean_0) ** 2  # 1 12 32 64 64
            dis_00 = t.sum(dis_00, dim=1)
            dis_00 = t.sqrt(dis_00)
            dis_00 = dis_00.reshape(1, 1, 32, 64, 64)

            dis_01 = abs(fea1_0 - mean_1) ** 2  # 1 12 32 64 64
            dis_01 = t.sum(dis_01, dim=1)
            dis_01 = t.sqrt(dis_01)
            dis_01 = dis_01.reshape(1, 1, 32, 64, 64)

            w_1 = t.exp(-dis_11) / (t.exp(-dis_11) + t.exp(-dis_10))  # 1 1 32 64 64
            w_1 = w_1 * label_t1_1  # 1 1 32 64 64
          
            
            #  print(w_1)
            w_0 = t.exp(-dis_00) / (t.exp(-dis_00) + t.exp(-dis_01))  # 1 1 32 64 64

            w_0 = w_0 * label_t1_0  # 1 1 32 64 64
         
            
            w_1[w_1 >= 0.6] = 1
            w_1[w_1 < 0.6] = 2
            w_0[(w_0 < 0.5)|(w_0 >= 0.8)] = 2
            w_0[(w_0 >= 0.5)&(w_0 < 0.8)] = 0
            label_w = w_1 * w_0
            label_w[label_w == 2] = 1
            label_w[label_w == 4] = 2
            label_w = label_w.data.cpu().data.numpy()
            # print(label_w)
            '''
            '''
            label_w = label_w.reshape(32, 4096)
            data1 = pd.DataFrame(label_w)
            data1.to_csv('1伪标签t' + str(h) + '.csv')
            '''
            

            










            size = 1
            probs = output

            
            
            
            _, _,d, h, w = probs.size()
        # softmax = F.softmax(probs, dim=1)
            p = size
            softmax_pad = F.pad(probs, [p]*6, mode='replicate')
            affinity_group = []
            i=0
            for st_z in range(0, 2*size+1, size):
                for st_y in range(0, 2 * size + 1, size):
                    for st_x in range(0, 2 * size + 1, size):
                        if st_y == size and st_x == size  :
                            if st_z!=size:
                                affinity_paired = t.sum(0.5 * (t.log(
                                    2 * probs / (softmax_pad[:, :, st_z: st_z + d, st_y:st_y + h, st_x:st_x + w] +
                                                 probs)) * probs + t.log(
                                    2 * softmax_pad[:, :, st_z: st_z + d, st_y:st_y + h, st_x:st_x + w] / (
                                                softmax_pad[:, :, st_z: st_z + d, st_y:st_y + h,
                                                st_x:st_x + w] + probs)) *
                                                               softmax_pad[:, :, st_z: st_z + d, st_y:st_y + h,
                                                               st_x:st_x + w]), dim=1)
                                affinity_group.append(affinity_paired.unsqueeze(1))
                        elif st_z==size:

                            affinity_paired = t.sum(0.5 * (t.log(
                                2 * probs / (softmax_pad[:, :, st_z: st_z + d, st_y:st_y + h, st_x:st_x + w] +
                                             probs)) * probs + t.log(
                                2 * softmax_pad[:, :, st_z: st_z + d, st_y:st_y + h, st_x:st_x + w] / (
                                            softmax_pad[:, :, st_z: st_z + d, st_y:st_y + h, st_x:st_x + w] + probs)) *
                                                           softmax_pad[:, :, st_z: st_z + d, st_y:st_y + h,
                                                           st_x:st_x + w]), dim=1)
                            affinity_group.append(affinity_paired.unsqueeze(1))
            affinity = torch.cat(affinity_group, dim=1)
            affinity = torch.mean(affinity, dim=1)
            affinity = (affinity - affinity.min()) / (affinity.max() - affinity.min())
            affinity = affinity.reshape(32,64,64)

           # b = output[:,1,:,:,:]
           # b = b.reshape(32,64,64)
            




            h=self.h
            '''
             # 伪标签的选取
            b = output[:, 1, :, :, :]
          #  d = output[:, 1, :, :, :]
            #c = output[:, 1, :, :, :]
         #   c = fea.reshape(4,32,64,64).data.cpu().data.numpy()
          #  c = c.transpose((1,2,3,0))
            b = b.reshape(1, 131072).data.cpu().data.numpy()
         #   d = d.reshape(1, 131072).data.cpu().data.numpy()
            #c = c.reshape(1, 131072).data.cpu().data.numpy()
            
            m = 0
            n = 0
            for p in np.nditer(b):   #非零点到零点的距离   距离图选取的是预测结果>0.9的，而伪标签选取的是预测结果>0.95的
                if p < 0.3 :
                    b[0][m] = 0
                    n = n + 1
               
               # else:
                   # b[0][m] = 0

                m = m + 1
           # print(n)
           
            b = b.reshape(32, 64,64)
            b = b[0,:,:]
            b = b.reshape(64,64)

          #  dis = ndimage.morphology.distance_transform_edt(b)
           # dis = np.clip(dis,a_min=0,a_max=8)
          #  dis = dis/8
         #   dis = dis.reshape(32, 4096)
            #  b=b.cpu.numpy()
            data1 = pd.DataFrame(b)
            data1.to_csv('1测试标签' + str(h) + '.csv')
            '''
            





            '''
            d = d.reshape(32, 4096)
            #  b=b.cpu.numpy()
            data1 = pd.DataFrame(d)
            data1.to_csv('weibiaoqian'+str(h) + '.csv')

            c = c.reshape(32, 16384)
            #  b=b.cpu.numpy()
            data1 = pd.DataFrame(c)
            data1.to_csv('特征' + str(h) + '.csv')
            '''


            output = output.argmax(dim=1)
            output = output.data.cpu().data.numpy()
            img_pred = output[0, :, :, :]
            img_pred = img_pred
            self.pred[region[0]: region[1], region[2]:region[3], region[4]:region[5]] += img_pred     # 拼接预测结果
        output = self.pred[:self.zPixel, :self.yPixel, :self.yPixel]
        np.clip(output, 0, 1)

        return output,affinity#,w_1,w_0

    def numOfSegment(self):
        """分割图像个数"""
        x = math.ceil(self.zPixel / 114)
        y = math.ceil(self.xPixel / 114)
        z = math.ceil(self.yPixel / 114)
        num = x * y * z

        return num

    def regionOfSegment(self, id):
        """分割图像范围, 处理数据大小非128倍数的数据"""
        x = math.ceil(self.zPixel / 114)
        y = math.ceil(self.xPixel / 114)
        z = math.ceil(self.yPixel / 114)

        z_num, xy_num = math.floor(id / (x * y)), int(id % (x * y))
        x_num, y_num = int(xy_num % x), math.floor(xy_num / x)

        if x_num == x-1:
            x1 = self.zPixel - 128
            x2 = self.zPixel
        else:
            x1 = x_num * block_size[0]
            x2 = x1 + 128
        if y_num == y-1:
            y1 = self.xPixel - 128
            y2 = self.xPixel
        else:
            y1 = y_num * block_size[1]
            y2 = y1 + 128
        if z_num == z-1:
            z1 = self.yPixel - 128
            z2 = self.yPixel
        else:
            z1 = z_num * block_size[2]
            z2 = z1 + 128
        img = self.img[x1: x2, y1:y2, z1:z2]
        img = self.zScore(img)        # zscore归一化
        img = t.from_numpy(img)       # 将图像数据转换为张量
        #img_tensor = t.zeros(1, 1, 128, 128, 128)
        img_tensor = t.zeros(1, 1, 32, 64, 64)
        img_tensor[0, 0, :, :, :] = img
        img_tensor = Variable(img_tensor.cuda())

        return img_tensor, [x1, x2, y1, y2, z1, z2]

    def zScore(self, img):
        mean = ndimage.mean(img)
        var = ndimage.variance(img)
        out = (img - mean) / var

        return out


def GetBlockNum(a, b, c):

    x = math.ceil(a / 128)
    y = math.ceil(b / 128)
    z = math.ceil(c / 128)

    return x*y*z

def ReadImg(img_path):
    img = sitk.ReadImage(img_path)  # change
    img_arr = sitk.GetArrayFromImage(img)

    return img_arr

def WriteImg(img, predict_path):
    img = img * 255
    output_8bit = img.astype(np.uint8)
    img = sitk.GetImageFromArray(output_8bit)  # sitk.WriteImage支持的数据类型是8位
    sitk.WriteImage(img, predict_path)  # change

def AreaHist(area):
    """绘制胞体大小分布图"""
    plt.hist(area, bins=50, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.show()

if __name__=='__main__':

    mkdir_file()       # 创建保存结果文件夹
    print(imgPath)
    img_dir = os.listdir(imgPath)
    print(img_dir)
    start = time.time()
    t.cuda.set_device(3)
    #model = UnetGenerator_3d(in_dim=1, out_dim=2, num_filter=4).cuda()
    #model = KI_ASPP.Ki_ASPP(in_dim=1, out_dim=2, num_filter=4).cuda()
    #model = ResUnet.ResUnet(in_dim=1, out_dim=2, num_filter=4).cuda()
    model = unet3D.UnetGenerator_3d(in_dim=1, out_dim=2, num_filter=4).cuda()
    #model = KI_ASPP_middle.Ki_ASPP(in_dim=1, out_dim=2, num_filter=4).cuda()
    #model = Ki_AGs.KiUnet_3D(in_dim=1, out_dim=2, num_filter=4).cuda()
    #model = Ki_ASPP_chuan.KiUnet_3D(in_dim=1, out_dim=2, num_filter=4).cuda()
    #model = qingliangKiUnet.KiUnet_3D(in_dim=1, out_dim=2, num_filter=4).cuda()
    #model = SE3DUnet1(in_dim=1, out_dim=2, num_filter=4).cuda()
    model.load_state_dict(t.load(modelPath, map_location=lambda storage, loc: storage))  # change
    m=0
    a=0
    b=0
    c=0
    d=0
    e=0
    a1=0
    b1=0
    c1=0
    d1=0
    e1=0
    f=np.random.rand(5, 2)


    for name in img_dir:
        m=m+1
        print("name: ", name)
        start_single = time.time()
        img_path = imgPath + os.sep + name
        img = ReadImg(img_path)              # 读取图像+标准化

        predict_path = outputPath + '/image/' + name
        pred ,ASA= test(model, img,m).eval()
        WriteImg(pred, predict_path)      # 输出预测结果
        '''
        range1 = (w_1 >= 0.5) & (w_1 < 0.6)
        range2 = (w_1 >= 0.6) & (w_1 < 0.7)
        range3 = (w_1 >= 0.7) & (w_1 < 0.8)
        range4 = (w_1 >= 0.8) & (w_1 < 0.9)
        range5 = (w_1 >= 0.9) & (w_1 < 1)
        count = range1.sum().item()
        a=a+count
        count = range2.sum().item()
        b=b+count
        count = range3.sum().item()
        c=c+count
        count = range4.sum().item()
        d=d+count
        count = range5.sum().item()
        e=e+count

        range1 = (w_0 >= 0.5) & (w_0 < 0.6)
        range2 = (w_0 >= 0.6) & (w_0 < 0.7)
        range3 = (w_0 >= 0.7) & (w_0 < 0.8)
        range4 = (w_0 >= 0.8) & (w_0 < 0.9)
        range5 = (w_0 >= 0.9) & (w_0 < 1)
        count = range1.sum().item()
        a1=a1+count
        count = range2.sum().item()
        b1=b1+count
        count = range3.sum().item()
        c1=c1+count
        count = range4.sum().item()
        d1=d1+count
        count = range5.sum().item()
        e1=e1+count
        '''

        
        
        ASA = ASA.data.cpu().data.numpy()
        ASA = ASA * 255

        ASA = ASA.astype(np.float64)
        output_8bit = ASA.astype(np.uint8)
        ASA = sitk.GetImageFromArray(output_8bit)  # sitk.WriteImage支持的数据类型是8位
        predict_path1 =  outputPath+ '/asa/' + str(m)+ '.tif'

        sitk.WriteImage(ASA, predict_path1)  # change
        
    





        name_name = name.split('.')[0]
        centroid, area, bw = LocationCell_without_fibre(pred)
        swc_path = outputPath + '/swc/' + name_name + '.swc'
        output_swc(centroid, swc_path=swc_path, z_pixel=z_pixel)
        print("cost time:   ", (time.time()-start_single))
    '''
    f[0][0]=a/128
    f[1][0]=b/128
    f[2][0]=c/128
    f[3][0]=d/128
    f[4][0]=e/128
    f[0][1]=a1/128
    f[1][1]=b1/128
    f[2][1]=c1/128
    f[3][1]=d1/128
    f[4][1]=e1/128
    data1 = pd.DataFrame(f)
    data1.to_csv('t-s权重的数量.csv')
    '''
    


    print("cost time；  ", (time.time()-start)/len(img_dir))







