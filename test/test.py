## 注意： 数据块的大小应该是blockz_size的整数倍

import torch as t
import math
from torch import nn
from model import unet3D as UnetGenerator_3d
import numpy as np
import SimpleITK as sitk
import cv2
import os
# from skimage import measure
from torch.autograd import Variable
from util import *
from scipy.ndimage.interpolation import zoom

# filePath = r'H:\zmt'                          #change
imgPath = r'/mnt/data/zq/zq/3D-Unet/256data/18426/test/mont128'                       # 需要预测的数据块路径
predict_img = r'/mnt/data/zq/zq/3D-Unet/256data/test_predict'
swc_result = r'/mnt/data/zq/zq/3D-Unet/256data/test_swc'
modelPath = '/mnt/data/zq/zq/3D-Unet/256data/model/CP999.pth'              # 训练好的模型
block_size = [128, 128, 128]                       # change 数据块的大小  数值：x,y,z
# volume = np.zeros(10)                         #change   数据块的个数

def get_image_block(block_num, img, block_size=[128, 128, 128]):
    a, b, c = img.GetSize()
    x, y, z = int(a/block_size[0]), int(b/block_size[1]), int(c/block_size[2])
    # x, y, z = math.ceil(a / block_size[0]), math.ceil(b / block_size[1]), math.ceil(c / block_size[2])
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

def predict(model, img, predict_path):
    for i in range(num):
        print('=================num:{}==============='.format(i))
        img1 = get_image_block(i, img, block_size=block_size)
        img1 = Variable(img1.cuda())
        img1 = t.sigmoid(model(img1)) * 255

        img1 = img1.data.cpu().data.numpy()
        # img2 = np.ones((128, 128, 128))  # change     与block_size保持一致
        img2 = img1[0, 0, :, :, :]

        ## 将预测的小数据拼接到一起
        a, b, c = img.GetSize()
        x, y, z = int(a / block_size[0]), int(b / block_size[1]), int(c / block_size[2])
        z1, num0 = int(i / (x * y)), int(i % (x * y))
        x1, y1 = int(num0 % x), int(num0 / x)
        pred[z1 * block_size[2]:(z1 + 1) * block_size[2], y1 * block_size[0]: (y1 + 1) * block_size[0],
        x1 * block_size[1]:(x1 + 1) * block_size[1]] = img2
    #
    pred_8bit = pred.astype(np.uint8)
    img3 = sitk.GetImageFromArray(pred_8bit)                  # sitk.WriteImage支持的数据类型是8位
    sitk.WriteImage(img3, predict_path)  # change

    return pred_8bit

if __name__=='__main__':

    img_dir = os.listdir(imgPath)
    for name in img_dir:

        img_path = imgPath + '/' + name

        img = sitk.ReadImage(img_path)  # change
        a, b, c = img.GetSize()
        print('===================img size:', a, b, c)
        num = int(a / block_size[0]) * int(b / block_size[1]) * int(c / block_size[2])
        # num = math.ceil(a / block_size[0]) * math.ceil(b / block_size[1]) * math.ceil(c / block_size[2])
        print('========the number of block:', num)

        # 输出数据块的设置
        pred = np.zeros((a, b, c))  # change     将分割后的图像拼接起来

        model = UnetGenerator_3d(in_dim=1, out_dim=1, num_filter=4).cuda()
        t.cuda.set_device(0)
        model.cuda()
        model.load_state_dict(t.load(modelPath, map_location=lambda storage, loc: storage))  # change
        predict_path = predict_img + '/' + name
        pred = predict(model, img, predict_path=predict_path)

        threshold = [65, 70, 75, 80, 85]
        for th in threshold:
            img_b = binarization(pred, th=th)
            centroid = ex_centroid(img_b)
            name_1 = name.split('.')[0]

            swc_path = swc_result + '/' + name_1 + '_' + str(th) + '.swc'
            output_swc(centroid, swc_path=swc_path)
            print("=========================={} saved!===================".format(swc_path))






