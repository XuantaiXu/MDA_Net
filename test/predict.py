import torch as t
from torch import nn
from model import unet3D as UnetGenerator_3d
import numpy as np
import SimpleITK as sitk
import cv2
# from skimage import measure
from torch.autograd import Variable
from scipy.ndimage.interpolation import zoom

filePath = r'train_data\193725_1.tif'                    # change
imgPath = r'train_data\193725_2.tif'                     #
modelPath = './model/CP42.pth'
block_size = [64, 64, 64]                   #change 数据块的大小  数值：x,y,z
volume = np.zeros(10)                   #change   数据块的个数

def get_image_block(block_num, img, block_size=[128, 128, 128]):
    a, b, c = img.GetSize()
    x, y, z = int(a/block_size[0]), int(b/block_size[1]), int(c/block_size[2])
    z_num, xy_num = int(block_num/(x*y)), int(block_num%(x*y))
    x_num, y_num = int(xy_num%x), int(xy_num/x)
    img_block = img[x_num*block_size[0]: (x_num+1)*block_size[0], y_num*block_size[1]: (y_num+1)*block_size[1], z_num*block_size[2]: (z_num+1)*block_size[2]]
    img_arr = sitk.GetArrayFromImage(img_block)
    img_arr = img_arr.astype(np.uint8)
    img_arr = t.from_numpy(img_arr)
    # print(img_arr.shape)
    # 将图像数据转换为张量
    img_arr_1 = t.zeros(1, 1, block_size[0], block_size[1], block_size[2])
    # print(img_arr_1.shape)
    img_arr_1[0, 0, :, :, :] = img_arr

    return img_arr_1

img = sitk.ReadImage(imgPath)                      #change
a, b, c = img.GetSize()
num = int(a/block_size[0]) * int(b/block_size[1]) * int(c/block_size[2])
print('===================the number of block:{}======================='.format(num))

# 输出数据块的设置
pred = np.zeros((a, b, c))               #change     将分割后的图像拼接起来

net = UnetGenerator_3d(in_dim=1,out_dim=1,num_filter=4).cuda()
t.cuda.set_device(0)
net.cuda()
net.load_state_dict(t.load(modelPath, map_location=lambda storage, loc: storage))   #change

for i in range(num):
    print(i)
    img1 = get_image_block(i, img, block_size=block_size)
    print(img1.shape)
    img1 = Variable(img1.cuda())
    img1 = t.sigmoid(net(img1))*255

    img1 = img1.data.cpu().data.numpy()
    print(img1.shape)
    img2 = np.ones((64, 64, 64))                  # change     与block_size保持一致
    img2 = img1[0, 0, :, :, :]

    ## 将预测的小数据拼接到一起
    a, b, c = img.GetSize()
    x, y, z = int(a/block_size[0]), int(b/block_size[1]), int(c/block_size[2])
    z1, num0 = int(i / (x*y)), int(i % (x*y))
    x1, y1 = int(num0 % x), int(num0/x)
    print(z1*block_size[2], (z1+1)*block_size[2], y1*block_size[0], (y1+1)*block_size[0], x1*block_size[1], (x1+1)*block_size[1])
    pred[z1*block_size[2]:(z1+1)*block_size[2], y1*block_size[0]: (y1+1)*block_size[0], x1*block_size[1]:(x1+1)*block_size[1]] = img2
#
pred_8bit = pred.astype(np.uint8)
img3 = sitk.GetImageFromArray(pred_8bit)        #sitk.WriteImage支持的数据类型是8位
sitk.WriteImage(img3, r'train_data\193725_2_predict.tif')     #change



