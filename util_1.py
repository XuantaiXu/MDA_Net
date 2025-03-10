import numpy as np
import SimpleITK as sitk
from skimage import measure, feature
import pandas as pd
from scipy import ndimage
import scipy
# import torch
import matplotlib.pyplot as plt
import time

start = time.time()

# def getOneHot(label, N):
#     '''将分割label转为hot'''
#     size = list(label.shape)
#     label = label.view(-1)
#     ones = torch.eye(N)
#     # print(ones)
#     ones = ones.index_select(0, label)
#     resize = np.zeros(5, dtype=int)
#     resize[0] = 1
#     resize[1:4] = size
#     resize[4] = N
#
#     return ones.view(*resize)

def binarization(img, th=35):
    img[img > th] = 255                                        # 将图像二值化
    img[img < th] = 0

    return img

def del_background(img, img_b):
    '''img
         提取前景信号
    :param img:   预测图像
    :param img_b:   二值化预测图像（去除小信号）
    :return:
    '''

    img[img_b == 0] = 0

    return img

def del_small_signal(img, value=10):
    '''
    :param img:   二值化图像
    :param value:   筛选胞体的体积
    :return:
    '''
    labels = measure.label(img, connectivity=2)
    props = measure.regionprops(labels)  # 统计连通域的信息
    num = len(props)
    for i in range(num):
        volume = props[i].area
        if volume < value:
            img[labels == i] = 0                    # 从二值化图像中去除小信号

    return img

def ex_centroid(img):
    '''
    :param img:  二值化图像
    :return:
    '''
    img = ndimage.binary_fill_holes(img)
    # img = ndimage.binary_erosion(img, iterations=2)

    labels = measure.label(img, connectivity=2)
    props = measure.regionprops(labels)                #统计连通域的信息
    num = len(props)
    centroid = np.zeros((num, 4))
    area = np.zeros(num)
    print("num:", num)
    for i in range(num):
        if props[i].area > 5:
            centroid[i, 0] = np.floor(props[i].centroid[0])
            centroid[i, 1] = np.floor(props[i].centroid[1])
            centroid[i, 2] = np.floor(props[i].centroid[2])
            area[i] = props[i].area

    # f = open("soma_centroid.txt", "w")
    # f.writelines(str(centroid))
    # f.close()
    # stistic_data = pd.DataFrame(centroid)
    # writer = pd.ExcelWriter(r'01_centroid.xlsx')  # 写入Excel文件
    # stistic_data.to_excel(writer, float_format='%.3f')  # ‘page_1’是写入excel的sheet名
    # writer.save()
    # writer.close()

    return centroid

def ex_centroid_with_fibre(img, predict_path):
    '''
    :param img:  二值化图像

        空洞填充
        三次腐蚀
        去除体积小于100的胞体
    :return:
    '''
    img = ndimage.binary_fill_holes(img)
    img = ndimage.binary_erosion(img, iterations=1)

    # output = img * 255
    # output_8bit = output.astype(np.uint8)
    # img3 = sitk.GetImageFromArray(output_8bit)                  # sitk.WriteImage支持的数据类型是8位
    # sitk.WriteImage(img3, predict_path)  # change

    labels = measure.label(img, connectivity=2)
    props = measure.regionprops(labels)                #统计连通域的信息
    num = len(props)
    centroid = np.zeros((num, 4))
    area = np.zeros(num)
    print("num:", num)
    for i in range(num):
        if props[i].area > 100:
            centroid[i, 0] = np.floor(props[i].centroid[0])
            centroid[i, 1] = np.floor(props[i].centroid[1])
            centroid[i, 2] = np.floor(props[i].centroid[2])
            area[i] = props[i].area

    return centroid

# def data_augment(img):

def output_swc(centroid, swc_path, z_pixel=1):

    f = open(swc_path, "w")
    b = np.zeros((1, 7))
    for i in range(len(centroid)):
        if centroid[i, 2] == 0.0 or centroid[i, 1] == 0.0 or centroid[i, 0] == 0.0:
            continue
        else:
            f.writelines(str(i)), f.writelines(' ')
            f.writelines(str(1)), f.writelines(' ')
            f.writelines(str(centroid[i, 2])), f.writelines(' ')
            f.writelines(str(centroid[i, 1])), f.writelines(' ')
            f.writelines(str(centroid[i, 0] * z_pixel)), f.writelines(' ')
            f.writelines(str(1)), f.writelines(' ')    ## 临时添加
            f.writelines(str(-1))
            f.writelines('\n')
    f.close()

def Dist3D(img):
    """三维距离矩阵计算"""
    # print("img shape:", img.shape)
    edt = ndimage.distance_transform_edt(img)        ## 将二值化图像转换为距离矩阵 耗时
    return edt


def ExtractMinBox(img, bbox):
    """提取包含单个连通域的小数据块"""

    initialInd = [bbox[0], bbox[1], bbox[2]]      # 数据块的起点坐标
    # print(initialInd)
    img1 = img[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]

    return initialInd, img1


def LocationCell(img):
    """
    解决胞体粘连的问题
    img：二值化图像
    :return: 胞体定位数据
    """
    img = ndimage.binary_opening(img)
    img = ndimage.binary_closing(img)  # 闭操作  先膨胀后收缩
    print("image closing")

    img_cp = img.copy()
    img_cp_lable = measure.label(img_cp, connectivity=2)  # 8连通域
    img_cp_props = measure.regionprops(img_cp_lable)  # 统计连通域的信息
    print("num: ", len(img_cp_props))
    for i in range(len(img_cp_props)):
        if img_cp_props[i].area < 50:
            print(img_cp_props[i].area)
            img_cp[img_cp_lable == img_cp_props[i].label] = 0
    print("del small soma!")

    # out = img_cp.astype(np.uint8)
    # out = sitk.GetImageFromArray(out)
    # sitk.WriteImage(out, r"F:\cell_detection_work\predict_result_data\195413_CH1\predict_model14_54\02_test_1.tif")
    # print("wirte end！")


    dist = Dist3D(img_cp)                        # 将二值化图像变为距离图像
    print("dist")
    lm = ndimage.maximum_filter(dist, size=10)
    print("maximum_filter")

    # 寻找局部最大值
    msk = dist.copy()
    msk[msk == 0] = -1
    ind = np.where(msk == lm)  # 局部最大值的坐标
    local_max = np.zeros_like(msk)
    local_max[ind] = 255

    # 输出局部最大值之后的结果
    # out = img.astype(np.uint8)
    # out = sitk.GetImageFromArray(out)
    # sitk.WriteImage(out, r"F:\xuediao_project_20190626\02_test_1.tif")
    # print('+====================')

    ## 胞体定位
    labels = measure.label(local_max, connectivity=2)  # 8连通域
    props = measure.regionprops(labels)  # 统计连通域的信息
    num = len(props)  # 胞体的个数
    centroid = np.zeros((num, 3))
    for i in range(num):
        centroid[i, 0] = np.floor(props[i].centroid[0])
        centroid[i, 1] = np.floor(props[i].centroid[1])
        centroid[i, 2] = np.floor(props[i].centroid[2])
    # print("centroid:  ", centroid)
    # output_swc(centroid, r"F:\xuediao_project_20190626\02-1.swc")

    return centroid

def LocationCell_1(img):
    """
    解决胞体粘连的问题
    img：二值化图像
    :return: 胞体定位数据
    """

    # 平滑
    img = ndimage.binary_opening(img)
    img = ndimage.binary_closing(img)  # 闭操作  先膨胀后收缩
    # print("image closing")

    img_cp = img.copy()
    img_cp_lable = measure.label(img_cp, connectivity=2)  # 8连通域
    img_cp_props = measure.regionprops(img_cp_lable)  # 统计连通域的信息
    # print("num: ", len(img_cp_props))
    centroid = []
    for i in range(len(img_cp_props)):
        if img_cp_props[i].area < 50:        # 去除体积较小的连通域
            # print(img_cp_props[i].area)
            img_cp[img_cp_lable == img_cp_props[i].label] = 0
        else:
            initialInd, img1 = ExtractMinBox(img, img_cp_props[i].bbox)
            dist = Dist3D(img1)
            lm = ndimage.maximum_filter(dist, size=10)

            # 寻找局部最大值
            msk = dist.copy()
            msk[msk == 0] = -1
            ind = np.where(msk == lm)  # 局部最大值的坐标
            local_max = np.zeros_like(msk)
            local_max[ind] = 255

            ## 胞体定位
            labels = measure.label(local_max, connectivity=2)  # 8连通域
            props = measure.regionprops(labels)  # 统计连通域的信息
            num = len(props)  # 胞体的个数

            for j in range(num):
                soma_x = np.floor(props[j].centroid[0]) + initialInd[0]
                soma_y = np.floor(props[j].centroid[1]) + initialInd[1]
                soma_z = np.floor(props[j].centroid[2]) + initialInd[2]
                centroid.append([soma_x, soma_y, soma_z])

    centroid = np.array(centroid)
    # print("centroid:  ", centroid.shape)
    # print(centroid[1, 2])
    return centroid

def LocationCell_without_fibre(img):
    """
    针对数据类型 #192420
    :param img:
    :return:
    """
    #平滑
    print("binary image smooth!!!")
    img = ndimage.binary_fill_holes(img)      # 空洞填充
    img = ndimage.binary_opening(img)
    img = ndimage.binary_closing(img)  # 闭操作  先膨胀后收缩

    ## 胞体定位
    print("extrace centorid!!!")
    labels = measure.label(img, connectivity=2)  # 8连通域
    props = measure.regionprops(labels)  # 统计连通域的信息
    num = len(props)  # 胞体的个数

    centroid = []
    area = []
    for j in range(num):
        if props[j].area < 50:
            soma_x = np.floor(props[j].centroid[0])
            soma_y = np.floor(props[j].centroid[1])
            soma_z = np.floor(props[j].centroid[2])
            centroid.append([soma_x, soma_y, soma_z])
            area.append(props[j].area)
        else:  # 当连通域体积大小700时，使用粘连胞体分割程序
            initialInd, img1 = ExtractMinBox(img, props[j].bbox)
            dist = ndimage.distance_transform_edt(img1)
            dist_1 = ndimage.maximum_filter(dist, size=4)  # 避免胞体过分割
            dist_2 = ndimage.maximum_filter(dist, size=12)  # 提取局部极大值

            # 寻找局部最大值
            dist_1[dist_1 == 0] = -1
            ind = np.where(dist_1 == dist_2)  # 局部最大值的坐标
            local_max = np.zeros_like(dist)
            local_max[ind] = 255

            ## 胞体定位
            labels = measure.label(local_max, connectivity=2)  # 8连通域
            props_local = measure.regionprops(labels)  # 统计连通域的信息
            num_local = len(props_local)  # 胞体的个数
            # print("num_local:", num_local)

            for k in range(num_local):
                soma_x = np.floor(props_local[k].centroid[0]) + initialInd[0]
                soma_y = np.floor(props_local[k].centroid[1]) + initialInd[1]
                soma_z = np.floor(props_local[k].centroid[2]) + initialInd[2]
                centroid.append([soma_x, soma_y, soma_z])

    centroid = np.array(centroid)
    area = np.array(area)

    return centroid, area, img

def LocationCell_without_fibre_2(img):
    """
    针对数据类型 #192420
    :param img:
    :return:
    """
    #平滑
    print("binary image smooth!!!")
    img = ndimage.binary_fill_holes(img)      # 空洞填充
    img = ndimage.binary_opening(img)
    img = ndimage.binary_closing(img)  # 闭操作  先膨胀后收缩

    ## 胞体定位
    print("extrace centorid!!!")
    labels = measure.label(img, connectivity=2)  # 8连通域
    props = measure.regionprops(labels)  # 统计连通域的信息
    num = len(props)  # 胞体的个数

    centroid = []
    area = []
    for j in range(num):
        if props[j].area < 50:
            soma_x = np.floor(props[j].centroid[0])
            soma_y = np.floor(props[j].centroid[1])
            soma_z = np.floor(props[j].centroid[2])
            centroid.append([soma_x, soma_y, soma_z])
            area.append(props[j].area)
        else:  # 当连通域体积大小700时，使用粘连胞体分割程序
            initialInd, img1 = ExtractMinBox(img, props[j].bbox)
            dist = ndimage.distance_transform_edt(img1)
            median = np.median(np.where(dist != 0))
            dist[dist < median] == 0
            dist_1 = ndimage.maximum_filter(dist, size=1)  # 避免胞体过分割
            dist_2 = ndimage.maximum_filter(dist, size=6)  # 提取局部极大值

            # 寻找局部最大值
            dist_1[dist_1 == 0] = -1
            ind = np.where(dist_1 == dist_2)  # 局部最大值的坐标
            local_max = np.zeros_like(dist)
            local_max[ind] = 255

            ## 胞体定位
            labels = measure.label(local_max, connectivity=2)  # 8连通域
            props_local = measure.regionprops(labels)  # 统计连通域的信息
            num_local = len(props_local)  # 胞体的个数
            # print("num_local:", num_local)

            for k in range(num_local):
                soma_x = np.floor(props_local[k].centroid[0]) + initialInd[0]
                soma_y = np.floor(props_local[k].centroid[1]) + initialInd[1]
                soma_z = np.floor(props_local[k].centroid[2]) + initialInd[2]
                centroid.append([soma_x, soma_y, soma_z])

    centroid = np.array(centroid)
    area = np.array(area)
    print("num: ", len(centroid))
    return centroid, area, img

def LocationCell_without_fibre_1(img):
    """
    针对数据类型 #192420
    :param img:
    :return:
    """
    #平滑
    print("binary image smooth!!!")
    img = ndimage.binary_fill_holes(img)      # 空洞填充
    img = ndimage.binary_opening(img)
    img = ndimage.binary_closing(img)  # 闭操作  先膨胀后收缩

    ## 胞体定位
    print("extrace centorid!!!")
    labels = measure.label(img, connectivity=2)  # 8连通域
    props = measure.regionprops(labels)  # 统计连通域的信息
    num = len(props)  # 胞体的个数

    centroid = []
    area = []
    for j in range(num):
        if props[j].area < 50:
            soma_x = np.floor(props[j].centroid[0])
            soma_y = np.floor(props[j].centroid[1])
            soma_z = np.floor(props[j].centroid[2])
            centroid.append([soma_x, soma_y, soma_z])
            area.append(props[j].area)
        else:  # 当连通域体积大小700时，使用粘连胞体分割程序
            initialInd, img1 = ExtractMinBox(img, props[j].bbox)
            img1 = ndimage.binary_opening(img1, iterations=2)
            img1 = ndimage.binary_closing(img1, iterations=2)  # 闭操作  先膨胀后收缩
            dist = ndimage.distance_transform_edt(img1)

            # print(dist)
            # # dist[dist < 3] = 0
            # dist_1 = ndimage.maximum_filter(dist, size=3)  # 避免胞体过分割
            # dist_2 = ndimage.maximum_filter(dist, size=6)  # 提取局部极大值

            # image_max = ndimage.maximum_filter(dist, size=5, mode='constant')
            # print(image_max)
            block_centroid = feature.peak_local_max(dist, indices=True, footprint=np.ones((3, 3, 3)), labels=img1)
            # result = feature.peak_local_max(image_max, min_distance=3, indices=True)
            #
            # markers = ndimage.label(local_maxi)[0]
            #
            # print(local_maxi)
            #
            block_centroid[:, 0] += initialInd[0]
            block_centroid[:, 1] += initialInd[1]
            block_centroid[:, 2] += initialInd[2]
            if not centroid:
                centroid = np.concatenate((centroid, block_centroid), axis=0)
            else:
                centroid = block_centroid
            # centroid = result
            # # 寻找局部最大值
            # dist_1[dist_1 == 0] = -1
            # ind = np.where(dist_1 == dist_2)  # 局部最大值的坐标
            # local_max = np.zeros_like(dist)
            # local_max[ind] = 255

            # ## 胞体定位
            # labels = measure.label(local_max, connectivity=2)  # 8连通域
            # props_local = measure.regionprops(labels)  # 统计连通域的信息
            # num_local = len(props_local)  # 胞体的个数
            # # print("num_local:", num_local)

            # for k in range(num_local):
            #     soma_x = np.floor(props_local[k].centroid[0]) + initialInd[0]
            #     soma_y = np.floor(props_local[k].centroid[1]) + initialInd[1]
            #     soma_z = np.floor(props_local[k].centroid[2]) + initialInd[2]
            #     centroid.append([soma_x, soma_y, soma_z])

    centroid = np.array(centroid)
    area = np.array(area)

    return centroid, area, img1

def LocationCell_with_fibre(img):
    """
    针对数据类型 #193742  #193743
    带纤维的胞体定位
    img：二值化图像
    :return: 胞体定位数据
    算法流程：

    """
    print("binary image smooth!!!")
    img = ndimage.binary_dilation(img, iterations=2)    # 膨胀
    img = ndimage.binary_fill_holes(img)  # 空洞填充
    img = ndimage.binary_erosion(img, iterations=2)     # 腐蚀

    # 平滑
    img = ndimage.binary_opening(img, iterations=1)
    img = ndimage.binary_closing(img)  # 闭操作  先膨胀后收缩

    ## 胞体定位
    print("extrace centorid!!!")
    labels = measure.label(img, connectivity=2)  # 8连通域
    props = measure.regionprops(labels)  # 统计连通域的信息
    num = len(props)  # 胞体的个数

    centroid = []
    area = []
    for j in range(num):
        if props[j].area < 50:
            pass
        if props[j].area < 600:
            soma_x = np.floor(props[j].centroid[0])
            soma_y = np.floor(props[j].centroid[1])
            soma_z = np.floor(props[j].centroid[2])
            centroid.append([soma_x, soma_y, soma_z])
            # area.append(props[j].area)
        else:      # 当连通域体积大小700时，使用粘连胞体分割程序
            initialInd, img1 = ExtractMinBox(img, props[j].bbox)
            dist = ndimage.distance_transform_edt(img1)
            dist_1 = ndimage.maximum_filter(dist, size=8)     # 避免胞体过分割
            dist_2 = ndimage.maximum_filter(dist, size=15)    # 提取局部极大值

            # 寻找局部最大值
            dist_1[dist_1 == 0] = -1
            ind = np.where(dist_1 == dist_2)  # 局部最大值的坐标
            local_max = np.zeros_like(dist)
            local_max[ind] = 255

            ## 胞体定位
            labels = measure.label(local_max, connectivity=2)  # 8连通域
            props_local = measure.regionprops(labels)  # 统计连通域的信息
            num_local = len(props_local)  # 胞体的个数
            # print("num_local:", num_local)

            for k in range(num_local):
                soma_x = np.floor(props_local[k].centroid[0]) + initialInd[0]
                soma_y = np.floor(props_local[k].centroid[1]) + initialInd[1]
                soma_z = np.floor(props_local[k].centroid[2]) + initialInd[2]
                centroid.append([soma_x, soma_y, soma_z])

    centroid = np.array(centroid)
    # area = np.array(area)

    return centroid

def histeq(img, n_bins=256):
    # 直方图均衡化
    # 获取图像的直方图
    imhist, bins = np.histogram(img.flatten(), n_bins, normed=True)

    # 获取变换后的灰度值
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]

    # 获取变换后的图像
    result = np.interp(img.flatten(), bins[:-1], cdf)

    return result.reshape(img.shape)

def histNormalied(InputImage, O_min = 0, O_max=255):
    # 直方图正规化
    I_min = np.min(InputImage)
    I_max = np.max(InputImage)

    # 输出映射的图像
    if(I_min != I_max):
        cofficient = float(O_max - O_min) / float(I_max - I_min)
        OutputImage = cofficient * (InputImage - I_min) + O_min

        return OutputImage
    else:
        return InputImage

if __name__ == "__main__":

    img = sitk.ReadImage(r"C:\Users\zhangjianmin\Desktop\test\192420-5.tif")
    img = sitk.GetArrayFromImage(img)
    # img_b = binarization(img, th=100)

    print("===============1=================")
    centroid, area, out_image = LocationCell_without_fibre_2(img)
    output_swc(centroid, r"C:\Users\zhangjianmin\Desktop\test\192420-5.swc", z_pixel=2)
    print(centroid)
    print("===============2=================")
    out_image = out_image.astype(np.uint8)
    out_image = sitk.GetImageFromArray(out_image)
    sitk.WriteImage(out_image, r"C:\Users\zhangjianmin\Desktop\test\192420-5_1.tif")
