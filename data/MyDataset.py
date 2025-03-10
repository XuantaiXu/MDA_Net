# from torch.utils.data import Dataset
import numpy as np
import json
import os
import SimpleITK as sitk
import shutil
from scipy import ndimage

# class MyDataset(Dataset):
#     def __getitem__(self, idx):
#         pass
#
#     def __len__(self):
#         pass

class GetWeakSupervisonForegroundLabel(object):
    """获取弱监督label，每个胞体用固定半径的小球表示"""

    def __init__(self, swcPath, radius):
        self.swcPath = swcPath          # swc的路径
        self.radius = radius            # 小球的半径

    def read_swc(self):
        """读取swc文件，提取其中的胞体位置数据，存储到数组中，swc对应的坐标为（z, y, x）"""
        # print("--------------------------从SWC中提取胞体数据--------------------------")
        with open(self.swcPath, 'r') as file_to_read:
            soma_x, soma_y, soma_z = [], [], []
            while True:
                lines = file_to_read.readline()
                # print("line:", len(lines))

                if not lines or len(lines) < 5:
                    break
                item = [i for i in lines.split()]
                soma_x.append(int(float(item[2])))
                soma_y.append(int(float(item[3])))
                soma_z.append(int(float(item[4])))

            soma_x = np.array(soma_x).reshape(-1, 1)
            soma_y = np.array(soma_y).reshape(-1, 1)
            soma_z = np.array(soma_z).reshape(-1, 1)
            self.soma_location = np.concatenate([soma_z, soma_y, soma_x], axis=1)  # 将z和x交换

        return self.soma_location

    def create_label(self, z_pixel, block_size):
        """基于swc的数据，创建一个半径为固定的值小球label"""
        x_size = block_size[0]
        y_size = block_size[1]
        z_size = block_size[2]

        # out = np.zeros((512, 512, 512))
        out = np.zeros((1300, 1024, 1024))

        soma_num = len(self.soma_location)
        # print("soma num: ", soma_num)

        # out_ = np.zeros((512 + 2*self.radius, 512 + 2*self.radius, 512 + 2*self.radius))
        out_ = np.zeros((1300 + 2*self.radius, 1024 + 2*self.radius, 1024 + 2*self.radius))
        # out_[self.radius:512 + self.radius, self.radius:512 + self.radius, self.radius:512 + self.radius] = out.copy()
        out_[self.radius:1300 + self.radius, self.radius:1024 + self.radius, self.radius:1024 + self.radius] = out.copy()

        pad = 2*self.radius+1
        kernel = np.zeros((pad, pad, pad))
        vector2 = np.array([self.radius, self.radius, self.radius])
        for z in range(pad):
            for y in range(pad):
                for x in range(pad):
                    vector1 = np.array([x, y, z])
                    distance = np.linalg.norm(vector1 - vector2)
                    if distance <= self.radius:
                        kernel[x, y, z] = 1
        # print(kernel)

        # print("--------------------------------提取胞体的label----------------------------")

        for i in range(soma_num):
            x = self.soma_location[i, 0]//z_pixel
            y = self.soma_location[i, 1]
            z = self.soma_location[i, 2]
            # print(x, y, z)
            # print(out_[x: x + pad, y: y + pad, z: z + pad])
            out_[x: x + pad, y: y + pad, z: z + pad] += kernel

            # print(x, x+7, y, y+7, z, z+7)
        out_[out_ > 1] = 1
        # out = out_[self.radius:x_size+self.radius, self.radius:y_size+self.radius, self.radius:z_size+self.radius]
        out = out_[self.radius:x_size+self.radius, self.radius:y_size+self.radius, self.radius:z_size+self.radius]

        return out

    def extractLabel(self, z_size, block_size=[512, 512, 512]):
        _ = self.read_swc()
        out = self.create_label(z_size, block_size)
        return out

class GetWeakSupervisonBackgroundLabel(GetWeakSupervisonForegroundLabel):
    "提取胞体的背景label，选择立方体以外的像素为背景"
    def __init__(self, swcPath, radius):      # 父类的输入变量需要重写
        self.swcPath = swcPath
        self.radius = radius
        self.cube_length = 2*(self.radius+3)+1

    def create_label(self, z_size, block_size):
        # out = np.zeros((512, 512, 512))
        soma_num = len(self.soma_location)

        length = int(self.cube_length / 2)      # 立方体长的一半
        out = np.ones((512 + 2 * length, 512 + 2 * length, 512 + 2 * length))

        pad = self.cube_length

        # print("--------------------------------提取胞体背景的label----------------------------")
        for i in range(soma_num):
            x = self.soma_location[i, 0] // z_size
            y = self.soma_location[i, 1]
            z = self.soma_location[i, 2]
            # print(x, y, z)
            out[x: x + pad, y: y + pad, z: z + pad] = 0
            # print(x, x+7, y, y+7, z, z+7)
        # out_ = out[length:512+length, length:512+length, length:512+length]
        out_ = out[length:block_size[0]+length, length:block_size[1]+length, length:block_size[2]+length]

        return out_



class GetPatch(object):
    """提取图像的小patch"""
    def __init__(self, imgPath, outputPath, start_id, blockSize=[128, 128, 128]):
        self.block_size = blockSize
        self.img_path = imgPath
        self.output = outputPath
        self.start_id = start_id

    def get_image_block(self, block_num, img):
        # a, b, c = img.GetSize()
        a, b, c = img.shape
        x, y, z = int(a / self.block_size[0]), int(b / self.block_size[1]), int(c / self.block_size[2])
        z_num, xy_num = int(block_num / (x * y)), int(block_num % (x * y))
        x_num, y_num = int(xy_num % x), int(xy_num / x)
        img_block = img[x_num * self.block_size[0]: (x_num + 1) * self.block_size[0], y_num * self.block_size[1]: (y_num + 1) * self.block_size[1], z_num * self.block_size[2]: (z_num + 1) * self.block_size[2]]

        return img_block

    def get_patch(self):

        img_name = os.listdir(self.img_path)
        id = self.start_id
        for name_ in img_name:
            print('===============name:{}===================='.format(name_))
            img_path_1 = self.img_path + '/' + name_
            img = sitk.ReadImage(img_path_1)
            a, b, c = img.GetSize()
            print(a, b, c)
            num = int(a / self.block_size[0]) * int(b / self.block_size[1]) * int(c / self.block_size[2])
            print('==============the number of block:{}========='.format(num))

            img = sitk.GetArrayFromImage(img)
            for i in range(num):
                print('===============num:{}==============='.format(id))
                img_block = self.get_image_block(i, img)
                # output_path = output_dir + '/' + str("%04d" % id) + '.tif'
                output_path = self.output + '/' + str("%05d" % id) + '.tif'
                img_block = sitk.GetImageFromArray(img_block)
                sitk.WriteImage(img_block, output_path)
                id += 1

class GetBlock(object):
    def __init__(self, input, output, start_id=0):
        self.input = input
        self.output = output
        self.start_id = start_id

    def get_block(self):

        img_list = os.listdir(self.input)
        while(True):
            print("id: ", self.start_id)
            src = self.input + '/' + str("%05d" %self.start_id) + '.tif'
            dst = self.output + '/'
            name = str("%05d" %self.start_id) + '.tif'
            if name in img_list:
                shutil.move(src, dst)
            else:
                break
            self.start_id += 8

class GenerateBackgroundLabel(object):
    """
    将前景label向外扩充n层，生成背景label
    """
    def __init__(self, foregroundLabel, iterations,  backgroundLabelPath=0):
        self.img = foregroundLabel
        self.iterations = iterations
        self.outputPath = backgroundLabelPath

    def generateLabel(self):
        self.img = ndimage.binary_dilation(self.img, iterations=self.iterations)
        self.img = self.img.astype(np.uint8)
        self.img[self.img==0] = 2
        self.img[self.img==1] = 0
        self.img[self.img==2] = 1

        return self.img


if __name__ == "__main__":
    # ## 提取弱监督label
    # swcPath = r"Y:\FeaturedData\Public\zhangjianmin\yuxiang\195037\swc"
    # labelPath = r"Y:\FeaturedData\Public\zhangjianmin\yuxiang\195037\segment"
    # file_list = os.listdir(swcPath)
    # print(file_list)
    #
    # for name in file_list:
    #     swc = swcPath + '/' + name
    #     print(swc)
    #     label = labelPath + '/' + name.split('.')[0] + '.tif'
    #     print(label)
    #
    #     getpatch = GetWeakSupervisonForegroundLabel(swc, 4)
    #     out = getpatch.extractLabel(z_size=1, block_size=[801, 301, 601])    # z_size: swc中z向分辨率
    #     print(out.shape)
    #     out = out.astype(np.uint8)
    #     img = sitk.GetImageFromArray(out)  # sitk.WriteImage支持的数据类型是8位
    #     sitk.WriteImage(img, label)
    #     print("write image!!!")


    ## 分割数据块
    img_path = r'Y:\FeaturedData\Public\zhangjianmin\BYLW\train_data_color\Ai139\foreground_label'
    output = r'Y:\FeaturedData\Public\zhangjianmin\BYLW\train_data_color\Ai139\f'
    block_size = [128, 128, 128]
    print(os.listdir(img_path))
    get_patch = GetPatch(img_path, output, 1, block_size)
    get_patch.get_patch()

    # ## 移动图像  编号间隔8
    # input = r"Y:\FeaturedData\Public\zhangjianmin\BYLW\train_data_color\Ai139_Ai140_all_neuro\train_data\train\montage"
    # output = r"Y:\FeaturedData\Public\zhangjianmin\BYLW\train_data_color\Ai139_Ai140_all_neuro\train_data\val\montage"
    # move_image = GetBlock(input, output, start_id=1)
    # move_image.get_block()

    ## 生成全0或全1的图像
    # img = np.ones((512, 512, 512))
    # out = img.astype(np.uint8)
    # out = sitk.GetImageFromArray(out)  # sitk.WriteImage支持的数据类型是8位
    # sitk.WriteImage(out, r"G:\ZJM\problem_20200826\background_label\1.tif")

    # ## 以前景label为基础向外扩充获得背景label
    # src = r"E:\BYLW\data\193742\hand_label"
    # dst = r"E:\BYLW\data\193742\background_label"
    # imgList = os.listdir(src)
    # for name in imgList:
    #     print(name)
    #     imgPath = src + '/' + name
    #
    #     print(imgPath)
    #     img = sitk.ReadImage(imgPath)
    #     imgArr = sitk.GetArrayFromImage(img).astype(np.uint8)
    #
    #     out = GenerateBackgroundLabel(imgArr, iterations=3).generateLabel()
    #
    #     outPath = dst + '/' + name
    #     result = out.astype(np.uint8)
    #     result = sitk.GetImageFromArray(result)
    #     sitk.WriteImage(result, outPath)

