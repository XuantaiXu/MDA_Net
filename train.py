<<<<<<< HEAD
=======


>>>>>>> 6f3edda40e28e86a6a80286feb21049eb70536a0
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

#from  model import doubleunet

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



# from tensorboardX import SummaryWriter
from scipy.ndimage.interpolation import zoom


train_path = "/mnt/data/xxt/MDA_Net/dataset/64_64_32"  
dir_checkpoint = "/mnt/data/xxt/MDA_Net/dataset/adam_model_64_64_32_GAN"


lr = [0.01, 0.001]   #Learning Rate
seed = 42  # Random Seed
alpha = 0.1 # Adversarial Loss Parameter
beta = 1  # Self-Training Loss Parameter
N = 60 # Parameter for the Onset of Self-Training

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



def recall(predict, target): 
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


def precision(predict, target):  
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


def F1(predict, target):  
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

def tenwayASCLoss(probs, size=1):   #similarity map
    _, _,d, h, w = probs.size()
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
    affinity = affinity.reshape(1,10,32,64,64)
    return affinity



class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)


    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None




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
    
    img_arr_1 = t.zeros(1, 1, block_size[0], block_size[1], block_size[2])
    img_arr_1[0, 0, :, :, :] = img_arr

    return img_arr_1


def img2tensor1(img_path, mantage=False):
    img = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img).astype(np.float64)
   
    if mantage:
        mean = ndimage.mean(img_arr)
        var = ndimage.variance(img_arr)
        img_arr = (img_arr - mean) / var
    img_arr = t.from_numpy(img_arr)
  
    img_arr_1 = t.zeros(1, 1, 32, 64, 64) 
    img_arr_1[0, 0, :, :, :] = img_arr
   # Sigmoid=nn.Sigmoid()
    #img_arr_1=Sigmoid(img_arr_1)

    return img_arr_1

def img2tensor2(img_path, mantage=False):
    img = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img).astype(np.float64)

    if mantage:
        mean = ndimage.mean(img_arr)
        var = ndimage.variance(img_arr)
        img_arr = (img_arr - mean) / var
    img_arr = t.from_numpy(img_arr)
   
    img_arr_1 = t.zeros(1, 1, 32, 64, 64) 
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
    :param N:   
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



    criterion = nn.BCELoss()                  
    criterion1 = nn.CrossEntropyLoss(ignore_index=2)


    model_D=discriminator.FCDiscriminator()   
    model_D1=discriminator1.FCDiscriminator()
    model_D.cuda()
    model_D1.cuda()


    optimizer1 = optim.Adam(model.parameters(), lr=lr[0], weight_decay=0.0005)
    optimizer2 = optim.Adam(model_D.parameters(), lr=lr[0], weight_decay=0.0005)
    optimizer3 = optim.Adam(model_D1.parameters(), lr=lr[0], weight_decay=0.0005)

    scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=[30], gamma=0.1)
    scheduler2 = lr_scheduler.MultiStepLR(optimizer2, milestones=[30], gamma=0.1)
    scheduler3 = lr_scheduler.MultiStepLR(optimizer3, milestones=[30], gamma=0.1)

    train_loss = []
    val_loss = []

    loss_summury = np.zeros((epochs, 2))
    for epoch in range(epochs):
        epoch_loss = 0
        img_dir1 = os.listdir(mantage1)
        img_num = len(img_dir1)

        loss_batch_D = 0
        loss_batch_G=0
        loss_batch=0
        gt1 = []
        gt2=[]
        predict_y_np_s = []



        for i, name in enumerate(img_dir1):
            i=i+1
            p = float(i + epoch * 128) / 1500 / 128
            Lambda= 2. / (1. + np.exp(-10 * p)) - 1
            mantage_path1 = mantage1 + '/' + name
            foreground_label_path1 = foreground_label1 + '/' + name
            mantage_path2 = mantage2 + '/' + name
            foreground_label_path2 = foreground_label2 + '/' + name

        #######
            #label  source
            mantage_img_block1 = img2tensor2(mantage_path1, mantage=True).cuda()
            # print("xingzhuang",mantage_img_block.shape)
            true_y = GetLabel(foreground_label_path1, 2)
            true_y_qu1 = t.squeeze(true_y)
            true_y_suoyin = t.argmax(true_y_qu1, dim=0)
            gt1 = np.append(gt1, true_y_suoyin)
            true_y = true_y.cuda()
          ########
            # label  target
            mantage_img_block2 = img2tensor1(mantage_path2, mantage=True).cuda()
            # print("xingzhuang",mantage_img_block.shape)
            true_y_2 = GetLabel(foreground_label_path2, 2)
            true_y_qu2 = t.squeeze(true_y_2)
            true_y_suoyin2 = t.argmax(true_y_qu2, dim=0)
            gt2 = np.append(gt2, true_y_suoyin2)
            true_y_2 = true_y_2.cuda()


            predict_y_s,adv_s= model(mantage_img_block1)
            A_s = tenwayASCLoss(predict_y_s)
            predict_y_s_d=ReverseLayerF.apply(adv_s, Lambda)
            predict_qu1_s = t.squeeze(predict_y_s)
            pre_suoyin_s = t.argmax(predict_qu1_s, dim=0)
            pre_suoyin_s = pre_suoyin_s.data.cpu()
            predict_y_np_s = np.append(predict_y_np_s, pre_suoyin_s)

            #target predict
            predict_y_t,adv_t = model(mantage_img_block2)
            A_t = tenwayASCLoss(predict_y_t)
            predict_y_t_d = ReverseLayerF.apply(adv_t, Lambda)


            if epoch >= N:    # generate pseudo-labels
                output1 = (predict_y_t + predict_y_t) / 2
                output2 = (predict_y_t + predict_y_t) / 2
                output3 = (predict_y_t + predict_y_t) / 2
                output4 = (predict_y_t + predict_y_t) / 2
                label_t_1 = output1[:, 1, :, :, :]  

                label_t_1[label_t_1 >= 0.5] = 1  
                label_t_1[label_t_1 < 0.5] = 0
                fea_1 = label_t_1 * adv_t  # 1 12 32 64 64
                fea_1 = fea_1.reshape(12, 32 * 64 * 64)
                mean_1 = t.mean(fea_1, dim=1)  # 12 1   
                mean_1 = mean_1.reshape(1, 12, 1, 1, 1)

                label_t1_1 = output2[:, 1, :, :, :]  
                label_t1_1[(label_t1_1 < 0.8)] = 0
                label_t1_1[(label_t1_1 >= 0.8)] = 1  

                fea1_1 = label_t1_1 * adv_t  # 1 12 32 64 64 

                label_t_0 = output3[:, 0, :, :, :]  

                label_t_0[label_t_0 >= 0.5] = 1  
                label_t_0[label_t_0 < 0.5] = 0
                fea_0 = label_t_0 * adv_t  # 1 12 32 64 64
                fea_0 = fea_0.reshape(12, 32 * 64 * 64)
                mean_0 = t.mean(fea_0, dim=1)  # 12 1    
                mean_0 = mean_0.reshape(1, 12, 1, 1, 1)

                label_t1_0 = output4[:, 0, :, :, :]  

                label_t1_0[(label_t1_0 < 0.8)] = 0
                label_t1_0[(label_t1_0 >= 0.8)] = 1  

                fea1_0 = label_t1_0 * adv_t  # 1 12 32 64 64  
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

                w_0 = t.exp(-dis_00) / (t.exp(-dis_00) + t.exp(-dis_01))  # 1 1 32 64 64
                w_0 = w_0 * label_t1_0  # 1 1 32 64 64

                w_1[(w_1 >= 0.5) ] = 1
                w_1[(w_1 < 0.5) ] = 2

                w_0[(w_0 < 0.5)] = -1
                w_0[(w_0 >= 0.5)] = 0
                w_0[(w_0 == -1)] = 2

                label_w = w_1 * w_0
                label_w[label_w == 2] = 1
                label_w[label_w == 4] = 2
                label_w = label_w.data.cpu().data.numpy()
                # print(label_w)

                label_w = label_w.reshape(32, 4096)
                data1 = pd.DataFrame(label_w)
                data1.to_csv('pseudo-labels_t' + str(i ) + '.csv')

            if epoch >= N:
                with open(str('pseudo-labels_t') + str(i ) + '.csv') as file_name:
                    file_read = csv.reader(file_name)
                    array1 = list(file_read)
                array1 = np.array(array1)
                # b = np.zeros(256)
                wt = array1[1:, 1:]
                wt = wt.reshape(1, 32, 64, 64).astype(np.float64)
                wt = t.LongTensor(wt).cuda(1)


            for param in model_D1.parameters():
                param.requires_grad = False


            source_D=t.ones( 1 , 1 , 1 , 2 , 2 ).detach().cuda(1)   
            target_D = t.zeros( 1 , 1, 1, 2 , 2 ).detach().cuda(1)  

            D_out_s = model_D(predict_y_s_d)
            D_out_as = model_D1(A_s)

            D_out_t = model_D(predict_y_t_d)
            D_out_at = model_D1(A_t)


            if epoch <N:
                loss_1 = criterion(predict_y_s, true_y) #+ 0.01*criterion1(predict_y_t, wt)
            else:
                loss_1 = criterion(predict_y_s, true_y)   + beta*criterion1(predict_y_t, wt)
            loss = loss_1
            loss_batch_G += loss
            D_loss1 = criterion(D_out_s, source_D)+ criterion(D_out_t, target_D)
            loss_adv =  criterion(D_out_at, source_D)
            D_loss =D_loss1+alpha*loss_adv
            loss_batch+=loss+D_loss

            epoch_loss += loss.data


            if i % 8 == 0:
                optimizer1.zero_grad()  
                optimizer2.zero_grad()  
                loss_batch.backward()
                optimizer1.step()
                optimizer2.step()

                print('======> Epoch_train: {}, iter_train: {}/{}, loss_G: {:.10f}'.format(epoch, i, img_num,
                                                                                           loss.data ))
               
                loss_batch_G = 0
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
                optimizer3.zero_grad()  
                loss_batch_D.backward()

                optimizer3.step()
                loss_batch_D=0


        scheduler1.step()
        scheduler2.step()
        scheduler3.step()


  
        pret_s = precision(predict_y_np_s, gt1)
        rect_s = recall(predict_y_np_s, gt1)
        f1t_s = F1(predict_y_np_s, gt1)






        epoch_loss /= img_num  
        train_loss.append(epoch_loss)
        loss_summury[epoch][0] = epoch_loss * 10000





        gt = []
        val_predict_y_np = []
        val_predict_y_np1 = []
        val_list = os.listdir(mantage3)
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
            epoch_loss , pret_s, rect_s, f1t_s)
        f = open(r'train_processing\646432_unet3D_source.txt', 'a')
        print(s, file=f)
        f.close()

        stistic_data = pd.DataFrame(loss_summury)
        writer = pd.ExcelWriter(r'losses\model_193742_1128.xls')  
        stistic_data.to_excel(writer, float_format='%.3f')  
        writer.save()
        writer.close()






        s = '======> Epoch_train: {}, iter_train: {}/{}, loss_train: {:.10f}, pre: {:.10f}, recall: {:.10f}, F1: {:.10f}'.format(
            epoch, epoch, epochs,
            val_epoch_loss , pre_v, rec_v, f1_v)
        f = open(r'train_processing\646432_unet3D_val.txt', 'a')
        print(s, file=f)
        f.close()



        t.save(model.state_dict(), dir_checkpoint + '/' + 'CP{}.pth'.format(epoch))  
        print('Checkpoint {} saved !'.format(epoch))

    return train_loss, val_loss


if __name__ == '__main__':
    #train source
    mantage_path1 = train_path + '/' + 'Hippocampus-CA1' + '/' + 'mont'
    foreground_label_path1 = train_path + '/' + 'Hippocampus-CA1' + '/' + 'label'
    background_label_path1 = train_path + '/' + 'Hippocampus-CA1' + '/' + 'back128'
   #train target
    mantage_path2 = train_path + '/' + 'LSL-H2B-GFP' + '/' + 'mont'
    foreground_label_path2 = train_path + '/' + 'LSL-H2B-GFP' + '/' + 'label'
    background_label_path2 = train_path + '/' + 'LSL-H2B-GFP' + '/' + 'back128'

   #test target
    mantage_path3 = train_path + '/' + 'LSL-H2B-GFP' + '/' + 'mont'
    foreground_label_path3 = train_path + '/' + 'LSL-H2B-GFP' + '/' + 'label'
    background_label_path3 = train_path + '/' + 'LSL-H2B-GFP' + '/' + 'back128'

    seed=init_random_seed(seed)

    t.cuda.set_device(1)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    model = unet3D.UnetGenerator_3d(in_dim=1, out_dim=2, num_filter=4).cuda()


    train_loss, val_loss = train_net(model, epochs=1500, mantage1=mantage_path1,mantage2=mantage_path2,mantage3=mantage_path3, foreground_label1=foreground_label_path1,
                                     background_label1=background_label_path1, foreground_label2=foreground_label_path2,background_label2=background_label_path2,foreground_label3=foreground_label_path3,background_label3=background_label_path3,
                                     )




