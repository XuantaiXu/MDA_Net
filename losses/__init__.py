import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

def WeakSupervisonLoss(predict_y, foreground, background):
    """
    :param predict_y:    tensor
    :param foreground:  标记的label  tensor
    :param background:  随机选择20%的背景像素点作为背景     tensor
    :return: 
    """
    # predict_y = np.squeeze(predict_y.data.cpu().numpy())
    foreground = np.squeeze(foreground.data.cpu().numpy())
    background = np.squeeze(background.data.cpu().numpy())      # predict_y有tensor转为numpy

    background_ = np.zeros_like(background).astype(int)
    ind_background = np.where(background == 1)
    num = len(ind_background[0])
    # print("num:", num)
    random_indices = np.random.randint(0, len(ind_background[0]), size=int(num*0.01))
    # print("random_indices", len(random_indices))
    # print(random_indices)
    background_[ind_background[0][random_indices], ind_background[1][random_indices], ind_background[2][random_indices]] = 1  # 标记背景
    # ind_b = np.where(background_ == 1)
    background_ = foreground + background_
    background_[background_>1] = 1
    background_ = torch.tensor(background_, requires_grad=True)
    background_ = background_.view(-1).cuda()
    predict_y = predict_y.view(-1)
    predict_y = predict_y.type(torch.DoubleTensor).cuda()
    # print(background_.type())

    predict_y = torch.mul(predict_y, background_).cuda()

    return predict_y

def Getmask(predict_img, foreground, background):
    """
    :param predict_y:    tensor
    :param foreground:  标记的label
    :param background:  随机选择20%的背景像素点作为背景
    :return: 
    """
    # foreground = np.squeeze(foreground.data.cpu().numpy())
    # background = np.squeeze(background.data.cpu().numpy())  # predict_y有tensor转为numpy
    predict_y = np.squeeze(predict_img.data.cpu().numpy())
    mask = np.zeros_like(foreground)
    mask[foreground == 1] = 1
    mask[background == 1] = 1
    # print(mask)
    predict_y[mask == 0] = 0                  # 使用模糊像素点label和预测的值
    predict_y_ = torch.tensor(predict_y, requires_grad=True)
    true_y = torch.tensor(foreground, requires_grad=False)
    # print(predict_y_)
    # print(true_y)
    predict_y_ = predict_y_.type(torch.FloatTensor).cuda()
    true_y = true_y.type(torch.FloatTensor).cuda()


    return predict_y_, true_y

def Getmask_1(predict_img, foreground, background):
    """
    :param predict_y:    tensor
    :param foreground:  tensor
    :param background:  tensor
    :return:
    """
    foreground = np.squeeze(foreground.data.cpu().numpy())
    background = np.squeeze(background.data.cpu().numpy())  # predict_y有tensor转为numpy

    mask = foreground+background
    mask = torch.tensor(mask, requires_grad=True).cuda()
    # predict_y[mask < 0.01] = 0
    predict_y = torch.mul(predict_img, mask).cuda()

    return predict_y, foreground

def Getmask_2(predict_img, true_y, foreground, background):
    """
    :param predict_y:    tensor
    :param foreground:  tensor
    :param background:  tensor
    :return:
    """

    mask = foreground + background
    # print("mask:  ", mask)
    predict_y = torch.mul(predict_img, mask).cuda()
    true_y = torch.mul(true_y, mask)
    # print(predict_y)
    return predict_y, true_y

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_Loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_Loss = F.binary_cross_entropy(inputs, targets, reduce=False)

        pt = torch.exp(-BCE_Loss)
        F_loss = self.alpha * (1-pt) ** self.gamma * BCE_Loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class CELDice:
    def __init__(self, dice_weight=0.2,num_classes=2):
        #self.BCE_Loss = nn.BCELoss()
        self.CEL_Loss = nn.CrossEntropyLoss()
        self.jaccard_weight = dice_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
       loss = (1 - self.jaccard_weight) * self.CEL_Loss(outputs, targets)
       if self.jaccard_weight:
           eps = 1e-15
           for cls in range(self.num_classes):
               jaccard_target = (targets == cls).float()
               jaccard_output = outputs[:, cls].exp()
               intersection = (jaccard_output * jaccard_target).sum()
               union = jaccard_output.sum() + jaccard_target.sum()
               loss -= torch.log((2*intersection + eps) / (union + eps)) * self.jaccard_weight
       return loss

if __name__=="__main__":
    predict_y = np.random.randint(0, 2, (4, 4))
    true_y = np.random.randint(0, 2, (4, 4))

    foreground = np.zeros((4, 4)).astype(int)
    foreground[1:3, 1:3] = 1              # 前景
    ind = np.where(foreground==0)
    # print(len(ind[0]))
    random_indices = np.random.randint(0, len(ind[0]), size=2)
    # print(random_indices)
    # print(ind)
    # print(ind[0][random_indices], ind[1][random_indices])
    foreground[ind[0][random_indices], ind[1][random_indices]] = 1    # 标记背景
    print("foreground\n", foreground)
    print("predict_y\n",predict_y)
    predict_y = predict_y * foreground
    print("predict_y\n",predict_y)
    true_y = true_y * foreground
