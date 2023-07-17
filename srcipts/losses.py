import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from . import base
from . import functional as F
import torch.nn.functional as fuc
from ..base.modules import Activation
from torch.autograd import Variable
import torch
import torch.nn.functional as FS


def bbox_CIoU(box1, box2, x1y1x2y2=True, eps=1e-7):
    """用计算回归损失
    :params box1: 预测框      shape:(4, n)
    :params box2: 预测框      shape:(n, 4)
    :return box1和box2的CIoU   shape:(1, n)
    """
    box2 = box2.T  # 转换为(4, n)
    box1 = box1.T
    # Get the coordinates of bounding boxes
    # 1. 转换为xyxy
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    # 2. 计算交集区域面积
    # Intersection area   tensor.clamp(0): 将矩阵中小于0的元数变成0
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    # 3. 计算并集区域面积
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    # 4. 计算iou
    iou = inter / union

    # ---开始计算CIoU---
    # 两个框的最小闭包区域的width
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)

    # 两个框的最小闭包区域的height
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

    # 勾股定理计算出最小闭包区域的对角线距离的平方
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared

    # 计算两个框中线点的距离的平方（先分别计算出两个box的中心坐标，然后求即可）
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared

    # 将长宽比因素考虑进去
    import math
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    # 计算alpha
    alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha).T # CIoU

class Diou(base.Loss):

    def __init__(self, eps=1.0, activation=None, ignore_channels=None, **kwargs):
        super(Diou, self).__init__(**kwargs)

    def forward(self,b_pr, b_gt):
        loss = bbox_CIoU(b_pr,b_gt)
        return loss
class JaccardLoss(base.Loss):
    def __init__(self, eps=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(base.Loss):
    def __init__(self, eps=1.0, beta=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )
    
class AutomaticWeightedLoss(base.Loss):
    """automatically weighted multi-task loss
    Params：str
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class BCEFocalLoss(base.Loss):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean',**kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


        self.gamma = gamma
class MultiCEFocalLoss(base.Loss):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean',**kwargs):
        super().__init__(**kwargs)
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num =  class_num

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1) # softmmax获取预测概率
        class_mask = F.one_hot(target, self.class_num) #获取target的one hot编码
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)] # 注意，这里的alpha是给定的一个list(tensor
#),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
# 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class KLLoss(base.Loss):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, p1, p2):
        # pred1 = fuc.sigmoid(p1)
        # pred2 = fuc.sigmoid(p2)
        # loss = fuc.kl_div(p1.log(),p2,reduction='batchmean')
        loss1= torch.mean(torch.sum(p1 * torch.log(1e-8 + p1 / (p2 + 1e-8)),0))
        loss2= torch.mean(torch.sum(p2 * torch.log(1e-8 + p2 / (p1 + 1e-8)), 0))
        loss = ( loss1+ loss2) / 2
        return  loss
    
class KLLoss_label(base.Loss):
    def __init__(self):
        super(KLLoss_label, self).__init__()

    def forward(self, p1, p2):
        p1=fuc.softmax(p1,dim=-1)
        p2=fuc.softmax(p2,dim=-1)

        # loss = fuc.kl_div(p1.log(), p2, reduction='batchmean')
        loss1 = torch.mean(torch.sum(p2* torch.log(1e-8 + p2 / (p1 + 1e-8)),0))
        loss2 = torch.mean(torch.sum(p1 * torch.log(1e-8 + p1 / (p2 + 1e-8)), 0))
        # loss1 = fuc.kl_div(p1.log(),p2,reduction= 'mean')
        # loss2 = fuc.kl_div(p2.log(), p1, reduction='sum')
        #
        loss = ( loss1+loss2) / 2
        return  loss
class L1Loss(nn.L1Loss, base.Loss):
    pass



class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass




class HausdorffLoss(base.Loss):
    def __init__(self, p=2):
        super(HausdorffLoss, self).__init__()
        self.p = p

    def torch2D_Hausdorff_distance(self, x, y, p=2):  # Input be like (Batch,1, width,height) or (Batch, width,height)
        x = x.float()
        y = y.float()
        distance_matrix = torch.cdist(x, y, p=p)  # p=2 means Euclidean Distance

        value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
        value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]

        value = torch.cat((value1, value2), dim=1)

        return value.max(1)[0].mean()

    def forward(self, x, y):  # Input be like (Batch,height,width)
        loss = self.torch2D_Hausdorff_distance(x, y, self.p)
        return loss
