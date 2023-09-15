import numpy as np
import torch
from  torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn

class Poly(_LRScheduler):
    # 传入优化器，迭代次数，num_epochs=len(train_loader)
    def __init__(self, optimizer, num_epochs, iters_per_epoch=0, warmup_epochs=0, last_epoch=-1):
        self.iters_per_epoch = iters_per_epoch
        self.cur_iter = 0
        self.N = num_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
        super(Poly, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        factor =  pow((1 - 1.0 * T / self.N), 0.9)  #学习率迭代的公式
        if self.warmup_iters > 0 and T < self.warmup_iters:
            factor = 1.0 * T / self.warmup_iters

        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1
        return [base_lr * factor for base_lr in self.base_lrs]

def eval_metrics(output, target, num_class):
    _, predict = torch.max(output.data, 1) # 按通道维度取最大,拿到每个像素分类的类别(1xhxw)
    predict = predict + 1 # 每个都加1避免从0开始,方便后面计算PA
    target = target + 1

    labeled = (target > 0) * (target <= num_class)  # 得到一个矩阵，其中，为true的是1，为false的是0
    # 标签中同时满足大于0 小于num_classes 的地方为T,其余地方为F  构成了一个蒙版
    correct, num_labeled = batch_pix_accuracy(predict, target, labeled)  #计算一个batch中预测正确像素点的个数和所有像素点的总数

    inter, union = batch_intersection_union(predict, target, num_class, labeled)
    return [np.round(correct, 5), np.round(num_labeled, 5), np.round(inter, 5), np.round(union, 5)]


def batch_pix_accuracy(predict, target, labeled):
    pixel_labeled = labeled.sum()   # 计算标签的总和，是一个batch中的所有标签的总数
    # 注意  python中默认的T为1 F为0  调用sum就是统计正确的像素点的个数
    pixel_correct = ((predict == target) * labeled).sum() # 将一个batch中预测正确的，且在标签范围内的像素点的值统计出来
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

def batch_intersection_union(predict, target, num_class, labeled):
    predict = predict * labeled.long()  # 返回预测中在指定范围内的像素点
    intersection = predict * (predict == target).long() # 过滤掉预测中不正确的像素值
    #   一个batch中只有正确的像素值才在intersection中，不正确的为0
    #torch.histc 统计图片中的从0-bins出现的次数，返回一个列表
    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    #  area_inter 会得到batch中每个类别 对应像素点(分类正确的)出现了多少次
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    #  area_inter 将batch中预测的所有像素点(不管正不正确) 在每个类别的次数统计出来
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    #  area_lab 将batch中每个类别实际有多少像素点统计出来
    area_union = area_pred + area_lab - area_inter  # 预测与标签相交的部分 每个类别对应像素点的数量
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()


# 初始化网络权重
def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
