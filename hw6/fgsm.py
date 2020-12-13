"""
Author: gystar
Date: 2020-12-07 19:11:58
LastEditors: gystar
LastEditTime: 2020-12-09 15:02:49
FilePath: /ml_hws/hw6/fgsm.py
Description:
    Fast Gradient Sign Method(FGSM)     
    FGSM方式的non-targeted的attack实现
    non-targeted:在约束条件下和真实label越远越好
    non-targeted:在约束条件下和真实label越远越好,同时和目标label越近越好
    在原图片的基础上，对于损失函数求图片的梯度，梯度大于0则加上最大偏移值e，否则减去e
    一次方向传播再修正即可得到目标图片
"""
import torch
import random
import utils
import multiprocessing


"""
description: 白盒non-targeted的FGSM attack的实现
param {0} model
param {1} image
param {2} label
param {3} tolerance：允许修改的范围
return {*}：(生成的攻击图片,(原图的预测的标签，预测的概率，真实标签的概率),(生成图片的预测的标签，预测的概率，真实标签的概率))
"""


def white_nontarget_attack(model, image, label, tolerance):
    model.eval()  # 预测的时候一定要用此函数
    image = image.unsqueeze(0)#对原来的image增加维度batch，并放在第一维（0）
    label = torch.tensor([label]).to(image.device)#将label转化为tensor并且放在image同样的位置
    softmax = torch.nn.functional.softmax  # 预测结果需要转化为概率方便观察
    image.requires_grad =  True#要求对image进行求梯度
    y = model(image)#调用model计算预测值y
    Loss_func = torch.nn.CrossEntropyLoss()#定义损失函数框架，注意，crossentropy已经有softmax，所以不需要对y进行softmax
    Loss = -1*Loss_func(y,label)#想要得到最不可能的结果，所以在原来的损失函数前加“负号”
    Loss.backward() #反向传播，为了能够计算出梯度
    image = image-tolerance*image.grad.sign()#对原来的input图片进行修饰，一步跨到最大限度（对计算出来的梯度使用sign符号函数，结果要么是正的最梯度容忍值，要么是最负容忍值）
    y = softmax(y.squeeze(0))#输出y除去batch维度（0）的概率（softmax）
    y1 = softmax(model(image).squeeze(0))#对比输出修正后的y（除去第一维度batch）的概率

    
    # 将结果保存到cpu上释放显存
    y, y1 = y.detach().cpu(), y1.detach().cpu()
    # 转化为在cpu上的图像矩阵
    image = utils.tensor2numpy(image.squeeze(0).detach().cpu())
    # 预测的标签，预测的概率，真实标签的概率
    info = (y.topk(1)[1], y.topk(1)[0], y[label])
    # 预测的标签，预测的概率，真实标签的概率
    info_new = (y1.topk(1)[1], y1.topk(1)[0], y1[label])

    return (
        image,  # 生成的攻击图片
        info,  # 原图的输出结果（预测的标签，预测的概率，真实标签的概率）
        info_new,  # 攻击图片的输出结果（预测的标签，预测的概率，真实标签的概率）
    )


###test
if __name__ == "__main__":
    import data_set
    import importlib
    import numpy as np
    import torchvision.models as models
    import matplotlib.pyplot as plt
    import utils

    importlib.reload(data_set)

    data = data_set.ImageSet()

    model = models.vgg16(pretrained=True)
    image, lable = data.__getitem__(6)
    y = torch.nn.functional.softmax(model(image.unsqueeze(0))[0])
    print(y[lable])
    y = torch.nn.functional.softmax(model(image.unsqueeze(0))[0])
    print(y[lable])
    rimage, (a1, a2, a3), (b1, b2, b3) = white_nontarget_attack(model, image, lable, 0.001)

    print('label is %d "%s"' % (lable, data.category_names[lable]))
    print('origin:\nprediction is %d(%f) "%s", \nlabel probability: %f' % (a1, a2, data.category_names[a1], a3))
    plt.figure()
    plt.imshow(utils.tensor2numpy(image))
    plt.show()
    print('attack result:\nprediction is %d(%f) "%s", \nlabel probability: %f' % (b1, b2, data.category_names[b1], b3))
    plt.figure()
    plt.imshow(rimage)
    plt.show()
