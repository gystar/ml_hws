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
    image = image.unsqueeze(0)

    softmax = torch.nn.functional.softmax  # 预测结果需要转化为概率方便观察
    # 会计算输入图片矩阵的导数
    image.requires_grad = True
    y = softmax(model(image).squeeze(0))
    # 损失函数：使对真实值label的预测尽可能小
    loss = -1 * y[label].squeeze(0)
    loss.backward()

    # 更新输入的图片
    image = image.detach().clone() + image.grad.sign() * tolerance
    # 用生成的图片进行预测
    y1 = softmax(model(image).squeeze(0))

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
