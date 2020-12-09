"""
Author: gystar
Date: 2020-12-09 15:46:37
LastEditors: gystar
LastEditTime: 2020-12-09 15:46:37
FilePath: /ml_hws/hw6/bim.py
Description: 
    Basic Iterative Methods(BIM)
    参考文献：https://arxiv.org/abs/1607.02533
    BIM通过多次迭代，使用较小的步长来慢慢改变损失函数
    邻域限定函数：clip(x1) = min{255, x+e, max{0,x-e, x1}}
    迭代函数：x(0) = x, x(k+1) = clip{ x(k) + a*sign(grad(x(k)))}
    迭代停止条件：到达边界
"""

import torch
import numpy as np
import utils
import random

"""
description: nontarget的白盒BIM attck实现
param {*} model
param {*} image
param {*} label
param {*} tolerance:每个像素允许改变的幅度
param {*} step_percent:每次更新的幅度系数，步长=step_percent*tolerance
param {*} epochs：训练轮次
return {*}(生成的攻击图片,(原图的预测的标签，预测的概率，真实标签的概率),(生成图片的预测的标签，预测的概率，真实标签的概率))
"""


def white_nontarget_attack(model, image, label, tolerance, step_percent=0.1, epochs=10):
    step = tolerance * step_percent  # 每次更新的大小
    image = image.unsqueeze(0)
    softmax = torch.nn.functional.softmax  # 预测结果需要转化为概率方便观察

    y0 = model(image).squeeze(0)
    y = None
    # [0,1]为transforms.ToTensor之后的取值范围
    min_value = torch.tensor(0).float().to(image.device)
    max_value = torch.tensor(1.0).float().to(image.device)
    floor = torch.where(image - tolerance < min_value, min_value, image - tolerance)
    ceiling = torch.where(image + tolerance > max_value, max_value, image + tolerance)
    for _ in range(epochs):
        # 会计算输入图片矩阵的导数
        # 下面会覆盖imgae的内容，所以需要每次声明一次，同时梯度就不需要每次清0
        image.requires_grad = True
        y = model(image).squeeze(0)
        loss = -1 * y[label]
        loss.backward()
        # 邻域限定函数：clip(x1) = min{max_value, x+e, max{min_value,x-e, x1}}
        # 下面与clip的定义等价
        image = image + step * image.grad.sign()
        image = torch.where(image < floor, floor, image)
        image = torch.where(image > ceiling, ceiling, image).detach()  # 若不detach下一次循环会报错

    y = model(image).squeeze(0)
    # 将结果解析一下，并存入cpu释放返回
    y0, y = softmax(y0).detach().cpu(), softmax(y).detach().cpu()
    # 转化为在cpu上的图像矩阵
    image = utils.tensor2numpy(image.squeeze(0).detach().cpu())
    # 预测的标签，预测的概率，真实标签的概率
    info = (y0.topk(1)[1], y0.topk(1)[0], y0[label])
    # 预测的标签，预测的概率，真实标签的概率
    info_new = (y.topk(1)[1], y.topk(1)[0], y[label])

    return (
        image,  # 生成的攻击图片
        info,  # 原图的输出结果（预测的标签，预测的概率，真实标签的概率）
        info_new,  # 攻击图片的输出结果（预测的标签，预测的概率，真实标签的概率）
    )


###test
if __name__ == "__main__":
    random.seed(100)
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
    rimage, (a1, a2, a3), (b1, b2, b3) = white_nontarget_attack(model.cuda(), image.cuda(), lable, 0.001)

    print('label is %d "%s"' % (lable, data.category_names[lable]))
    print('origin:\nprediction is %d(%f) "%s", \nlabel probability: %f' % (a1, a2, data.category_names[a1], a3))
    plt.figure()
    plt.imshow(utils.tensor2numpy(image))
    plt.show()
    print('attack result:\nprediction is %d(%f) "%s", \nlabel probability: %f' % (b1, b2, data.category_names[b1], b3))
    plt.figure()
    plt.imshow(rimage)
    plt.show()
