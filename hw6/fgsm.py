"""
Author: gystar
Date: 2020-12-07 10:17:04
LastEditTime: 2020-12-07 17:20:41
LastEditors: Please set LastEditors
Description:
    FGSM方式的non-targeted的attack实现
    non-targeted:在约束条件下和真实label越远越好
    non-targeted:在约束条件下和真实label越远越好,同时和目标label越近越好
    在原图片的基础上，对于损失函数求图片的梯度，梯度大于0则加上最大偏移值e，否则减去e
    一次方向传播再修正即可得到目标图片
FilePath: /ml_hws/hw6/fgsm.py
"""
import torch
from torchvision import transforms
import random
import numpy as np


def tensor2numpy(t):
    # 将tensor反解析为图像矩阵
    tansformer = transforms.ToPILImage()
    return np.array(tansformer(t))


def attack(model, image, label, tolerance):
    random.seed(10)
    image = image.unsqueeze(0)

    softmax = torch.nn.functional.softmax
    # 会计算输入图片矩阵的导数
    image.requires_grad = True
    y = softmax(model(image).squeeze(0))
    # 预测的标签，预测的概率，真实标签的概率
    info = (y.topk(1)[1], y.topk(1)[0], y[label])
    # 损失函数：使对真实值label的预测尽可能小
    loss = -1 * y[label].squeeze(0)
    loss.backward()
    # 更新输入的图片
    update = torch.zeros(image.grad.shape, device=image.device)
    update[image.grad > 0] = tolerance
    update[image.grad < 0] = -1 * tolerance

    image_new = image.clone() + update
    # 再预测一次
    y1 = softmax(model(image_new).squeeze(0))
    # 预测的标签，预测的概率，真实标签的概率
    info_new = (y1.topk(1)[1], y1.topk(1)[0], y1[label])

    return (
        tensor2numpy(image_new.detach().cpu().squeeze(0)),  # 生成的攻击图片
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

    importlib.reload(data_set)
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    data = data_set.ImageSet()

    a = torch.tensor([[1, -1], [1, 2]])
    b = torch.where(a > 0, 1, -1)

    image, lable = data.__getitem__(5)
    rimage, (a1, a2, a3), (b1, b2, b3) = attack(alexnet, image, lable, 0.01)

    print('label is %d "%s"' % (lable, data.category_names[lable]))
    print(
        'origin:\nprediction is %d(%f) "%s", \nlabel probability: %f'
        % (a1, a2, data.category_names[a1], a3)
    )
    plt.figure()
    plt.imshow(tensor2numpy(image))
    plt.show()
    print(
        'attack result:\nprediction is %d(%f) "%s", \nlabel probability: %f'
        % (b1, b2, data.category_names[b1], b3)
    )
    plt.figure()
    plt.imshow(rimage)
    plt.show()
