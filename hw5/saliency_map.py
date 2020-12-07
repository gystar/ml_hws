# saliency map的实现
# 基于hw3中的letnet5训练好的model来实现
# 思路：求出输入图片的梯度，将梯度矩阵用一定的方式转换为图像矩阵显示出来
import math
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt


def tensor2numpy(t):
    # 将tensor反解析为图像矩阵,方便画出原图
    tansformer = transforms.ToPILImage()
    return np.array(tansformer(t))#因为plt只能识别numpy进行画图，所以我们要将tensor强制转换为numpy


def grad2numpy(grad):
    # 将图片梯度转换为矩阵输出,#为方便起见，可将其先“标准化”再输出；方便画出对比图
    x = (grad - grad.min()) / (grad.max() - grad.min())#对梯度标准化，得到的是tensor
    x = x.permute(1, 2, 0).numpy()#对tensor求转置：permute；强制转为numpy矩阵(因为plt只能识别numpy矩阵进行画图)
    return x


# 定义函数：图片某一区域最整体结果影响最大——以图片梯度最大的区域为依据
def draw(model, images, labels, fontsize=15):
    # 引用model,求出预测值；定义损失函数；使用反向传播更新参数
    images.requires_grad = True#计算机默认不会对images求梯度，所以这里我们要求对images进行指定操作：求梯度
    y_pred = model(images)#以images为输入计算预测值y_pred
    Loss_func = torch.nn.CrossEntropyLoss()#计算对整体影响最大的那一区域的grad
    Loss = Loss_func(y_pred, labels)#套用Loss_func函数的计算，这两行可以联合写成：Loss = torch.nn.CrossEntropyLoss()(y_pred, labels)
    Loss.backward()#反向传播计算Loss的参数，不需要写为：Loss = Loss.backward(),这样反而会覆盖原来的Loss

#关于函数顺序的疑惑：
#函数draw()会调用函数grad2numpy()；通过draw函数求出想要的梯度区域，再使用grad2numpy函数输出矩阵；使用plt进行画图

    num = images.shape[0]  # 图片数量

    # 所有图像的梯度
    images_grad = images.grad.detach().cpu()

    # 绘制图片
    fig = plt.figure(figsize=(8, 8))
    cols = 4  # 绘制的列数
    for i in range(num):
        # 原图像和显著图呈上下位置分布，计算出他们在图像中的位置
        row, col = math.floor(i / cols), i % cols
        pos1 = row * cols * 2 + col + 1  # 原图像在大图中的位置，即第几个子图
        pos2 = pos1 + cols  # 对应的显著图在大图中的位置，即第几个子图
        # 先画原图
        ax = fig.add_subplot(3 * 2, 4, pos1, xticks=[], yticks=[])  # 添加原图像的子图
        ax.imshow(tensor2numpy(images[i]))
        ax.set_title("Image %s" % str(i), fontsize=fontsize, color="r")  # 设置标题
        # 再画出显著图
        ax = fig.add_subplot(3 * 2, 4, pos2, xticks=[], yticks=[])  # 添加显著图的子图
        ax.imshow(grad2numpy(images_grad[i]))
        ax.set_title("saliency %s" % str(i), fontsize=fontsize, color="r")


##test
if __name__ == "__main__":
    import sys
    import importlib

    sys.path.append("../hw3")  ##直接使用hw3中的model
    import image_classification
    import image_set

    importlib.reload(image_set)
    importlib.reload(image_classification)
    # 指定模型类别
    model_class = image_classification.GYHF_LetNet5
    data = image_set.LearningSet("../hw3/data/training", model_class.input_size, False)
    # 使用已经训练好的hw3中的model
    model = torch.load(
        "../hw3/<class 'image_classification.GYHF_LetNet5'>.pkl",
        map_location=torch.device("cpu"),
    )
    model = model.cpu()
    images, lables = data.GetBatch([0, 1, 2])
    draw(model, images, lables, 7)