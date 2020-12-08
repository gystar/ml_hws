"""
Author: gystar
Date: 2020-12-08 09:18:21
LastEditTime: 2020-12-08 09:31:38
LastEditors: Please set LastEditors
Description: 常用实现
FilePath: /ml_hws/hw6/utils.py
"""
from torchvision import transforms
import numpy as np


def tensor2numpy(t):
    # 将tensor反解析为[w,h,c]形式的图像矩阵
    tansformer = transforms.ToPILImage()
    return np.array(tansformer(t))
