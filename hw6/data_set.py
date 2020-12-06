# 加载图片数据的实现
import torch
import os
from torch import random
from torchvision import transforms
from PIL import Image
from torchvision.transforms.transforms import CenterCrop
import numpy as np
import pandas as pd


# 用来进行训练的train、validation图片集加载类
# 图片的命名为"类别_名称.jpg"

# 定义图片的的读取方式（dataset）

class LearningSet(torch.utils.data.Dataset):
    def __init__(self):
        self.labels = pd.read_csv("./data/labels.csv").loc[:, "TrueLabel"]
        self.images_dir = './data/images'
        self.images = os.listdir('./data/images')

        return None

    def __len__(self):
        return len(self.images)

    def __getitem__(self):
        return None


# test
if __name__ == "__main__":
