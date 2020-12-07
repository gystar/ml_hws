'''
Author: your name
Date: 2020-12-06 20:34:52
LastEditTime: 2020-12-07 16:54:33
LastEditors: your name
Description: In User Settings Edit
FilePath: /ml_hws/hw6/data_set.py
'''
# 加载图片数据的实现
import torch
import os
from torch import random
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd


# 用来进行训练的train、validation图片集加载类
# 图片的命名为"类别_名称.jpg"

# 定义图片的的读取方式（dataset）

class ImageSet(torch.utils.data.Dataset):
    def __init__(self):
        self.labels = np.array(list(pd.read_csv("./data/labels.csv").loc[:, "TrueLabel"]))
        self.category_names = np.array(list(pd.read_csv("./data/categories.csv").loc[:,"CategoryName"]))
        self.images_dir = './data/images'
        images = os.listdir('./data/images')
        images.sort() #按照名字排序
        self.images = np.array(images)  
        self.transfomer = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),            
            ])

        return None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = os.path.join(self.images_dir, self.images[index])
        image = self.transfomer(Image.open(path))
        label = self.labels[index]
        return image, label
    
    def GetBatch(self, idx):
        #根据ID取出一批图片的数据
        images = []
        labels = [] 
        for j in idx:
            image, label = self.__getitem__(j)
            images.append(image.numpy())
            labels.append(label)

        return torch.tensor(images), torch.tensor(labels)


# test
if __name__ == "__main__":
    data = ImageSet()
    image,label = data.__getitem__(0)
    images,labels = data.GetBatch([0,1,2])
    
    
