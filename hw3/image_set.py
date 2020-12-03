# 加载图片数据集的实现
# 一个set加载一个文件夹中的所有图片
import torch
import os
from torch import random
from torchvision import transforms
from PIL import Image
from torchvision.transforms.transforms import CenterCrop
import numpy as np


# 用来进行训练的train、validation图片集加载类
# 图片的命名为"类别_名称.jpg"

#定义图片的的读取方式（dataset）

class LearningSet(torch.utils.data.Dataset):#继承父类，命名为LearingsSet,Dataset表示一种框架
    #init表示初始化；self表示具体对象；dir:路径；
    # True：用来区分training（翻转增加training集合数量）集合和validation集合；最后一个参数的""="号，表示默认参数
    def __init__(self, dir, ouput_size, is_train=True):  # ouput_size：输出的每一张图的大小，init表示初始化
        self.dir = dir#self只是固定格式，会自动生成
        self.ouput_size = ouput_size

        # 将文件名和对应类别获取并存到一个列表中
        names = os.listdir(self.dir)
        labels = [int((name.split('_'))[0]) for name in names]
        self.images = [names, labels]
        #相当于int[]定义了一个函数，表示映射：names->int[]
        # 设置训练时图片转换器，转换成tensor,增加一些变换操作以增加学习集
        transformer_train = transforms.Compose([#使用transforms包中的方法（Compose）生成转换器    
            transforms.Resize(ouput_size),    #重设大小        
            transforms.RandomHorizontalFlip(),  # 随机水平翻转图片
            transforms.RandomRotation(15),  # 随机旋转图片
            transforms.ToTensor(),   #转换成tensor(其中前面加To,只是固定表达方式）            
        ])

        # 进行验证时，使用图片转换器，转换成tensor，不需要对validation做翻转等操作
        transformer_validation = transforms.Compose([
            transforms.Resize(ouput_size),  #重设大小
            transforms.ToTensor(),#转换成Tensor            
        ])

        # 根据is_train选择转换器
        self.transformer = transformer_train if is_train else transformer_validation

    def __len__(self):  #_len_ &_getitem_这两个函数是dataset要求的两个函数
        return len(self.images[0])#返回函数值为self.images的第一个分量的长度，也就是name的个数

    def GetLen(self):   #自定义函数调用上一个定义的_len_函数
        return self.__len__()#与上一个函数无实质性差别，只是写法不同

    def __getitem__(self, index):
        # 打开一个index对应的图片，并转换为tensor返回，同时返回label
        path, label = self.images[0][index], self.images[1][index]
        #用self.images[0][index]表示文件名path； self.images[1][index]表示分类label
        path = os.path.join(self.dir, path)  #os.path.join表示连接文件夹self.dir和文件名path两个路径    
        image = self.transformer(Image.open(path).convert('RGB'))
        #打开path中的jpg文件并使用转换器将其变为Tensor
        return image, label    

    # 获取类别的数量
    def GetClassNum(self):
        return max(self.images[1])-min(self.images[1])+1
    
    def GetBatch(self, idx  : list):
        #根据ID取出一批图片的数据
        images = np.zeros((1, 3,self.ouput_size[0], self.ouput_size[1])) #空矩阵，便于使用concatenate
        labels = np.zeros((1,)) #空矩阵，便于使用concatenate  
        for j in idx:
            image, label = self.__getitem__(j)
            images = np.concatenate((images, image.unsqueeze(0)), axis = 0)
            labels = np.concatenate((labels, np.array([label])), axis = 0)
        images = torch.from_numpy(images[1:]).float() #去掉第一行空行
        labels = torch.from_numpy(labels[1:]).long()#去掉第一行空行
        return images, labels

# 用来进行测试的test图片集加载类


class TestingSet(torch.utils.data.Dataset):#继承
    def __init__(self, dir, ouput_size,):
        self.dir = dir
        # 将文件名和对应类别获取并存到一个列表中
        names = os.listdir(self.dir)
        self.images = names
        self.ouput_size = ouput_size

        # 设置图片转换器，转换成tensor
        transformer = transforms.Compose([
            transforms.Resize(ouput_size),
            transforms.ToTensor(),
        ])

        self.transformer = transformer

    def __len__(self):
        return len(self.images)

    def GetLen(self):
        return self.__len__()

    def __getitem__(self, index):
        path = self.images[index]
        path = os.path.join(self.dir, path)
        image = self.transformer(Image.open(path).convert('RGB'))
        #注意区分：Image表示类包，但是image 仅仅是我们自己定义的函数
        return image
    
    def GetBatch(self, idx  : list):
        #根据ID取出一批图片的数据
        images = np.zeros((1, 3,self.ouput_size[0], self.ouput_size[1])) #空矩阵，便于使用concatenate
        labels = np.zeros((1,)) #空矩阵，便于使用concatenate  
        for j in idx:
            image, label = self.__getitem__(j)
            images = np.concatenate((images, image.unsqueeze(0)), axis = 0)
            labels = np.concatenate((labels, np.array([label])), axis = 0)
        images = torch.from_numpy(images[1:]).float() #去掉第一行空行
        labels = torch.from_numpy(labels[1:]).long()#去掉第一行空行
        return images, labels
