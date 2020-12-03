'''
Author: gystar
Date: 2020-12-03 19:23:33
LastEditTime: 2020-12-03 19:41:18
LastEditors: Please set LastEditors
Description: 通过梯度下降生成能使某个类别概率最大的输入图片
FilePath: /ml_hws/hw5/model_image.py
'''
import torch
import numpy as np
from torchvision import transforms

def tensor2numpy(t):
    #将tensor反解析为图像矩阵
    tansformer = transforms.ToPILImage() 
    return np.array(tansformer(t))

def generate(model, input_size,class_index, iters = 10, lr = 0.001):
    input = torch.zeros(input_size).unsqueeze(0) #从全0图片开始进行修正
    
    input.requires_grad = True
    opt = torch.optim.Adam([input], lr=lr)  # 优化器（梯度下降的具体算法Adam）
    model.train()  # 会打开dropout、batchnorm等
    for _ in range(iters):#不断的梯度下降，修改输入的图片，得到能使当前类别的概率最大的图片
        ouput = model(input)
        #损失函数定义为当前类别的概率的得分，即相当于model最后输出的的某个slot对应的数字
        loss = -ouput[0,class_index]        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    return tensor2numpy(input.squeeze(0))
        
###test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys
    import importlib   
    sys.path.append('../hw3') ##直接使用hw3中的model
    import image_classification
    import image_set
    importlib.reload(image_set)
    importlib.reload(image_classification)
    #指定模型类别
    model_class = image_classification.GYHF_LetNet5
    data = image_set.LearningSet( "../hw3/data/training", model_class.input_size, False)
    #使用已经训练好的hw3中的model
    model = torch.load("../hw3/<class 'image_classification.GYHF_LetNet5'>.pkl")
    image =  generate(model, (3,model.input_size[0],model.input_size[1]), 1, 10000, 0.01) 
    plt.figure(figsize=(2,2))
    plt.imshow(image)     
