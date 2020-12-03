'''
Author: gystar
Date: 2020-12-03 14:52:09
LastEditTime: 2020-12-03 17:27:20
LastEditors: Please set LastEditors
Description: 通过hook和gradient ascend方式，生成并显示pytorh的某一层cnn的filter能最大激活的图像
             最大激活：即被filter过滤后，得到的矩阵的各元素之和最大
FilePath: /ml_hws/hw5/filter_visualization.py
'''


import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
import numpy as np



def tensor2numpy(image:torch.tensor):
    #将生成的图片tensor转换为图像矩阵   
    tansformer = transforms.ToPILImage() 
    return np.array(tansformer(image))

'''
description: 
注意：使用之前必须声明一个全局变量用来hook，且固定名称为g_output
param {*} images:需要生成activation结果的图片
param {torch} model
param {*} model_layer：需要hook的层所在，由于不同的构造方法不同，索引各层的方式也不同，因此需要传入此参数
          例如image_classification.py中的GYHF_LetNet5，要hook第一个卷积层，则传入model.features[0]
param {*} filter_id：filter在当前cnn的id
param {*} input_size：模型输入的图片大小，例如(3,32,32)
param {*} global_var:学习率
param {*} iters：梯度上升得到图片的训练轮次
param {*} lr:学习率
return {*} 生成的该层能最大激活的图片，每张图片对应激活结果（均已经转换为[w,h,c]形式的图像矩阵）
'''

def generate_image(images, model:torch.nn.Module, model_layer,filter_id, input_size, iters = 10, lr = 0.1):
    def hook(model, input, output):
        global g_output
        g_output = output #将激活后的结果通过全局变量传递出去
        
    hook_handle = model_layer.register_forward_hook(hook) #hook到目标层
    input = torch.zeros(input_size).unsqueeze(0)
    input.requires_grad = True    
    
    #得到图片的激活结果图
    model(images)
    activations = g_output[:,filter_id,:,:]
    
    #通过多次梯度上升逐步修改输入的图片来得到能最大激活该层的图片
    #注意，此处要 梯度下降的是input    
    opt = torch.optim.Adam([input], lr=lr)  # 优化器（梯度下降的具体算法Adam）
    model.train()  # 会打开dropout、batchnorm等
    for _ in range(iters):
        model(input)
        #损失函数定义为激活后的矩阵元素之和，由于要增加它，则加一个负号
        loss = -g_output[:,filter_id,:,:].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    hook_handle.remove() #移除hook        
    image = input.detach().cpu().squeeze(0)
    activations = np.array([tensor2numpy(activation) for activation in activations])
    return tensor2numpy(image), activations

g_output = None
##test
if __name__ == "__main__":
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
    images,lables = data.GetBatch([1,3,5])
    image,activations  = generate_image(images,model, model.features[0], 0, (3,model.input_size[0], model.input_size[1]),iters=1000, lr=0.01)
    fig, axs = plt.subplots(2, len(images), figsize=(10, 10))
    for i, img in enumerate(images):
        axs[0][i].imshow(img.permute(1, 2, 0))
    for i, img in enumerate(activations):
        axs[1][i].imshow(img)
    plt.show()

        
    
    
    
