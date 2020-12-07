'''
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
'''
import torch
from torchvision import transforms


cuda_ok = torch.cuda.is_available()

def tensor2numpy(t):
    #将tensor反解析为图像矩阵
    tansformer = transforms.ToPILImage() 
    return np.array(tansformer(t))

def attack(model, image, label, tolerance):
    image = image.unsqueeze(0)
    label = torch.tensor([label])
    
    if cuda_ok:
        image,label,model = image.cuda(),label.cuda(), model.cuda()
     
    #会计算输入图片矩阵的导数
    image.requires_grad = True
    y = model(image).squeeze(0)
    prediction = y.topk(1)[1]
    #损失函数：使原来的预测值尽可能的小
    loss = -1*y[prediction].squeeze(0)
    loss.backward()
    #更新输入的图片
    update = torch.zeros(image.grad.shape)
    update[image.grad >0] = tolerance
    update[image.grad <0] = -1*tolerance
    if cuda_ok:
        update = update.cuda()
    image_new =image.clone()+ update 
    #再预测一次
    y1 = model(image_new).squeeze(0)
    prediction_new = model(image).squeeze(0).topk(1)[1]  
    print(y.topk(1)[0])
    print(y1.topk(1)[0])
    
    return  prediction, prediction_new, tensor2numpy(image_new.detach().cpu().squeeze(0))

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
    
    a = torch.tensor([[1,-1],[1,2]])
    b = torch.where(a >0, 1, -1)
    
    image,lable = data.__getitem__(0)
    p1,p2,ret = attack(alexnet, image, lable, 0.1)
    print(data.category_names[p1])
    print(data.category_names[p2])
    plt.figure()
    plt.imshow(tensor2numpy(image))
    plt.figure()
    plt.imshow(ret)
    




