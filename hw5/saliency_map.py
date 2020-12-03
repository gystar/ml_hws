#saliency map的实现
#基于hw3中的letnet5训练好的model来实现
#思路：求出输入图片的梯度，将梯度矩阵用一定的方式转换为图像矩阵显示出来
import math
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

def tensor2numpy(t):
    #将tensor反解析为图像矩阵
    tansformer = transforms.ToPILImage() 
    return np.array(tansformer(t))
def grad2numpy(grad):
    #将图片的梯度转换为图像矩阵
    img = grad.detach().abs().numpy()
    #将每个梯度值做这种比较简单的标准化，以免不同的图色差太大。
    return ((img - img.min()) / (img.max() - img.min())).transpose(1,2,0)#注意这里需要转置，因为plt画图通道维度默认在最后

def draw(model, images, labels, fontsize = 15):
    images.requires_grad = True #指定需要求导，使输入的图片矩阵在反向传播的时候也会被求导
    model.eval()
    y_pred = model(images)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, labels)    
    loss.backward() #一次反向传播，进行求导
    
    num = images.shape[0] #图片数量

    #所有图像的梯度
    images_grad = images.grad.detach().cpu()
    #绘制图片
    fig = plt.figure(figsize=(8,8))
    cols = 4  #绘制的列数
    for i in range(num):  
        #原图像和显著图呈上下位置分布，计算出他们在图像中的位置
        row ,col = math.floor(i/cols), i%cols
        pos1 = row*cols*2+col+1 #原图像在大图中的位置，即第几个子图
        pos2 = pos1 + cols      #对应的显著图在大图中的位置，即第几个子图
        #先画原图  
        ax = fig.add_subplot(3*2, 4, pos1, xticks=[], yticks=[]) #添加原图像的子图
        ax.imshow(tensor2numpy(images[i]))
        ax.set_title('Image %s' % str(i), fontsize=fontsize,color='r') #设置标题
        #再画出显著图
        ax = fig.add_subplot(3*2, 4, pos2, xticks=[], yticks=[]) #添加显著图的子图
        ax.imshow(grad2numpy(images_grad[i]))
        ax.set_title('saliency %s' % str(i), fontsize=fontsize,color='r')    


##test
if __name__ == "__main__" :
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
    images,lables = data.GetBatch([0,1,2])
    draw(model, images, lables, 7)