#saliency map的实现
#基于hw3中的letnet5训练好的model来实现
import os,sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import importlib
import matplotlib.pyplot as plt
sys.path.append('../hw3') ##直接使用hw3中的model
import image_classification
import image_set
importlib.reload(image_set)
importlib.reload(image_classification)


current_dir = os.path.dirname(__file__)
MODEL_PATH =  os.path.join(current_dir, "../hw3/<class 'image_classification.GYHF_LetNet5'>.pkl")
TRAIN_DIR = os.path.join(current_dir, "../hw3/data/training")
OUTPUT_DIR = os.path.join(current_dir, "output")
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

#指定模型类别
model_class = image_classification.GYHF_LetNet5
data = image_set.LearningSet( TRAIN_DIR, model_class.input_size, False)
#使用已经训练好的model
model = torch.load(MODEL_PATH)
#对于输入的image求导,saliencymap中的像素值取3通道偏导中最大值的绝对值
#这个偏导的数值代表了显著程度
#取每个类别得分最高的10张图求平均
#得分函数即对应的概率大小，不加softmax则为最后一层的输出的大小

data_loader = torch.utils.data.DataLoader(
    data, 1024, shuffle=False,
    num_workers=2)
class_num = data.GetClassNum()
#先计算出所有图片的输出
output = np.zeros((1, class_num)) #空矩阵，便于使用concatenate
model.eval()  # 会关闭dropout、batchnorm等
with torch.no_grad():  # 不构建计算图
    for _, info in enumerate(data_loader):
        images, labels = info             
        y_pred = model(images).numpy()                
        output = np.concatenate((output, y_pred), axis=0)            
output = torch.tensor(output[1:]).float() #去掉第一行的空行


fig = plt.figure(figsize=(40, 40))
k = 10   #每一个类别中选出得分(概率)最高的k张图
for i in range(class_num):
    topk_idx = output[:,i].topk(k)[1] #得到所需图片的索引
    #取出这些图片，并且重新预测一次，并且计算输入的偏导
    images = np.zeros((1, 3,model_class.input_size[0], model_class.input_size[1])) #空矩阵，便于使用concatenate 
    for j in topk_idx:
        image, _ = data.__getitem__(j)
        images = np.concatenate((images, image.unsqueeze(0)), axis = 0)
    images = torch.from_numpy(images[1:]).float()
    images.requires_grad = True
    y_pred = model(images)    
    for j in range(k):#计算每一张图片的梯度        
        loss = y_pred[j,i]
        loss.backward(retain_graph=True) 
    #取图像每个像素的每个通道的梯度绝对值最大的作为代表
    #k幅图再求平均
    mean_map = images.grad.abs().max(axis = 1)[0].mean(axis = 0)
    #转为PIL图像
    saliency_map = transforms.ToPILImage()(mean_map)
    #绘制图片
    ax = fig.add_subplot(4, 3, i+1, xticks=[], yticks=[]) 
    ax.imshow(saliency_map.convert("L"),cmap='gray')
    ax.set_title('Class %s' % str(i), fontsize=50,color='r')   
    #保存图片
    saliency_map.save(OUTPUT_DIR+"/saliencymap_"+str(i)+".png")   





