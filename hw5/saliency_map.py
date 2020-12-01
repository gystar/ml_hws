#saliency map的实现
#基于hw3中的letnet5训练好的model来实现
import os,sys,math
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import importlib
import matplotlib.pyplot as plt
import random
sys.path.append('../hw3') ##直接使用hw3中的model
import image_classification
import image_set
importlib.reload(image_set)
importlib.reload(image_classification)
random.seed(10)


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

#每个列别取出一张概率最大的图片进行可视化
idx = []
for i in range(class_num):
    idx.append(output[:,i].topk(1)[1].item()) #得到所需图片的索引

#取出这些图片，并且重新预测一次，并且计算输入的偏导
images = np.zeros((1, 3,model_class.input_size[0], model_class.input_size[1])) #空矩阵，便于使用concatenate
labels = np.zeros((1,)) #空矩阵，便于使用concatenate  
for j in idx:
    image, label = data.__getitem__(j)
    images = np.concatenate((images, image.unsqueeze(0)), axis = 0)
    labels = np.concatenate((labels, np.array([label])), axis = 0)
images = torch.from_numpy(images[1:]).float()
labels = torch.from_numpy(labels[1:]).long()
images.requires_grad = True
model.eval()
y_pred = model(images)
loss_func = torch.nn.CrossEntropyLoss()
loss = loss_func(y_pred, labels)    
loss.backward()

def tensor2numpy(t):
    #将tensor反解析为图像
    tansformer = transforms.ToPILImage() 
    return np.array(tansformer(t))
def grad2numpy(grad):
    #将图片的梯度转换为图像矩阵
    img = grad.detach().abs().numpy()
    return ((img - img.min()) / (img.max() - img.min())).transpose(1,2,0)
#所有图像的梯度
images_grad = images.grad.detach().cpu()
#绘制图片
fig = plt.figure(figsize=model_class.input_size)
cols = 4
rows = 3*2
for i in range(k):  
    row ,col = math.floor(i/cols), i%cols
    pos1 = row*cols*2+col+1
    pos2 = pos1 + cols
    #先画原图  
    ax = fig.add_subplot(3*2, 4, pos1, xticks=[], yticks=[]) 
    ax.imshow(tensor2numpy(images[i]))
    ax.set_title('Image %s' % str(i), fontsize=30,color='r')
    #再画出显著图
    ax = fig.add_subplot(3*2, 4, pos2, xticks=[], yticks=[]) 
    ax.imshow(grad2numpy(images_grad[i]))
    ax.set_title('saliency %s' % str(i), fontsize=30,color='r')    






