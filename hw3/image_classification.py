# 定义图片分类的pytorch的model
# 按照李宏毅课程CNN那一节来构造


from torch import nn #从torch中引用nn(neural network）
import torch
from typing import Union, List, cast , Dict,Any

#经典模型一（需学习）
class GYHF_LetNet5(nn.Module):#继承父类
    # 典型的LetNet5实现
    #注意，原本的LetNet5是用来处理手写字母数字识别的，输入灰度图像即1个通道的，而且第二个卷积层较为复杂，
    #此处实现，是对彩色3通道数据进行处理，第二个卷积层设置也很简单，还加入了relu激活函数
    #因此，对于此题的分类效果很不理想
    # 现在假设输入的图像均为(32,32)大小的，以下数字根据size和stride等计算得出
    input_size = (32, 32)#输入图像均为（32,32)的矩阵

    def __init__(self, class_count):
        super(GYHF_LetNet5, self).__init__()#调用父类的初始化方法
        self.features = nn.Sequential( #使用nn.Sequential去定义特征
            nn.Conv2d(in_channels=3, out_channels=6,#6个5*5*3的filter
                      kernel_size=5),  # (6,28,28)
            nn.ReLU(), #Rectified Linear Unit
            nn.MaxPool2d(2, 2),  # (6,14,14)   2*2的pooling ,步长为2
            nn.Conv2d(6, 16, 5),  # (16,10,10)   16个5*5*6的filter
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (16,5,5)
            nn.Conv2d(16, 120, 5),  # (120,1,1)
            nn.ReLU(),
            nn.Flatten()  # (120,)
        )#从15行到26行表示特征提取
         #28行——33行表示对特征进行线性分类
        self.classifer = nn.Sequential(#定义分类器
            nn.Linear(in_features=120, out_features=84),#通过flatten中的120个特征，自定义选出84个
            nn.ReLU(),
            nn.Linear(84, class_count),
            # nn.Softmax()#不需要进行softmax，因为是递增函数，和求前一步的最大值一样
        )

    def forward(self, x):  #正向传播
        x = self.features(x)
        x = self.classifer(x)
        return x
#模型二：AlexNet
class GYHF_AlexNet(nn.Module):
    input_size = (224, 224)
    def __init__(self,num_classes=10):        
        super(GYHF_AlexNet,self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=2,bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
 
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
            nn.Conv2d(in_channels=96,out_channels=192,kernel_size=5,stride=1,padding=2,bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
 
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
            nn.Conv2d(in_channels=192,out_channels=384,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
 
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
 
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256*6*6,out_features=4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(True),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
    def forward(self,x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0),256*6*6)
        x = self.classifier(x)
        return x

#模型三:VGG
def make_vgg_layers(cfg: List[Union[str, int]], batch_norm: bool = False):
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
#下面四种配置分别代表11，13，16，19层的VGG网络的feature层配置
cfgs: Dict[str, List[Union[str, int]]] = { \
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], \
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], \
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], \
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], \
    }
class GYHF_VGG(nn.Module):#继承父类
    # 现在假设输入的图像均为(3,224,224)大小的，以下数字根据size和stride等计算得出
    input_size = (224, 224)     
    def __init__(
        self,        
        num_classes: int,
        cfg:str, #vgg11等，表示不同的VGG层级 
        batch_norm:bool,      
    ):
        super(GYHF_VGG, self).__init__()
        self.features = make_vgg_layers(cfgs[cfg], batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x  
    
#经典模型四
class GYHF_TestNet(nn.Module):#继承父类
    # 典型的LetNet5实现
    #注意，原本的LetNet5是用来处理手写字母数字识别的，输入灰度图像即1个通道的，而且第二个卷积层较为复杂，
    #此处实现，是对彩色3通道数据进行处理，第二个卷积层设置也很简单，还加入了relu激活函数
    #因此，对于此题的分类效果很不理想
    # 现在假设输入的图像均为(32,32)大小的，以下数字根据size和stride等计算得出
    input_size = (32, 32)#输入图像均为（32,32)的矩阵

    def __init__(self, class_count):
        super(GYHF_TestNet, self).__init__()#调用父类的初始化方法
        self.features = nn.Sequential( #使用nn.Sequential去定义特征
            nn.Conv2d(in_channels=3, out_channels=64,#64个5*5*3的filter
                      kernel_size=3),  # (64,30,30)
            nn.BatchNorm2d(64),
            nn.ReLU(), #Rectified Linear Unit
            nn.MaxPool2d(2, 2),  # (64,15,15)   2*2的pooling ,步长为2
            nn.Conv2d(64, 128, 3),  # (128,13,13)   16个5*5*6的filter
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (128,6,6)
            nn.Conv2d(128, 256, 3),  # (256,4,4)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (256,2,2)
            nn.Flatten()  # (1024,)
        )
        #从15行到26行表示特征提取
         #28行——33行表示对特征进行线性分类
        self.classifer = nn.Sequential(#定义分类器
            nn.Linear(in_features=1024, out_features=512),#通过flatten中的120个特征，自定义选出84个
            nn.ReLU(),
            nn.Linear(512, class_count),
            # nn.Softmax()#不需要进行softmax，因为是递增函数，和求前一步的最大值一样
        )

    def forward(self, x):  #正向传播
        x = self.features(x)        
        x = self.classifer(x)
        return x
