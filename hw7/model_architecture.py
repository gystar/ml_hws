"""
Author: gystar
Date: 2020-12-14 10:59:54
LastEditors: gystar
LastEditTime: 2020-12-14 10:59:54
FilePath: /ml_hws/hw7/model_architecture.py
Description: 构造一个小的模型，能达到接近原有大模型的效果。
"""
import torch
import torchvision.models as models

# 使用depthwise和pointwise方法对卷基层进行简化，减少其参数数量
def SmartConv2d(origin: torch.nn.Conv2d):
    # 使用两个卷积层来来代替之前的一个卷积层
    # 第一个Conv2d:先使用torch.nn.Conv2d的gourp属性达到depthwise的效果
    # 第二个Conv2d:使用大小为1的卷积核来实现各个channel间的联系
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            # 令groups = out_channels = 输入的in_channels
            # 则每个卷积核分别对一个channel进行卷积
            in_channels=origin.in_channels,
            out_channels=origin.in_channels,
            groups=origin.in_channels,
            kernel_size=origin.kernel_size,
            stride=origin.stride,
            padding=origin.padding,
            dilation=origin.dilation,
            bias=(origin.bias != None),
        ),
        torch.nn.Conv2d(
            in_channels=origin.in_channels,
            out_channels=origin.out_channels,
            groups=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=(origin.bias != None),
        ),
    )


class SmartResnet18(torch.nn.Module):
    def __init__(self):
        super(SmartResnet18, self).__init__()
        # 直接使用resnet18的结构，但是将每一个conv2d换成SmartConv2d来减少参数
        # 具体的结构可以在加载后打印出来进行分析，后面的修改都基于具体的结构
        self.resnet18 = models.resnet18(pretrained=False, num_classes=11)
        # 修改输入层的conv2d
        self.resnet18.conv1 = SmartConv2d(self.resnet18.conv1)
        # 修改每个layerx中的conv2d
        for i in range(4):
            layer = getattr(self.resnet18, "layer" + str(i + 1))
            for j in range(2):
                block = layer[j]
                block.conv1 = SmartConv2d(block.conv1)
                block.conv2 = SmartConv2d(block.conv2)

    def forward(self, x):
        return self.resnet18(x)


##test
if __name__ == "__main__":
    ##测试一下group的效果
    conv2d_1 = torch.nn.Conv2d(in_channels=3, out_channels=3, groups=3, kernel_size=(2, 2), stride=1, bias=True)
    for i in range(len(conv2d_1.weight)):  # 将卷积核初始化为权重为1，bias为7，便于验证计算
        conv2d_1.weight[i] = torch.full(conv2d_1.weight[i].shape, 1)
        conv2d_1.bias[i] = 7
    print(conv2d_1.weight[0])
    print(conv2d_1.bias)

    a = torch.randn((1, 3, 4, 4))
    a[:, 0, :, :] = 1
    a[:, 1, :, :] = 2
    a[:, 2, :, :] = 3
    print(a)
    b = conv2d_1(a)
    print(b.data)
    print(b.shape)

    conv2d_2 = torch.nn.Conv2d(in_channels=3, out_channels=2, groups=1, kernel_size=1, stride=1, bias=True)
    for i in range(len(conv2d_2.weight)):  # 将卷积核初始化为权重为1，bias为5，便于验证计算
        conv2d_2.weight[i] = torch.full(conv2d_2.weight[i].shape, 1)
        conv2d_2.bias[i] = i
    print(conv2d_2.weight[0])
    print(conv2d_2.bias)
    c = conv2d_2(b)
    print(c.data)
    print(c.shape)

    # 测试一下smartconv2d
    sconv2d = SmartConv2d(torch.nn.Conv2d(3, 2, 2))
    for j in range(2):
        conv2d = sconv2d[j]
        for i in range(len(conv2d.weight)):  # 将卷积核初始化为权重为1，bias为7，便于验证计算
            conv2d.weight[i] = torch.full(conv2d.weight[i].shape, 1)
            conv2d.bias[i] = i
        print(conv2d.weight[0])
        print(conv2d.bias)
    print(sconv2d(a))
    # 打印一个对象的所有属性
    print("\n".join(["%s:%s" % item for item in torch.nn.Conv2d(3, 2, 2, bias=False).__dict__.items()]))

    snet = SmartResnet18()
    print(snet.resnet18)
    image = torch.randn((1, 3, 224, 224))
    print(snet(image))
    # 保存一下，比较大小，可以看到压缩后的大小接近1/10
    torch.save(snet.state_dict(), "./data/smart_resnet18.bin")
