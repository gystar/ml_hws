from torch import nn
import torch


class ConvEnoder(nn.Module):
    def __init__(self):
        super(ConvEnoder, self).__init__()
        # 输入图片大小为3*32*32
        self.codedim = 32  # encoder得到的隐藏向量维度
        # MaxUnpool2d需要对应的MaxPool2d给出indeces，因此需要分开写

        # sequence:（省略了pool和relu）
        # conv1 conv1 conv1 fc1 fc2 fc3 fc4 conv_r1 conv_r2 conv_r3
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 256)  # 输入的维度由上面几层的卷积过程计算得到
        self.fc2 = nn.Linear(256, self.codedim)
        self.fc3 = nn.Linear(self.codedim, 256)
        self.fc4 = nn.Linear(256, 256 * 4 * 4)
        self.conv_r1 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1)
        self.conv_r2 = nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1)
        self.conv_r3 = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(2, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(2)

    # for training
    def forward(self, x: torch.tensor):
        # encoder
        x, (idx1, idx2, idx3) = self.encode(x)

        # decoder
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = x.view(x.shape[0], 256, 4, 4)
        x = self.maxunpool(x, idx3)
        x = self.conv_r1(x)
        x = self.relu(x)
        x = self.maxunpool(x, idx2)
        x = self.conv_r2(x)
        x = self.relu(x)
        x = self.maxunpool(x, idx1)
        x = self.conv_r3(x)
        x = nn.Tanh()(x)

        return x  # [b,3*32*32]

    # for encoding
    def encode(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x, idx1 = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x, idx2 = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x, idx3 = self.maxpool(x)

        x = x.view(x.shape[0], -1)  # flatten before fc
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x, (idx1, idx2, idx3)


class DCN(nn.Module):
    def __init__(self):
        super(DCN, self).__init__()
        # 输入图片大小为3*32*32
        self.codedim = 6  # encoder得到的隐藏向量维度
        # MaxUnpool2d需要对应的MaxPool2d给出indeces，因此需要分开写
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 3 * 32 * 4),
            nn.ReLU(True),
            nn.Linear(3 * 32 * 4, 3 * 16),
            nn.ReLU(True),
            nn.Linear(3 * 16, self.codedim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(6, 6 * 8),
            nn.ReLU(True),
            nn.Linear(6 * 8, 6 * 8 * 8),
            nn.ReLU(True),
            nn.Linear(6 * 8 * 8, 6 * 8 * 8 * 8),
        )

    def encode(self, x):
        return self.encoder(x), None  # 和上一个模型返回相同形式的，作为兼容

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(x.shape[0], 3, 32, 32)
