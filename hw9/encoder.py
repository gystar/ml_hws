from torch import nn
import torch


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.codedim = 256 * 4 * 4  # encoder得到的图片像素点数目，通过后面计算得到
        # MaxUnpool2d需要对应的MaxPool2d给出indeces，因此需要分开写

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_r1 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1)
        self.conv_r2 = nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1)
        self.conv_r3 = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(2, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(2)
        """
        self.encoder = nn.Sequential(  # stride=1, padding=1则w和h不变
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),  # [256，4，4]
        )
        self.decoder = nn.Sequential(
            nn.MaxUnpool2d(2),
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxUnpool2d(2),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxUnpool2d(2),
            nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        """

        """
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 2048),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, self.codedim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.codedim, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(2048, 3 * 32 * 32),
        )
        """

    # for training
    def forward(self, x: torch.tensor):
        # encoder
        x = self.conv1(x)
        x = self.relu(x)
        x, idx1 = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x, idx2 = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x, idx3 = self.maxpool(x)

        # decoder
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
        x, _ = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x, _ = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x, _ = self.maxpool(x)
        return x.view(x.shape[0], -1)
