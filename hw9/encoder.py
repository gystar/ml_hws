from torch import nn
import torch


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
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
            nn.Linear(512, 256),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
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

    # for training
    def forward(self, x: torch.tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # [b,3*32*32]

    # for encoding
    def encode(self, x):
        return self.encoder(x)
