from torch import nn
import torch


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 32 * 32),
        )

    # for training
    def forward(self, x: torch.tensor):
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(b, c, h, w)
        return x

    # for encoding
    def encode(self, x):
        b, c, _, _ = x.shape
        x = x.view(b, c, -1)
        return self.encoder(x)
