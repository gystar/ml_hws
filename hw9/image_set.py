import numpy as np
from numpy.core.numeric import binary_repr
import torch
from torchvision import transforms


class NoLabeledSet(torch.utils.data.Dataset):
    def __init__(self, augment=False):
        if augment:
            transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),  # 随机水平翻转图片
                    transforms.RandomRotation(15),  # 随机旋转图片
                ]
            )
        else:
            transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        data = np.load("./data/trainX.npy")
        self.images = torch.zeros(
            (data.shape[0], data.shape[3], data.shape[1], data.shape[2])
        )
        for i, image in enumerate(data):
            self.images[i] = transformer(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


class ValidationSet(torch.utils.data.Dataset):
    def __init__(self):
        transformer = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        data = np.load("./data/valX.npy")
        self.images = torch.zeros(
            (data.shape[0], data.shape[3], data.shape[1], data.shape[2])
        )
        for i, image in enumerate(data):
            self.images[i] = transformer(image)

        self.labels = torch.from_numpy(np.load("./data/valY.npy"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


###test
if __name__ == "__main__":
    set1 = NoLabeledSet(True)
    set2 = ValidationSet()
    a = torch.randn((2, 3, 16, 16))
    b = torch.randn((2, 3, 16, 16))
    from torch import nn

    nn.MSELoss()(a, b)
    print(1)
