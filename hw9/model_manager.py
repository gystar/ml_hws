import torch
from torch import nn
from torch import optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import image_set
import encoder
import datetime
import multiprocessing
import math
import os

importlib.reload(image_set)
importlib.reload(encoder)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    model.load_state_dict(torch.load(path))
    return model.to(device)


def train_model(
    model,
    data,
    save_path,
    device=torch.device("cpu"),
    lr=0.0001,
    epochs=10,
    nbatch=32,
    opt=0,  # 0 Adam,1 SGDM,2 Adadelta
    momentum=0.9,
    weight_decay=0,
    clip_norm=1.0,
):
    # dataloader
    data_loader_train = torch.utils.data.DataLoader(
        data,
        batch_size=nbatch,
        num_workers=multiprocessing.cpu_count(),  # 线程数等于cpu核数
        pin_memory=False,
        shuffle=True,
    )

    loss_func = nn.L1Loss()
    if opt == 0:
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # 优化器（梯度下降的具体算法Adam）
    elif opt == 1:
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        opt = optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss = np.zeros(epochs)
    loss_pre = None
    model = model.to(device)
    model.train()  # 会打开dropout、batchnorm等
    for i in range(epochs):
        loss_sum = 0
        time_start, time_pre = datetime.datetime.now(), datetime.datetime.now()
        for j, images in enumerate(data_loader_train):
            images = images.to(device).contiguous()
            y_pred = model(images)
            loss_cur = loss_func(images, y_pred)
            loss_sum += loss_cur.item()
            opt.zero_grad()
            loss_cur.backward()
            # 防止梯度爆炸，提高训练速度，clip_grad_norm_会进行inplace
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            opt.step()
            if (j + 1) % 10 == 0:
                time_now = datetime.datetime.now()
                print(
                    "%d inputs, avarage loss: %f , duration: %f"
                    % (
                        min((j + 1) * nbatch, data.__len__()),
                        loss_sum / (j + 1),
                        (time_now - time_pre).total_seconds(),
                    )
                )
                time_pre = time_now
            del images, y_pred, loss_cur
            torch.cuda.empty_cache()
        loss[i] = loss_sum / math.ceil(data.__len__() / nbatch)
        # 如果获得loss更小的模型，则保存
        if (not os.path.exists(save_path)) or (loss_pre != None and loss[i] < loss_pre):
            print("Got a better model, save it.")
            save_model(model, save_path)
        loss_pre = loss[i]
        print(
            "[epochs  %d / %d ] loss: %f duration: %f"
            % (
                i + 1,
                epochs,
                loss[i],
                (datetime.datetime.now() - time_start).total_seconds(),
            )
        )

    # 绘图
    plt.figure()
    plt.plot([x for x in range(0, epochs)], loss)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()
    return model


def encode(model, device, data, nbatch=512):
    model = model.to(device)
    model.eval()  # 会关闭dropout、batchnorm等optim
    data_loader = torch.utils.data.DataLoader(data, nbatch, shuffle=False, num_workers=multiprocessing.cpu_count())
    ret = torch.zeros((data.__len__(), model.codedim))
    i = 0
    with torch.no_grad():
        for _, (inputs, _) in enumerate(data_loader):
            inputs = inputs.to(device)
            y_pred = model.encode(inputs).cpu()
            count = y_pred.shape[0]
            ret[i : i + count] = y_pred
            i += count

    return ret


###test
if __name__ == "__main__":
    import importlib
    import image_set
    import encoder

    importlib.reload(image_set)
    importlib.reload(encoder)

    set = image_set.NoLabeledSet(True)
    model = encoder.AutoEncoder()
    model = train_model(model, set, "./data/model.pkl")