import torch
from torch import nn
from torch import optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import sentense_set
import en2cn_model
import datetime
import multiprocessing
import math

importlib.reload(sentense_set)
importlib.reload(en2cn_model)


def save_model(model, path):
    torch.save(model.cpu().state_dict(), path)


def load_model(model, path, device):
    model.load_state_dict(torch.load(path))
    return model.to(device)


def train_model(
    model,
    data,
    sampling=0.5,
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

    loss_func = nn.CrossEntropyLoss(ignore_index=0)

    if opt == 0:
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # 优化器（梯度下降的具体算法Adam）
    elif opt == 1:
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        opt = optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss = np.zeros(epochs)
    model = model.to(device)
    model.train()  # 会打开dropout、batchnorm等
    for i in range(epochs):
        loss_sum = 0
        time_start, time_pre = datetime.datetime.now(), datetime.datetime.now()
        for j, (ens, cns) in enumerate(data_loader_train):
            ens, cns = ens.to(device), cns.to(device)
            y_pred = model(ens, cns, sampling)
            # input是三维的，使用CrossEntropyLoss需要将类别维度放在第二个位置
            # cns[batch, seq_len2]
            # y_pred[batch, seq_len2, cn_vsize]
            loss_cur = loss_func(y_pred[:, 1:, :].permute(0, 2, 1), cns[:, 1:])  # 第一个字符相同，不计算
            loss_sum += loss_cur.item()
            opt.zero_grad()
            loss_cur.backward()
            # 防止梯度爆炸，提高训练速度，clip_grad_norm_会进行inplace
            nn.utils.clip_grad_norm_(model.parameters(), 100)
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
            del ens, cns, y_pred, loss_cur
            torch.cuda.empty_cache()
        loss[i] = loss_sum / math.ceil(data.__len__() / nbatch)
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


def translate(model, device, data, nbatch=512):
    model = model.to(device)
    model.eval()  # 会关闭dropout、batchnorm等optim
    data_loader = torch.utils.data.DataLoader(data, nbatch, shuffle=False, num_workers=multiprocessing.cpu_count())
    y_test = []
    with torch.no_grad():
        for _, (inputs, _) in enumerate(data_loader):
            inputs = inputs.to(device)
            y_pred = model(inputs)
            y_test.extend(y_pred)
    return y_test
    # 数字转为中文字符a = torch.tensor([1], requires_grad=True)


def translate_one(model, device, data: sentense_set.SentenseSet, en_sen):
    model = model.to(device)
    model.eval()  # 会关闭dropout、batchnorm等optim
    en_codes = data.EN2Numbers(en_sen).to(device)
    with torch.no_grad():
        y_pred = model(en_codes.unsqueeze(0))
    return data.Numbers2CN(y_pred[0])


##test
if __name__ == "__main__":
    import os

    dic = sentense_set.Dictionary()
    data = sentense_set.SentenseSet("./data/training.txt", dic)
    model = en2cn_model.EN2CN(len(dic.en_ix2word), len(dic.cn_ix2word), data.BOS, data.EOS)
    device = torch.device("cuda" if True & torch.cuda.is_available() else "cpu")
    path = "./data/model.pkl"
    if os.path.exists(path):
        model = load_model(model, path, device)
    yy = translate_one(model, device, data, "i love you .")
