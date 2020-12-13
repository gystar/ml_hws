"""
Author: your name
Date: 2020-12-05 16:53:26
LastEditTime: 2020-12-08 15:25:53
LastEditors: your name
Description: In User Settings Edit
FilePath: /ml_hws/hw3/model_manager.py
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import image_set  # 图片数据集实现类
import image_classification  # 分类模型实现类
import datetime
import multiprocessing
import math

# 如果自定义的模块代码改变，需要reload
importlib.reload(image_set)
importlib.reload(image_classification)


def train_model(
    model,
    data,
    device=torch.device("cpu"),
    lr=0.001,
    epochs=10,
    nbatch=32,
    opt=0,  # 0 Adam,1 SGDM,
    weight_decay=0,
):
    # dataloader
    data_loader_train = torch.utils.data.DataLoader(
        data,
        batch_size=nbatch,
        num_workers=math.floor(multiprocessing.cpu_count()),  # 线程数等于cpu核数
        pin_memory=False,
        shuffle=True,
    )

    # 损失函数，实际上等于LogSoftmax+NLLLoss
    # LogSoftmax=log+softmax
    # NLLLoss:negative log likelihood loss
    loss_func = torch.nn.CrossEntropyLoss()  # 损失函数
    if opt == 0:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # 优化器（梯度下降的具体算法Adam）
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    loss = np.zeros(epochs)
    model = model.to(device)
    model.train()  # 会打开dropout、batchnorm等
    for i in range(epochs):
        loss_sum = 0
        time_start, time_pre = datetime.datetime.now(), datetime.datetime.now()
        for j, info in enumerate(data_loader_train):
            images, labels = info
            images, labels = images.to(device), labels.to(device)
            y_pred = model(images)  # 1.先预测
            loss_cur = loss_func(y_pred, labels)  # 2.计算损失
            # 暂不需看
            loss_sum += loss_cur.item()
            # 暂不需看
            opt.zero_grad()  # 3.对之前的梯度清零
            loss_cur.backward()  # 4.计算反向传播梯度
            opt.step()  # 5.反向传播的梯度更新
            # 暂不需看
            if (j + 1) % 10 == 0:
                time_now = datetime.datetime.now()
                print(
                    "%d images input, avarage loss: %f , duration: %f"
                    % (
                        min((j + 1) * nbatch, data.__len__()),
                        loss_sum / (j + 1),
                        (time_now - time_pre).total_seconds(),
                    )
                )
                time_pre = time_now
            del images, labels, y_pred, loss_cur
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


def predict(model, device, dir, nbatch=128):
    # 对目录中的图片进行预测
    model.eval()  # 会关闭dropout、batchnorm等
    model = model.to(device)
    data_test = image_set.TestingSet(dir, model.input_size)
    data_loader_test = torch.utils.data.DataLoader(data_test, nbatch, shuffle=False, num_workers=multiprocessing.cpu_count())
    y_test = []
    with torch.no_grad():
        for _, images in enumerate(data_loader_test):
            images = images.to(device)
            y_pred = model(images).detach().cpu().squeeze()
            # 获得类别，即最大元素下标
            y_test.extend(list(np.argmax(y_pred.numpy(), 1)))
            del images, y_pred
            torch.cuda.empty_cache()

    return y_test
