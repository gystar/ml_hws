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


def train_model(model, data, epochs=10, nbatch=128,  cuda_ok=True, lr = 0.001, weight_decay = 0.01):
    # dataloader
    data_loader_train = torch.utils.data.DataLoader(
        data,
        batch_size=nbatch,
        num_workers=math.floor(multiprocessing.cpu_count()),  # 线程数等于cpu核数
        pin_memory=False,
        shuffle=True)

    # 损失函数，实际上等于LogSoftmax+NLLLoss
    # LogSoftmax=log+softmax
    # NLLLoss:negative log likelihood loss
    loss_func = torch.nn.CrossEntropyLoss()  # 损失函数
    opt = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)  # 优化器（梯度下降的具体算法Adam）

    loss = np.zeros(epochs)
    model.train()  # 会打开dropout、batchnorm等
    for i in range(epochs):
        loss_sum = 0
        time_start, time_pre = datetime.datetime.now(), datetime.datetime.now()
        for j, info in enumerate(data_loader_train):
            images, labels = info
            if cuda_ok:
                images, labels = images.cuda(), labels.cuda()
            y_pred = model(images)  # 1.先预测
            loss_cur = loss_func(y_pred, labels)  # 2.计算损失
            # 暂不需看
            loss_sum += loss_cur.item()
            if (j+1) % 10 == 0:
                time_now = datetime.datetime.now()
                print((j+1)*nbatch, "images input,", "avarage loss:", loss_sum/(j+1),
                      ", duration:", (time_now-time_pre).total_seconds())
                time_pre = time_now
            # 暂不需看
            opt.zero_grad()  # 3.对之前的梯度清零
            loss_cur.backward()  # 4.计算反向传播梯度
            opt.step()  # 5.反向传播的梯度更新
            # 暂不需看
            del images, labels, y_pred, loss_cur
            torch.cuda.empty_cache()
        loss[i] = loss_sum/math.ceil(data.GetLen()/nbatch)
        print("[epochs ", i+1, "/", epochs, "] loss:", loss[i], "duration:",
              (datetime.datetime.now()-time_start).total_seconds())

    # 绘图
    plt.figure()
    plt.plot([x for x in range(0, epochs)], loss)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()
    return model
