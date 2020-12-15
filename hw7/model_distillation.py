"""
Author: gystar
Date: 2020-12-14 10:59:54
LastEditors: gystar
LastEditTime: 2020-12-14 10:59:54
FilePath: /ml_hws/hw7/model_distillation.py
Description: 使用小模型去学习大的模型
"""
import torch
import multiprocessing
import numpy as np
import datetime
import math


def train_student(teaher_model, student_model, data, device, epochs=10, nbatch=32, lr=0.001, opt=0):
    # dataloader
    data_loader_train = torch.utils.data.DataLoader(
        data,
        batch_size=nbatch,
        num_workers=multiprocessing.cpu_count(),  # 线程数等于cpu核数
        pin_memory=False,
        shuffle=True,
    )

    loss_func = torch.nn.MSELoss(reduction="mean")  # 损失函数
    opt_func = torch.optim.Adam if opt == 0 else torch.optim.SGD
    opt = opt_func(student_model.parameters(), lr=lr, weight_decay=0.001)
    loss = np.zeros(epochs)

    teaher_model, student_model = teaher_model.to(device), student_model.to(device)
    student_model.train()
    teaher_model.eval()
    for i in range(epochs):
        loss_sum = 0
        time_start, time_pre = datetime.datetime.now(), datetime.datetime.now()
        for j, (images, _) in enumerate(data_loader_train):
            images = images.to(device)
            with torch.no_grad():
                y_origin = teaher_model(images)
            y = student_model(images)
            loss_cur = loss_func(y_origin, y)
            loss_sum += loss_cur.item()
            opt.zero_grad()
            loss_cur.backward()
            opt.step()
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
            del images, y_origin, y, loss_cur
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
