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

#小（旧）Model: teacher_model,想要得到的新Model:student_model,就是使用旧的model去训练新model
def train_student(teacher_model, student_model, data, device, epochs=10, nbatch=32, lr=0.001, opt=0):
    # dataloader
    data_loader_train = torch.utils.data.DataLoader(
        data,
        batch_size=nbatch,
        num_workers=multiprocessing.cpu_count(),  # 线程数等于cpu核数
        pin_memory=False,#不放入显存
        shuffle=True,
    )

    loss_func = torch.nn.MSELoss(reduction="mean")  # 损失函数，使用MSE平均值后的值为损失函数输出
    opt_func = torch.optim.Adam if opt == 0 else torch.optim.SGD#如果opt==0则使用Adam更新opt的参数，否则使用SGD更新参数
    opt = opt_func(student_model.parameters(), lr=lr, weight_decay=0.001)#opt更新的是student——model的参数，weight——decay表示正则化
    loss = np.zeros(epochs)#希望输出每次循环的loss

    teacher_model, student_model = teacher_model.to(device), student_model.to(device)
    student_model.train()#打开dropout（每个参数都有p[0,1]的机率被置为零）,normalbatch（类似于增加惩罚项）
    teacher_model.eval()#关闭dropout,normalbatch
    for i in range(epochs):
        loss_sum = 0
        time_start, time_pre = datetime.datetime.now(), datetime.datetime.now()
        for j, (images, _) in enumerate(data_loader_train):#返回dataloader 中的两项（image，_）枚举会返回两个结果：序号、（）
            images = images.to(device)#使用"="为了覆盖原值，to不改变images本身，images.to(device)只是增加了一个临时副本，不使用就会被释放
            with torch.no_grad():#不保存teacher_model 的求导过程流程图既可以省显存又可以增加运算速度（从with的控制块可以看到是不保存teacher的求导）
                y_origin = teacher_model(images)#使用teacher_model 去得到old概率分布
            y = student_model(images)
            loss_cur = loss_func(y_origin, y)
            loss_sum += loss_cur.item()#loss_cur默认保存在GPU上，此处使用item可将loss_cur从GPU 中转放在CPU上，可以省显存
            opt.zero_grad()#为参数提前申请位置
            loss_cur.backward()
            opt.step()#对backward算出来的参数进行性更新
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
