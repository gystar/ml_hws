# 语义识别的RNN模型定义
# 输入语句向量，输出属于positive的概率
from torch import nn
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import importlib
import utils
importlib.reload(utils)

cuda_ok = torch.cuda.is_available()
print("Cuda is available" if cuda_ok else "There is no gpu available.")

class SentimentModel(nn.Module):
    def __init__(self, dim_word):
        super(SentimentModel, self).__init__()
        HSIZE = 512
        self.rnns = nn.LSTM(input_size=dim_word,
                            hidden_size=HSIZE,
                            num_layers=3,
                            batch_first=False,  # 第二维为batch
                            )
        self.fcs = nn.Sequential(
            nn.Linear(HSIZE, 256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (x, _) = self.rnns(x)
        # 只取hn来使用
        x = x.squeeze(0)
        x = self.fcs(x)
        return x[-1]


def predict(model, sentenses, wv):
    # 预测一个大的文本集合，因为较大，所以不宜全部放进内容进行计算
    nbatch = 5000
    ret = torch.tensor([]).cuda() if cuda_ok else torch.tensor([])    
    m = model.cuda() if cuda_ok else model.cpu()
    m.eval()
    with torch.no_grad():
        for i in range(0, len(sentenses), nbatch):
            x_sentense = sentenses[i:min(
                len(sentenses), i+nbatch)]
            vecs = utils.sens2vecs(x_sentense, wv)
            if cuda_ok:
                vecs = vecs.cuda()      
            y_pred = m(vecs).squeeze()           
            ret = torch.cat((ret, y_pred), 0)
    return ret.cpu()


def train_model(sentenses, labels, wv, dim_word, model = None, epochs=10, nbatch=100):
    if model == None:#否则继续训练
        model = SentimentModel(dim_word)
    if cuda_ok:
        model = model.cuda()
    loss_func = nn.BCELoss()  # 损失函数
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    iters = math.ceil(len(sentenses)/nbatch)
    loss_all = np.zeros(epochs)

    for i in range(epochs):
        model.train()
        loss_sum = 0
        for j in range(iters):
            idx = np.random.randint(0, len(sentenses), 5)
            x_sentenses = sentenses[idx]
            x_labels = torch.tensor(labels[idx]).float()
            vecs = utils.sens2vecs(x_sentenses, wv)
            if cuda_ok:
                x_labels,vecs = x_labels.cuda(),vecs.cuda()
            y_pred = model(vecs).squeeze()
            loss = loss_func(y_pred, x_labels)
            loss_sum += loss.item()
            if (j+1) % 100 == 0:
                print((j+1)*nbatch, " sentenses input, avarage loss:", loss_sum/(j+1))
            if j+1 == iters:
                loss_all[i] = loss_sum/(j+1)
                print("[epochs", i+1, "/", epochs,
                      "], avarage loss:", loss_all[i])
            opt.zero_grad()
            loss.backward()
            opt.step()

    # 绘图
    plt.figure()
    plt.plot([x for x in range(0, epochs)], loss_all)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

    return model.cpu()
