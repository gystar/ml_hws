import torch
import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import random

# 从csv文件中读入数据，注意此繁体中文csv使用的是big5编码
cols = [i for i in range(3, 27)]
tdata = pd.read_csv(
    # os.path.join(os.getcwd(),"hw1/data/train.csv"), encoding="big5",header=None,
    "./data/train.csv", encoding="big5", header=None,
    usecols=cols  # 只取有用的一些列，前三列丢掉
)
tdata = tdata[1:]  # 去掉第一行
tdata = tdata.reset_index(drop=True)  # 重新生成索引，从0开始，便于访问
tdata.columns -= 3  # 索引从0开始便于访问
# nr的字符串改为0，便于后面进行数值运算
tdata = tdata.applymap(lambda x: "0" if x == "NR" else x)

ngroup = 10  # 每10小时为一组
nc = 18  # 18个指标
pm_position = 9  # PM2.5数值所在的列标号
cur_hours = 0  # 记录已经纳入到当前分组中的记录小时数
rownum = 0  # 当前记录的行号
# 10个小时数据一组，前9小时的所有数据作为输入，第10小时的PM2.5的值放在最后一列作为目标
mtrain = np.ndarray(shape=(math.floor(
    tdata.shape[0]*tdata.shape[1]/(ngroup*nc)), nc*(ngroup-1)+1), dtype=float)  # 创建未初始化的矩阵
for i in range(0, tdata.shape[0]-nc+1, nc):  # 18行为一个子表
    for j in range(0, tdata.shape[1]):  # 遍历每个子表的24个小时数据
        cur = tdata.loc[i:i+nc-1, j].to_numpy()
        cur_hours += 1
        if cur_hours == ngroup:  # 已经获得了10个小时的数据
            mtrain[rownum, mtrain.shape[1]-1] = cur[pm_position]
            rownum += 1
            cur_hours = 0
            continue  # 第10行数据只把PM2.5填入
        mtrain[rownum, (cur_hours-1)*nc:(cur_hours-1)*nc+nc] = cur


# 数据标准化
m_mean = np.mean(mtrain, axis=0)
m_std = np.std(mtrain, axis=0)
std_train = np.zeros(mtrain.shape)
for i in range(mtrain.shape[1]-1):  # 对所有的x进行标准化
    std_train[:, i] = (mtrain[:, i]-m_mean[i])/m_std[i]
std_train[:, mtrain.shape[1]-1] = mtrain[:, mtrain.shape[1]-1]  # t这一列不要标准化

# 分割训练集和验证集
len_train = math.floor(0.7*len(std_train))  # 训练集行数
x_train = std_train[:len_train, 0:std_train.shape[1]-1]  # nbatch行的矩阵作为输入参数
t_train = std_train[:len_train, std_train.shape[1]-1]
x_validation = std_train[len_train:std_train.shape[0], 0:std_train.shape[1]-1]
t_validation = std_train[len_train:, std_train.shape[1]-1]

# 线性回归计算,数据比较少，直接放进去计算loss


def Fy(wb, x):
    # 建立目标函数式：Fy=wx+b
    # w,b分别为权值向量和偏移量,x为nbacth行的矩阵
    # 返回y向量
    return np.dot(x, wb[:wb.shape[0]-1])+wb[wb.shape[0]-1]


def Lw(t, y, wb, r):
    # 损失函数Lw=(1/n)(∑(ti-yi)^2+λ∑wi^2)
    # t,y,w,分别为目标值向量、拟合值向量、权值向量
    # r
    # 返回平均损失函数值
    c = t-y
    return (np.dot(c, c)+r*np.dot(wb[:wb.shape[0]-1], wb[:wb.shape[0]-1]))/t.shape[0]


def Pwb(t, y, x, wb, r):
    # wi的偏导Pw：-∑(tk-yk)*xki+2λwi,xki表示yk对应的第i个变量
    # b的偏导Pb：-∑(ti-yi)
    # t,y,x,w分别为目标值向量、拟合值向量、变量向量、权值向量
    # r
    # 返回w和的梯度向量，b放在最后一个位置，求的平均梯度
    ret = np.zeros(x.shape[1]+1)
    ret[:x.shape[1]] = (2*np.dot(y-t, x)+2*r*wb[:x.shape[1]])/x.shape[0]
    ret[x.shape[1]] = (y-t).sum()/x.shape[0]
    return ret


# 初始化w为0向量，b为0;w的迭代关系：w=w-rate*pw;b的迭代关系：b=b-rate*pb;
r = 0  # 正则项系数
rate = 5e-3
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
momentum = 0
v = 0
off = 1e-12  # 精度
wb = np.random.randn(x_train.shape[1]+1)
loss = 0
beta1_power = 1
beta2_power = 1
iters = math.floor(3e4)
for i in range(iters):  # 梯度下降，使梯度尽量接近0,最多5W次
    x_index = [i for i in range(x_train.shape[0])]
    x = x_train[x_index]
    t = t_train[x_index]
    y = Fy(wb, x)
    loss_pre = loss
    loss = Lw(t, y, wb, r)
    if (i+1)%100 == 0:
        print(i+1, " loss:", loss)
    g = Pwb(t, y, x, wb, r)
    beta1_power *= beta1
    beta2_power *= beta2
    momentum = (beta1*momentum+(1-beta1)*g)
    v = beta2*v+(1-beta2)*(g**2)  # 此处不能除以常用的分母1-beta1，否则不能到达最优点
    dwb = rate*momentum/(np.sqrt(v)+eps)
    if abs((loss-loss_pre)/loss) <= off:  # loss基本不再变化，则停止
        print("loss no longer change,break.")
        break
    wb -= dwb
# 用验证集进行验证
y_train = Fy(wb, x_train)
loss_train = Lw(t_train, y_train, wb, r)
y_validation = Fy(wb, x_validation)
loss_validation = Lw(t_validation, y_validation, wb, r)
print("loss:\n",  \
    "train loss:",loss_train, \
    "\nvalidation loss:", loss_validation)

# 用pytorch计算
D_in, D_out = x_train.shape[1], 1
model = torch.nn.Sequential(torch.nn.Linear(D_in, D_out)) #整个网络模块
loss_func= torch.nn.MSELoss(reduction="mean")             #损失函数
opt = torch.optim.Adam(model.parameters(),lr = 1e-2)      #优化器，选用Adam算法
x_torch = torch.from_numpy(x_train).float()
t_torch = torch.from_numpy(t_train).float().unsqueeze(1)
iters = 5000
loss = np.zeros(iters)
for i in range(iters):
    y_pred = model(x_torch) #计算输出
    loss_torch = loss_func(y_pred,t_torch) #损失函数
    loss[i] = loss_torch.detach().numpy()
    if (i+1) % 100 == 0:
        print(i+1," loss:", loss_torch)
    opt.zero_grad() #计算之前清零梯度
    loss_torch.backward() #反向传播计算梯度
    opt.step()    #梯度下降更新参数
#绘图
plt.figure()
plt.plot([x for x in range(0, iters)], loss)
plt.xlabel("iters")
plt.ylabel("loss")
plt.show()

y_train = model(x_torch)
loss_train = loss_func(y_train,t_torch).detach().numpy().squeeze()
y_validation = model(torch.from_numpy(x_validation).float())
loss_validation = loss_func(y_validation,torch.from_numpy(t_validation).unsqueeze(1)
              ).detach().numpy().squeeze()
print("torch loss:\n",  \
    "train loss:",loss_train, \
    "\nvalidation loss:", loss_validation)

# test
cols = [i for i in range(2, 11)]
testdata = pd.read_csv(
    "./data/test.csv", encoding="big5", header=None,
    usecols=cols
)
testdata = testdata.applymap(lambda x: "0" if x == "NR" else x)
testdata.columns = testdata.columns-2
x_test = np.zeros(shape=(math.floor(testdata.shape[0]/nc), (ngroup-1)*nc))
for i in range(0, testdata.shape[0]-nc+1, nc):
    x = np.zeros((ngroup-1)*nc)
    for j in range(ngroup-1):
        # 注意DataFrame.loc[i:j,t]包含第j行，DataFram[i:j]不包含
        x[j*nc:j*nc+nc] = testdata.loc[i:i+nc-1, j].to_numpy()
    x_test[math.floor(i/nc)] = (x-m_mean[0:m_mean.shape[0]-1]
                                )/m_std[0:m_std.shape[0]-1]  # 标准化
y_test = Fy(wb, x_validation)

print("predictions:", y_test)
ids = ["id_"+str(x) for x in range(0, y_test.shape[0])]
pd.DataFrame({"id": ids, "value": list(y_test)}).to_csv(
    "./data/result.csv", index=False)
