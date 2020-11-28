import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import random
import copy
import torch

# 从csv文件中读入数据，注意此繁体中文csv使用的是big5编码
tb_x_train = pd.read_csv(
    "./data/X_train", encoding="big5", header=None
)
tb_y_train = pd.read_csv(
    "./data/Y_train", encoding="big5", header=None
)
x_total = tb_x_train.loc[1:, 1:].to_numpy(dtype=int)
t_total = tb_y_train.loc[1:, 1].to_numpy(dtype=int)

# 标准化
x_mean = x_total.mean(axis=0)
x_std = x_total.std(axis=0)
for i in range(x_total.shape[1]):
    if x_std[i] > 0:
        x_total[:, i] = (x_total[:, i]-x_mean[i])/x_std[i]

# 分割训练集和验证集
# 训练集
len_train = math.floor(t_total.shape[0]*0.7)
x_train = x_total[:len_train]
t_train = t_total[:len_train]
# 测试集
x_validation = x_total[len_train:]
t_validation = t_total[len_train:]


def Probability2Class(p, tag=True):
    # 将p(c1|x)的概率转化为0或者1
    # p(c1|x)小于0.5分类为0，否则分类为1
    # 输入x为c1的概率p
    # 返回分类标志数组
    ret = np.zeros(p.shape[0])
    for i in range(p.shape[0]):
        ret[i] = 0 if (p[i] < 0.5) else 1
    return ret


def CalcAccuracy(t, y):
    # 计算正确率
    count_right = 0
    for i in range(t.shape[0]):
        if t[i] == y[i]:
            count_right += 1
    return count_right/t.shape[0]


# 1.用高斯概率生成模型来分类，具体说明见高斯generate.pdf
# 划分为两个集合c1,c2；分别表示超过50K和不超过50K
N1 = np.count_nonzero(t_train)
N2 = t_train.shape[0]-N1
x_c1 = np.zeros(shape=(N1, x_train.shape[1]))
x_c2 = np.zeros(shape=(N2, x_train.shape[1]))
i1 = 0
i2 = 0
for i in range(x_train.shape[0]):
    if t_train[i] == 1:
        x_c1[i1] = x_train[i]
        i1 += 1
    else:
        x_c2[i2] = x_train[i]
        i2 += 1
# 计算均值和协方差矩阵的最大似然估计
# c1中x的均值的最大似然估计
#u1 =x_c1.mean()
u1 = np.mean(x_c1, axis=0)
# c2中x的均值的最大似然估计
#u1 =x_c2.mean()
u2 = np.mean(x_c2, axis=0)


def f1(x, u):
    # 定义x的运算：∑(xi-u)*transpose(xi-u)
    # x的列表示维度
    s = np.zeros(shape=(x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        a = x[i]-u
        s += np.dot(a.reshape(a.shape[0], 1), a.reshape(1, a.shape[0]))
    return s


# x的协方差矩阵的最大似然估计
# cov = (N1/N)*((1/N1)*∑(xi-u1)*transpose(xi-u1))+(N2/N)*((1/N2)*∑(xi-u2)*transpose(xi-u2))
x_cov = (f1(x_c1, u1) + f1(x_c2, u2))/(N1+N2)
# x_cov的条件数很大，可能是个病态矩阵，使用np.linalg.inv求出的结果不正确，导致预测结果不可信
# 通过svd分解算出逆矩阵，避免x_cov接近奇异矩阵的情况
u, s, v = np.linalg.svd(x_cov, full_matrices=False)
x_cov_inverse = np.matmul(v.T * 1 / s, u.T)


# P(c1|x) = lim P(x|c1)P(c1)/{P(x|c1)P(c1)+P(x|c2)P(c2)} (接近x时的极限)
# 根据高斯分布的概率密度公式可以化为:
# P(c1|x) = 1/(1+exp(-z)，其中：
# z=transpose(u1-u2)*inverse(cov)*x
# -0.5*transpose(u1)*inverse(cov)*u1
# +0.5*transpose(u2)*inverse(cov)*u2
# +In(N1/N2)


def F1_pc1x(u1, u2, n1, n2, cov_inverse, x):
    # 定义P(C1|x)的函数，返回概率向量
    ret = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z = -1*(x[i]-u1).dot(cov_inverse).dot(x[i]-u1)/2 \
            + (x[i]-u2).dot(cov_inverse).dot((x[i]-u2).transpose())/2 \
            + np.log(n1/n2)
        ret[i] = 1/(1+np.exp(-1*z))
    return ret


# 用验证集验证
y_validation = Probability2Class(
    F1_pc1x(u1, u2, N1, N2, x_cov_inverse, x_validation))
print("The accuracy of generative model:",
      CalcAccuracy(t_validation, y_validation)*100, "%")

# 2.用logistics的discriminative方法来分类


def F2_pc1x(wb, x):
    # 用logistics返回P(c1|x)的概率，输入向量wb，x向量
    # c1表示大于50K，对应的t值为1
    #p = 1(1+exp(-1*(wx+b)))
    # 返回概率向量
    z = np.dot(x, wb[0:wb.shape[0]-1])+wb[wb.shape[0]-1]
    p = 1/(1+np.power(math.e, -1*z))
    return p


def Lwb(t, y):
    # 损失函数L(w,b) = -∑{tn*In(yn)+(1-tn)*In(1-yn)}
    # 对最大概率估计函数取对数加负号得到
    # 输入目标值向量t，函数值向量y
    # 返回损失值
    sum_loss = 0  # 注意计算机处理0*inf的时候回返回NAN,所以手动累加
    for i in range(y.shape[0]):
        sum_loss += np.log(1-y[i]) if t[i] == 0 else np.log(y[i])
    return -1*sum_loss/t.shape[0]


def Pwb(t, y, x):
    # 求w和b的偏导，输入目标值向量t，函数值向量y和参数矩阵x
    #pw = -(t-y)*x,注意要求均值
    # pb = -∑(t-y)，注意要求均值
    # 返回w和b的偏导，b的偏导放在最后
    ret = np.zeros(x.shape[1]+1)
    ret[:x.shape[1]] = np.dot(y-t, x)/x.shape[0]
    ret[x.shape[1]] = (y-t).sum()/x.shape[0]
    return ret


def is_zero(v, off):
    # 向量v的每个分量绝对值不大于off则判定v为0向量
    # 输入向量v，绝对值下限off
    # 返回是否0向量
    for i in range(v.shape[0]):
        if abs(v[i]) > off:
            return False
    return True


# 使用Adam算法来进行梯度下降
rate = 1e-4
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
momentum = 0
v = 0
wb = np.zeros(x_train.shape[1]+1)
beta1_power = 1
beta2_power = 1

nbatch = 100  # 随机选取nbatch个x来进行梯度下降
iters = 30  # 进行循环的轮数
loss = np.zeros(iters+1)
print("try to SGD with Adam algorithm...")
for i in range(iters):
    loss[i] = Lwb(t_train, F2_pc1x(wb, x_train))
    for _ in range(math.ceil(x_train.shape[0]/nbatch)):
        x_index = np.random.randint(x_train.shape[0], size=nbatch)
        x = x_train[x_index]
        t = t_train[x_index]
        y = F2_pc1x(wb, x)
        g = Pwb(t, y, x)
        beta1_power *= beta1
        beta2_power *= beta2
        momentum = beta1*momentum+(1-beta1)*g
        v = beta2*v+(1-beta2)*(g**2)    
        dwb = rate*momentum/(np.sqrt(v)+eps)
        wb -= dwb
wb_Adam = copy.deepcopy(wb)
loss[iters] = Lwb(t_train, F2_pc1x(wb_Adam, x_train))
# 绘图
plt.figure()
plt.plot([x for x in range(0, iters+1)], loss)
plt.xlabel("iters")
plt.ylabel("loss")
plt.show()
# 用验证集验证
y_validation = Probability2Class(F2_pc1x(wb_Adam, x_validation))
print("The accuracy of logistics model with Adam:", 100 *
      CalcAccuracy(t_validation, y_validation), "%")

#使用SGDM算法来进行梯度下降  
rate = 5e-3
beta1 = 0.9
nbatch = 100  # 随机选取nbatch个x来进行梯度下降
iters = 50  # 进行循环的轮数
loss = np.zeros(iters+1)
v= 0
wb = np.zeros(x_train.shape[1]+1)
print("try to SGD with SGDM algorithm...")
for i in range(iters):
    loss[i] = Lwb(t_train, F2_pc1x(wb, x_train))
    for _ in range(math.ceil(x_train.shape[0]/nbatch)):
        x_index = np.random.randint(x_train.shape[0], size=nbatch)
        x = x_train[x_index]
        t = t_train[x_index]
        y = F2_pc1x(wb, x)
        g = Pwb(t, y, x)     
        v = beta1*v-rate*g       
        wb += v  
wb_SGDM  = copy.deepcopy(wb)         
loss[iters] = Lwb(t_train, F2_pc1x(wb_SGDM, x_train))
# 绘图
plt.figure()
plt.plot([x for x in range(0, iters+1)], loss)
plt.xlabel("iters")
plt.ylabel("loss")
plt.show()
# 用验证集验证
y_validation = Probability2Class(F2_pc1x(wb_SGDM, x_validation))
print("The accuracy of logistics model with SDMG:", 100 *
      CalcAccuracy(t_validation, y_validation), "%")

##用pytorch来进行计算
model = torch.nn.Sequential(#构建序列模型
    torch.nn.Linear(x_train.shape[1], 1),    
    torch.nn.Sigmoid()
)
loss_func = torch.nn.BCELoss()  #损失函数
opt = torch.optim.Adam(model.parameters(), lr = 1e-3)#优化器
x_torch = torch.from_numpy(x_train).float()
t_torch = torch.from_numpy(t_train).unsqueeze(1).float()
iters = 2000
loss = np.zeros(iters)
print("Try to calc with pytorch model of Adam...")
for i in range(iters):      
    y_pred = model(x_torch) #计算输出
    loss_cur = loss_func(y_pred, t_torch) #计算损失
    if (i+1) %99 == 0:
        print(i+1, " loss:", loss_cur.detach().numpy().squeeze())   
    loss[i] = loss_cur.detach().numpy().squeeze()    
    opt.zero_grad() #计算之前清零之前的梯度
    loss_cur.backward() #反向传播自动计算梯度
    opt.step()#梯度下降更新参数
# 绘图
plt.figure()
plt.plot([x for x in range(0, iters)], loss)
plt.xlabel("iters")
plt.ylabel("loss")
plt.show()
#验证
y_train = Probability2Class(model(x_torch).detach().squeeze().numpy())
y_validation = Probability2Class(model(torch.from_numpy(x_validation).float()).detach().squeeze().numpy())
print("The accuracy of pytorch model with Adam:\n", \
    "test:",100 *CalcAccuracy(t_validation, y_train), "%" \
    "\nvalidation:",100 *CalcAccuracy(t_validation, y_validation), "%")

# test
tb_x_test = pd.read_csv(
    "./data/X_test", encoding="big5", header=None
)
x_test = tb_x_test.loc[1:, 1:].to_numpy(dtype=int)
# 标准化
for i in range(x_test.shape[1]):
    if x_std[i] > 0:
        x_test[:, i] = (x_test[:, i]-x_mean[i])/x_std[i]
y_test = Probability2Class(model(torch.from_numpy(x_test).float()).detach().squeeze().numpy())
pd.DataFrame({"id": [x for x in range(0, y_test.shape[0])], "label": list(y_test)}).to_csv(
    "./data/result.csv", index=False)
