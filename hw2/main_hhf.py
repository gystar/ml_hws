import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import random

# 从csv文件中读入数据，注意此繁体中文csv使用的是big5编码
tb_x_train = pd.read_csv(
    "./data/X_train", encoding="big5", header=None
)
tb_y_train = pd.read_csv(
    "./data/Y_train", encoding="big5", header=None
)
x_total = tb_x_train.loc[1:, 1:].to_numpy(dtype=int)
t_total = tb_y_train.loc[1:, 1].to_numpy(dtype=int)

# 划分为两个集合c1,c2；分别表示不超过50K和超过50K,且选取1：1的比例
N_C = min(t_total.shape[0]-np.count_nonzero(t_total),
          np.count_nonzero(t_total))
x_c1 = np.zeros(shape=(N_C, x_total.shape[1]))
x_c2 = np.zeros(shape=(N_C, x_total.shape[1]))
i1 = 0
i2 = 0
for i in range(x_total.shape[0]):
    if t_total[i] == 0:
        if i1 < N_C:  # 使两个类别的数据达到1：1
            x_c1[i1] = x_total[i]
            i1 += 1
    else:
        if i2 < N_C:  # 使两个类别的数据达到1：1
            x_c2[i2] = x_total[i]
            i2 += 1

# 分割训练集和验证集
# 训练集
len_train = math.floor(N_C*0.7)
x_c1_train = x_c1[:len_train]
x_c2_train = x_c2[:len_train]
x_train = np.zeros((len_train*2, x_c1_train.shape[1]))
x_train[:len_train] = x_c1_train
x_train[len_train:] = x_c2_train
t_train = np.zeros(len_train*2)
t_train[len_train:] = 1

# 测试集
x_validation = np.zeros(((N_C-len_train)*2, x_c1_train.shape[1]))
x_validation[:N_C-len_train] = x_c1[len_train:]
x_validation[N_C-len_train:] = x_c2[len_train:]
t_validation = np.zeros((N_C-len_train)*2)
t_validation[N_C-len_train:] = 1

# 去掉数值完全一样的列(线性相关的列全部找到代价太大)，可能导致协方差矩阵奇异
cols = np.full((x_train.shape[1],), True, dtype=bool)
for i in range(x_train.shape[1]):
    for j in range(x_train.shape[0]):
        if x_train[j, i] != x_train[0, i]:
            break
        if j == x_train.shape[0]-1:
            cols[i] = False  # 此列的所有值都相同
x_train = x_train[:, cols]
x_c1_train = x_c1_train[:, cols]
x_c2_train = x_c2_train[:, cols]
x_validation = x_validation[:, cols]


# 1.用高斯概率生成模型来分类，具体说明见高斯generate.pdf
# 计算均值和协方差矩阵的最大似然估计
# c1中x的均值的最大似然估计
#u1 =x_c1.mean()
u1 = np.mean(x_c1_train, axis=0)
# c2中x的均值的最大似然估计
#u1 =x_c2.mean()
u2 = np.mean(x_c2_train, axis=0)


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
x_cov = (f1(x_c1_train, u1) + f1(x_c2_train, u2))/(N_C*2)
print("the norm of x_cov is very big:", np.linalg.norm(x_cov))
#x_cov的条件数很大，可能是个病态矩阵，使用np.linalg.inv求出的结果不正确，导致预测结果不可信
#x_cov_inverse = np.linalg.inv(x_cov)  
x_cov_inverse = x_cov #此处为奇异矩阵先不计算



# P(c1|x) = lim P(x|c1)P(c1)/{P(x|c1)P(c1)+P(x|c2)P(c2)} (接近x时的极限)    为什么要用极限？
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
        z = -(x[i]-u1).dot(cov_inverse).dot(x[i]-u1)/2 \
            + (x[i]-u2).dot(cov_inverse).dot((x[i]-u2).transpose())/2 \
            + np.log(n1/n2)
        ret[i] = 1/(1+np.exp(-1*z))
    return ret


# 用验证集验证
y_validation = F1_pc1x(u1, u2, N_C, N_C, x_cov_inverse, x_validation)
count_right = 0
for i in range(y_validation.shape[0]):
    bres = 1 if (y_validation[i] > 0.5) else 0
    if bres == t_validation[i]:
        count_right += 1
print("right percent of generative model:",
      100*count_right/y_validation.shape[0], "%")

# 2.用logistics的discriminative方法来分类


#y(w,b)=1/(1+exp{-1*(z)})
def Fy(wb,x):
    w=wb[:x_train.shape[1]]
    b=wb[x_train.shape[1]]
    z=np.dot(x,w)+b
    return 1/(1+np.exp(-1*z))
    

#定义损失函数L(t,y)
#L(t,y)=-{\sum{t*Lny+(1-t)*Ln(1-y)}}
def L(t,y):   
    sum_l = 0
    for i in range(t.shape[0]):
        if (t[i]==1):
            sum_l += -1*np.log(y[i])
        else:
            sum_l += -1*np.log(1-y[i])
    return sum_l


#分别对w和b求偏导
def pwb(t,wb,x,y):
    g=np.zeros(x.shape[1]+1)
    g[:x.shape[1]]=-1*(np.dot(t-y,x))/x.shape[0]
    g[x.shape[1]]=-1*(t-y).sum()/x.shape[0]
    return g


#使用Adam梯度下降法
#m=betal1*m+(1-betal1)g     其中g表示偏导（Pw,pb）
#v=betal2*v+(1-betal2)g**2
#wb=-{(rate*m)/(sqrt(v)+eps)}+wb  
betal1=0.9
betal2=0.999
rate=1e-4
eps=1e-8
m=np.zeros(x_train.shape[1]+1)
v=0
wb=np.zeros(x_train.shape[1]+1)
nbatch=100
for _ in range(math.floor(1e4)):
    r=np.random.randint(x_train.shape[0],size=nbatch)
    x_1=x_train[r]
    t_1=t_train[r]
    y_1=Fy(wb,x_1)
    g=pwb(t_1,wb,x_1,y_1)
    m=betal1*m+(1-betal1)*g
    v=betal2*v+(1-betal2)*(g**2)
    print("loss=",L(t_1,y_1))
    wb-=((rate*m)/(np.sqrt(v)+eps))



# 用验证集验证
y_validation = Fy(wb, x_validation)
count_right = 0
for i in range(y_validation.shape[0]):
    bres = 1 if (y_validation[i] > 0.5) else 0
    if bres == t_validation[i]:
        count_right += 1
print("right percent of logistics model:", 100 *
      count_right/y_validation.shape[0], "%")

#test
tb_x_test = pd.read_csv(
    "./data/X_test", encoding="big5", header=None
)
x_test = tb_x_test.loc[1:, 1:].to_numpy(dtype=int)
x_test = x_test[:,cols]
y_test = F2_pc1x(wb, x_test)
value = [0 if x <0.5 else 1 for x in y_test]
pd.DataFrame({"id": [x for x in range(1,y_test.shape[0]+1)], "value": value}).to_csv(
    "./data/result.csv", index=False)