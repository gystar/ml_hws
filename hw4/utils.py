# 一些常用函数的实现
import numpy as np
import torch


def p2label(p):
    # 概率转化为0或者1
    # 返回分类标志数组
    ret = [0 if i < 0.5 else 1 for i in p]
    return ret


def calc_accuracy(t, y):
    # 计算正确率
    count_right = 0
    for i in range(t.shape[0]):
        if t[i] == y[i]:
            count_right += 1
    return count_right/t.shape[0]


'''
# lstm测试代码,对于理解pack和lstm很有帮助
a1 = torch.tensor([1, 2, 3, 4])
a2 = torch.tensor([6])
a3 = torch.tensor([4, 5])
lengths = [len(a1), len(a2), len(a3)]
a4 = torch.nn.utils.rnn.pad_sequence([a1, a2, a3])

# 每个元素转化为词向量
word_dim = 2
emb = torch.nn.Embedding(20, word_dim, padding_idx=0)

lstm = torch.nn.LSTM(input_size=word_dim, hidden_size=7, batch_first=False)

a_input = emb(a4)
a_packed_input = torch.nn.utils.rnn.pack_padded_sequence(
    input=a_input, lengths=lengths, batch_first=False, enforce_sorted=False)
packed_out, (hn, cn) = lstm(a_packed_input)
out, hc = pad_packed_sequence(packed_out, batch_first=False)
'''


def sens2vecs(sens, wv):
    # 将句子转换为词向量矩阵，并且pack好准备给lstm使用
    # sens为句子list，wv为词向量字典
    # 用训练好的字典将语句转化为词向量矩阵
    vecs = [torch.tensor(wv[sen]) for sen in sens]
    lengths = [len(sen) for sen in sens]
    # 用0将各个句子填充成等长
    vecs = torch.nn.utils.rnn.pad_sequence(vecs)
    # 将填充的0去掉，所有句子打包为一个一维数组给lstm使用
    vecs = torch.nn.utils.rnn.pack_padded_sequence(
        input=vecs, lengths=lengths,
        batch_first=False,  # batch在第二维度
        enforce_sorted=False  # 句子长度乱序，会自动排列成递减
    )
    return vecs
