# 翻译模型的实现
# 主要由四部分组成：encoder、decoder、attention,translater

import torch
from torch import nn


# 输入整个句子，输出rnn的output和h
class Encoder(nn.Module):
    def __init__(
        self,
        vsize,  # 字典大小
        en_word_dim,  # embedding的输出维度
        hidden_size,  # Encoder中GRU的hidden_size
        num_layers,
    ):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vsize, en_word_dim)
        self.rnn = nn.GRU(
            en_word_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=0.5,
            bidirectional=True,  # 上下文可能都会都依赖
        )

    def forward(self, x):
        # x:[batch, seq_len]
        x = self.embedding(x)
        # batch_first=True￼
        output, h = self.rnn(x)
        # output:[batch, seq_len, h_size*num_directions]
        # h:[num_layers * 2, batch, hidden_size]
        return output, h


class Decoder(nn.Module):
    def __init__(
        self,
        vsize,  # 字典大小
        cn_word_dim,  # embedding的输出维度
        encoder_hidden_size,  # Encoder中双向GRU的hidden_size
        num_layers,
    ):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vsize, cn_word_dim)
        self.rnn = nn.GRU(
            cn_word_dim,
            # Decoder中GRU的hidden_size为encoder_hidden_size的两倍，
            # 即Encoder中的双向h拼接在一起作为Decoder中单向GRU的context
            encoder_hidden_size * 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=False,  # 翻译的时候只从头往后推断
        )
        # h转换为onehot编码分布
        self.hidden2onehot = nn.Sequential(
            nn.Linear(encoder_hidden_size * 2, vsize),
        )

    def forward(self, x, h):
        # 输入一个字符x，预测下一个字符
        # 这个输入的字符应该是cn字符
        # x: [batch]
        # h: [num_layers * 1, batch, encoder_hidden_size*2]
        x = x.unsqueeze(1)
        x = self.embedding(x)
        _, h = self.rnn(x)
        # 使用最后一层的h来进行预测
        x = self.hidden2onehot(h[-1])
        # x:[batch,cn_word_dim]
        # h:[num_layers * num_directions, batch, encoder_hidden_size*2]
        return x, h


# 输入encoder的output和decoder的h，返回新的h作为decoder的context
# 参数通过自动学习得到,可以得到合理的attention计算方法
# https://arxiv.org/abs/1409.0473
class Attention(nn.Module):
    def __init__(
        self,
        encoder_hidden_size,  # Encoder中双向GRU的hidden_size
    ):
        super(Attention, self).__init__()
        self.attention_func = nn.Sequential(
            nn.Linear(2 * encoder_hidden_size, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 2 * encoder_hidden_size),
        )

    def forward(self, x, y):
        # x:Encoder中的output[batch, seq_len, encoder_hidden_size*2]
        # y:Decoder的隐藏层[num_layers * 1, batch, encoder_hidden_size*2]
        # 返回Attention方法得到的Decoder的context
        # 使用xwy计算得到权系数a，再对x带权求和得到context
        # 每个隐藏层分别和x进行attention运算
        # 权重矩阵weights[batch, seq_len, num_layers]
        # 高纬矩阵乘法matmul,只匹配最后两个维度，前面的维度若不一致会作braodcast转换
        softmax = nn.Softmax(dim=1)
        weights = self.attention_func(x).matmul(y.permute(1, 2, 0))
        weights = softmax(weights)
        # 对encoder的各个h带权求和即可得到attention结果
        # 这个地方注意返回size应该和y保持一致,因此最后一步进行转置
        context = x.permute(0, 2, 1).matmul(weights).permute(2, 0, 1)
        return context


# 主要由四部分组成：encoder、decoder、attention,translater
class EN2CN(nn.Module):
    def __init__(
        self,
        en_vsize,  # 英文字典大小
        cn_vsize,  # 中文字典大小
        sampling=1.0,  # sampling的概率
    ):
        super(EN2CN, self).__init__()
        self.hsize = 512
        self.rnn_layers = 5
        self.sampling = sampling
        self.cn_vsize = cn_vsize
        self.encoder = Encoder(en_vsize, 512, self.hsize, self.rnn_layers)
        self.decoder = Decoder(cn_vsize, 512, self.hsize, self.rnn_layers)
        self.attention = Attention(self.hsize)

    def forward(
        self,
        x,  # en语句
        y=None,  # cn语句，当正式翻译的时候不要输入此参数
    ):
        # 训练的时候由于使用sampling，所以会使用y即正确结果
        # 正式翻译的时候不需要输入y
        return self.__inference__(x) if y == None else self.__train__(x, y)

    # 训练的时候调用此函数
    def __train__(self, x, y):
        # en语句x:[batch, seq_len1]
        # cn语句y:[batch, seq_len2]
        # 要使用sampling需要依赖y,有概率直接用预测的值而不是y中的值
        encoder_ouput, h = self.encoder(x)
        # encoder的输出h[num_layers * 2, batch, encoder_hidden_size]
        # decoder的输入h[num_layers * 1, batch, encoder_hidden_size*2]
        h = torch.cat((h[: self.rnn_layers], h[self.rnn_layers :]), 2)
        input = y[:, 0]  # 起始符
        # 预测的one-hot分布
        ret = torch.zeros((y.shape[0], y.shape[1], self.cn_vsize), device=x.device)
        ret[:, 0, y[0, 0]] = 1.0  # 第一个都是是起始字符，放入
        for i in range(1, y.shape[1]):  # 预测出和y等长的语句
            out, h = self.decoder(input, h)
            h = self.attention(encoder_ouput, h)
            ret[:, i] = out
            pred_next = out.topk(1, axis=1)[1].squeeze()
            input = pred_next if torch.rand((1,)).item() < self.sampling else y[:, i]

        return ret

    # 正式翻译的时候会调用此函数
    def __inference__(self, x):
        # en语句x:[batch, seq_len1]
        return None


# test
if __name__ == "__main__":

    mm = EN2CN(100, 200)
    r = mm(torch.randint(0, 100, (3, 20)), torch.randint(0, 200, (3, 30)))
