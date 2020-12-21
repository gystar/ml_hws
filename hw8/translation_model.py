# 翻译模型的实现
# 主要由三部分组成：encoder、decoder、attention,translater

import torch
from torch import nn


# 输入整个句子，输出rnn的output和h
class Encoder(nn.Module):
    def __init__(
        self,
        vsize,  # 字典大小
        word_dim,  # embedding的输出维度
        hidden_size,  # Encoder中GRU的hidden_size
    ):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vsize, word_dim)
        self.rnn = nn.GRU(
            word_dim,
            hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,  # 双向RNN，因为翻译的时候后文有可能会影响前文；反向的h就是逆序独立计算的
        )

    def forward(self, x):
        # x:[batch, seq_len]
        x = self.embedding(x)
        # batch_first=True￼
        output, h = self.rnn(x)
        # output:[batch, seq_len, h_size*num_directions]
        # h:[num_layers * 2, batch, hidden_size]

        return output, h


# 输入encoder的output和decoder的h，返回新的h作为decoder的context
# 参数通过自动学习得到,可以得到合理的attention计算方法
class Attention(nn.Module):
    def __init__(
        self,
        encoder_hidden_size,  # Encoder中双向GRU的hidden_size
    ):
        super(Attention, self).__init__()
        self.attention_func = nn.Sequential(
            nn.Linear(2 * encoder_hidden_size, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 2 * encoder_hidden_size),
        )

    def forward(self, x, y):
        # x:Encoder中的output[batch, seq_len, encoder_hidden_size*2]
        # y:Decoder的隐藏层[num_layers * 1, batch, encoder_hidden_size*2]
        # 返回Attention方法得到的Decoder的context
        # 使用xwy计算得到权系数a，再对x带权求和得到context
        # 每个隐藏层分别和x进行attention运算
        # 权重矩阵weights[batch, seq_len, num_layers]
        # 高纬矩阵乘法matmul,只匹配最后两个维度，前面的维度若不一致会作braodcast转换
        weights = self.attention_func(x).matmul(y.permute(1, 2, 0))
        # 带权求和即可得到attention结果的context
        # 这个地方注意返回size应该和y保持一致,因此最后一步进行转置
        context = x.permute(0, 2, 1).matmul(weights).permute(2, 0, 1)
        return context


class Decoder(nn.Module):
    def __init__(
        self,
        vsize,  # 字典大小
        word_dim,  # embedding的输出维度
        encoder_hidden_size,  # Encoder中双向GRU的hidden_size
    ):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vsize, word_dim)
        self.rnn = nn.GRU(
            word_dim,
            # Decoder中GRU的hidden_size为encoder_hidden_size的两倍，
            # 即Encoder中的双向h拼接在一起作为Decoder中单向GRU的context
            encoder_hidden_size * 2,
            num_layers=3,
            batch_first=True,
            dropout=0.5,
            bidirectional=False,  # 翻译的时候只从头往后推断
        )

    def forward(self, x, h):
        # x: [batch,seq_len]
        # h: [num_layers * 1, batch, encoder_hidden_size*2]
        x = self.embedding(x)

        return None


class Translater(nn.Module):
    def __init__(self):
        super(Translater, self).__init__()


# test
if __name__ == "__main__":
    x = torch.randn((3, 5, 6))
    gru = nn.GRU(6, hidden_size=7, num_layers=5, batch_first=True, bidirectional=True)
    a, b = gru(x)
    nn.LSTM
    x.view

    a = torch.randn((3, 10, 16 * 2))
    b = torch.randn((7, 3, 16 * 2))
    att = Attention(16)
    c = att(a, b)
