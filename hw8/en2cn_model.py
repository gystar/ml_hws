# 翻译模型的实现
# 主要由四部分组成：encoder、decoder、attention,translater

import torch
from torch import nn
import heapq


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
            batch_first=True,
        )

    def forward(self, x):
        # x:[batch, seq_len]
        x = self.embedding(x)
        # batch_first=True￼
        output, h = self.rnn(x)
        # output:[batch, seq_len, h_size*2]
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
            # 即Encoder中的双向h拼接在一起作为Decoder中单向GRU的conheapq.nlargest(2, a, key=lambda x: x[0])text
            encoder_hidden_size * 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=False,  # 翻译的时候只从头往后推断
        )
        # h转换为onehot编码分布
        self.hidden2onehot = nn.Sequential(
            nn.Linear(encoder_hidden_size * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, vsize),
        )

    def forward(self, x, h):
        # 输入一个字符x，预测下一个字符
        # 这个输入的字符应该是cn字符
        # x: [batch]
        # h: [num_layers * 1, batch, encoder_hidden_size*2]
        x = x.unsqueeze(1)
        x = self.embedding(x)
        _, h = self.rnn(x, h.contiguous())
        # 使用最后一层的h来进行预测
        x = self.hidden2onehot(h[-1])
        # x:[batch,cn_word_dim]
        # h:[num_layers * 1, batch, encoder_hidden_size*2]
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
        # weight = encoder_output*w*decoder_hidden
        self.weight_func = nn.Sequential(
            nn.Linear(2 * encoder_hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * encoder_hidden_size),
        )
        self.combine_func = nn.Sequential(
            nn.Linear(4 * encoder_hidden_size, 2 * encoder_hidden_size),
        )

    def forward(self, x, y):
        # x:Encoder中的output[batch, seq_len, encoder_hidden_size*2]
        # y:Decoder的隐藏层[num_layers * 1, batch, encoder_hidden_size*2]
        # 返回Attention方法得到的Decoder的context
        # 使用xwy计算得到权系数a，再对x带权求和得到context
        # 每个隐藏层分别和x进行attention运算
        # 权重矩阵weights[batch, seq_len, num_layers]
        # 高纬矩阵乘法matmul,只匹配最后两个维度，前面的维度若不一致会作broadcast转换
        # bmm会检查是否都是三维
        h = y[-1].unsqueeze(0)  # 只取最后一层的隐藏层来作attention
        softmax = nn.Softmax(dim=1)
        weights = self.weight_func(x).bmm(h.permute(1, 2, 0))
        # weights = x.bmm(h.permute(1, 2, 0))
        weights = softmax(weights)
        # 对encoder的各个h带权求和即可得到attention结果
        # 这个地方注意返回size应该和y保持一致便于cat,因此最后一步进行转置
        attention = x.permute(0, 2, 1).bmm(weights).permute(2, 0, 1)
        # 将attention和decoder的h连接在一起，然后转化为新的h
        # 连接在一起之后第三个维度会加倍，用一个全连接层进行转换
        attention_context = self.combine_func(torch.cat([h, attention], dim=2))
        ret = y.clone()
        ret[-1] = attention_context  # 最后一层为attion
        return ret


# 主要由四部分组成：encoder、decoder、attention,translater
class EN2CN(nn.Module):
    def __init__(
        self,
        en_vsize,  # 英文字典大小
        cn_vsize,  # 中文字典大小
        BOS,  # 起始字符
        EOS,  # 结束字符
    ):
        super(EN2CN, self).__init__()
        self.hsize = 512
        self.rnn_layers = 3
        self.cn_vsize = cn_vsize
        self.encoder = Encoder(en_vsize, 512, self.hsize, self.rnn_layers)
        self.decoder = Decoder(cn_vsize, 512, self.hsize, self.rnn_layers)
        self.attention = Attention(self.hsize)
        self.BOS = BOS
        self.EOS = EOS

    def forward(
        self,
        x,  # en语句
        y=None,  # cn语句，当正式翻译的时候不要输入此参数
        sampling=0.5,  # sampling的概率,当正式翻译的时候不要输入此参数
    ):
        # 训练的时候由于使用sampling，所以会使用y即正确结果梯度
        # 正式翻译的时候不需要输入y
        return self.__inference__(x) if y == None else self.__train__(x, y, sampling)

    # 训练的时候调用此函数
    def __train__(
        self,
        x,
        y,
        sampling,  # sampling的概率
    ):
        # en语句x:[batch, seq_len1]
        # cn语句y:[batch, seq_len2]
        # 要使用sampling需要依赖y,有概率直接用预测的值而不是y中的值
        encoder_ouput, h = self.encoder(x)
        # encoder的输出h[num_layers * 2, batch, encoder_hidden_size]
        # decoder的输入h[num_layers * 1, batch, encoder_hidden_size*2]
        # 将encoder的双向h拼接在一起给decoder作为context使用
        h = h.view(self.rnn_layers, 2, h.shape[1], -1)  # 见nn.GRU的output说明
        h = torch.cat((h[:, -2, :, :], h[:, -1, :, :]), dim=2)
        input = y[:, 0]  # 起始符
        # 预测的one-hot分布
        ret = torch.zeros((y.shape[0], y.shape[1], self.cn_vsize), device=x.device)
        ret[:, 0, y[0, 0]] = 1.0  # 第一个都是是起始字符，放入
        for i in range(1, y.shape[1]):  # 预测出和y等长的语句
            out, h = self.decoder(input, h)
            h = self.attention(encoder_ouput, h)
            ret[:, i] = out
            pred_next = out.topk(1, axis=1)[1].squeeze()
            # 语言模型，即根据上一个字预测下一个字,输入了前n-1个词，预测后n-1个词
            # 有概率使用正确答案（target）来指导语句的生成，使用目标语中的词输入进行预测，可以加速训练，减少误差
            tearcher_forcing = torch.rand((1,)).item() < sampling
            input = y[:, i] if tearcher_forcing else pred_next

        return ret

    # 正式翻译的时候会调用此函数
    def __inference__(self, x):
        # en语句x:[batch, seq_len1]
        # 作beam search找出综合得分最高的句子,可以理解为每次只选取有限个点的树广度优先搜索
        # 得分：sum{log(p(yi|x,yi-1))}/len,相当于求最大似然估计，同时加上长度惩罚
        logsoftmax = nn.LogSoftmax(dim=0)
        WIDTH = 5  # beam 宽度
        MAX_NUM = 20  # 最多找到的句子数量
        MAX_LEN = 50  # 句子最大长度
        rets = []
        for b in x:  # 每个batch要分开处理，因为很可能不会同时出现结束符，导致查找结束
            # 初始输入
            encoder_ouput, h = self.encoder(b.unsqueeze(0))
            input = self.BOS  # 起始字符
            h = h.view(self.rnn_layers, 2, h.shape[1], -1)  # 见nn.GRU的output说明
            h = torch.cat((h[:, -2, :, :], h[:, -1, :, :]), dim=2)

            # 每次都从当前层生成的所有节点中选出topk进行下一次迭代计算
            # 注意先把输出结束符的句子取出来并保存
            ret = []  # 最终找到的所有句子(sen,score)
            nodes = [(input, h, [self.BOS], 0)]  # (要输入的字符，h，已经生成的字符，当前得分)
            got_all = False
            for i in range(MAX_LEN):
                nodes_all = []  # 所有生成的新节点
                for input, h, sen, score_sum in nodes:
                    input = torch.tensor([input], device=x.device)
                    out, h = self.decoder(input, h)
                    out = logsoftmax(out.squeeze())
                    predk = out.topk(WIDTH)
                    h = self.attention(encoder_ouput, h)
                    for j in range(WIDTH):
                        pred = predk.indices[j]
                        score_new = predk.values[j].item() + score_sum
                        sen_new = sen.copy()
                        sen_new.append(pred.item())
                        if predk.indices[j] == self.EOS:  # 已经输出了结束符
                            ret.append((sen_new, score_new / (i + 1)))
                            got_all = len(ret) >= MAX_NUM
                            if got_all:
                                break
                            continue
                        nodes_all.append((pred, h, sen_new, score_new))
                    if got_all:
                        break
                if got_all:
                    break
                # 选出得分最高的k个节点
                nodes = heapq.nlargest(min(WIDTH, len(nodes_all)), nodes_all, key=lambda x: x[3])
            if len(ret) == 0:  # 没有找到最后输出结束符的正常句子,则将现有得分最高的句子放入rets
                sen_best = nodes[0][2]
            else:
                # 选出得分最高的句子
                sen_best, _ = heapq.nlargest(1, ret, key=lambda x: x[1])[0]

            rets.append(sen_best)

        return rets


# test
if __name__ == "__main__":

    mm = EN2CN(100, 200)
    r = mm(torch.randint(0, 100, (3, 20)), torch.randint(0, 200, (3, 30)))
    nn.GRU
