# 翻译的文字内容读取类实现
import torch
import json
import os

# 中英文的one-hot编码和词的双向映射字典
class Dictionary(object):
    def __init__(self):
        super(Dictionary, self).__init__()
        self.en_ix2word = json.load(open("./data/int2word_en.json"))
        self.cn_ix2word = json.load(open("./data/int2word_cn.json"))
        self.en_word2ix = json.load(open("./data/word2int_en.json"))
        self.cn_word2ix = json.load(open("./data/word2int_cn.json"))


# 翻译数据源的读取
class SentenseSet(torch.utils.data.Dataset):
    def __init__(self, path, word_dic: Dictionary):
        # 将句子转换为one-hot编码并pad成等长，分别组成en句子和cn句子的矩阵
        self.dic = word_dic
        self.BOS = word_dic.en_word2ix["<BOS>"]  # 起始字符
        self.EOS = word_dic.en_word2ix["<EOS>"]  # 结束字符
        self.UNK = word_dic.en_word2ix["<UNK>"]  # 未知字符
        self.PAD = word_dic.en_word2ix["<PAD>"]  # 填充字符
        self.sentenses_en = []  # 所有英文句子
        self.sentenses_cn = []  # 所有中文句子
        # 再遍历每个句子进行处理
        for line in open(path).readlines():
            # 去掉换行符，根据‘\t’分割为英文和中文部分
            line = line.strip("\n").split("\t")
            # 分词并将词语转换为one-hot编码
            en_codes = [word_dic.en_word2ix.get(c, self.UNK) for c in line[0].split(" ")[:-1]]  # 注意句子最后多了一个空格,因此会被多split出来一个空字符
            cn_codes = [word_dic.cn_word2ix.get(c, self.UNK) for c in line[1].split(" ")[:-1]]
            # 在开头和结尾加上标识符
            en_codes.insert(0, self.BOS)
            en_codes.append(self.EOS)
            cn_codes.insert(0, self.BOS)
            cn_codes.append(self.EOS)
            self.sentenses_en.append(torch.LongTensor(en_codes))
            self.sentenses_cn.append(torch.LongTensor(cn_codes))
        # 分别将英文和中文中每一个句子pad成等长，将batch放在第一个维度
        self.sentenses_en = torch.nn.utils.rnn.pad_sequence(self.sentenses_en, batch_first=True, padding_value=self.PAD)
        self.sentenses_cn = torch.nn.utils.rnn.pad_sequence(self.sentenses_cn, batch_first=True, padding_value=self.PAD)
        # pad之后句子的长度
        self.en_len = self.sentenses_en.shape[1]
        self.cn_len = self.sentenses_cn.shape[1]

    def __len__(self):
        return len(self.sentenses_en)

    def __getitem__(self, index):
        return self.sentenses_en[index], self.sentenses_cn[index]

    def EN2Numbers(self, en_sen: str):
        ret = [self.BOS]
        for c in en_sen.split(" "):
            ret.append(self.dic.en_word2ix.get(c, self.UNK))
        ret.append(self.EOS)
        return torch.tensor(ret)

    def Numbers2CN(self, numbers: list):
        ret = []
        for i in numbers:
            if i == self.BOS:
                continue
            if i == self.EOS:
                break
            ret.append(self.dic.cn_ix2word.get(str(i), "$"))
        return "".join(ret)


##test
if __name__ == "__main__":
    dic = Dictionary()
    path = "./data/training.txt"
    data = SentenseSet(path, dic)
    print(data.__getitem__(0))
    a1 = torch.tensor([1, 2, 3, 4])
    a2 = torch.tensor([6])
    a3 = torch.tensor([4, 5])
    lengths = [len(a1), len(a2), len(a3)]
    a4 = torch.nn.utils.rnn.pad_sequence([a1, a2, a3])
