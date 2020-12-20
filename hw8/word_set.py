# 翻译的文字内容读取类实现
import torch
import json
import os

# 中英文的one-hot编码和词的双向映射字典
class dictionary(object):
    def __init__(self):
        super(dictionary, self).__init__()
        self.en_ix2word = json.load(open("./data/int2word_en.json"))
        self.cn_ix2word = json.load(open("./data/int2word_cn.json"))
        self.en_word2ix = json.load(open("./data/word2int_en.json"))
        self.cn_word2ix = json.load(open("./data/word2int_cn.json"))


# 翻译数据源的读取
class SentenseSet(torch.utils.data.Dataset):
    def __init__(self, path, word_dic: dictionary):
        self.dic = word_dic
        BOS = word_dic.en_word2ix["<BOS>"]  # 起始字符
        EOS = word_dic.en_word2ix["<EOS>"]  # 结束字符
        UNK = word_dic.en_word2ix["<UNK>"]  # 未知字符
        PAD = word_dic.en_word2ix["<PAD>"]  # 填充字符
        self.sentenses_en = []  # 所有英文句子
        self.sentenses_cn = []  # 所有中文句子
        # 再遍历每个句子进行处理
        for line in open(path).readlines():
            # 去掉换行符，根据‘\t’分割为英文和中文部分
            line = line.strip("\n").split("\t")
            # 分词并将词语转换为one-hot编码
            en_codes = [
                word_dic.en_word2ix.get(c, UNK) for c in line[0].split(" ")[:-1]
            ]  # 注意句子最后多了一个空格,因此会被多split出来一个空字符
            cn_codes = [
                word_dic.cn_word2ix.get(c, UNK) for c in line[1].split(" ")[:-1]
            ]
            # 在开头和结尾加上标识符
            en_codes.insert(0, BOS)
            en_codes.append(EOS)
            cn_codes.insert(0, BOS)
            cn_codes.append(EOS)
            self.sentenses_en.append(torch.LongTensor(en_codes))
            self.sentenses_cn.append(torch.LongTensor(cn_codes))
        # 分别将英文和中文中每一个句子pad成等长，将batch放在第一个维度
        self.sentenses_en = torch.nn.utils.rnn.pad_sequence(
            self.sentenses_en, batch_first=True, padding_value=PAD
        )
        self.sentenses_cn = torch.nn.utils.rnn.pad_sequence(
            self.sentenses_cn, batch_first=True, padding_value=PAD
        )
        # pad之后句子的长度
        self.en_len = self.sentenses_en.shape[1]
        self.cn_len = self.sentenses_cn.shape[1]

    def __len__(self):
        return len(self.sentenses)

    def _getitem__(self, index):
        return self.sentenses_en[index], self.sentenses_cn[index]


##test
if __name__ == "__main__":
    dic = dictionary()
    path = "./data/training.txt"
    data = SentenseSet(path, dic)
    print(data._getitem__(0))
    a1 = torch.tensor([1, 2, 3, 4])
    a2 = torch.tensor([6])
    a3 = torch.tensor([4, 5])
    lengths = [len(a1), len(a2), len(a3)]
    a4 = torch.nn.utils.rnn.pad_sequence([a1, a2, a3])
