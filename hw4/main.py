from gensim.models import Word2Vec
import multiprocessing
import importlib
import numpy as np
import pandas as pd
import sentiment
import utils
from os import path
importlib.reload(sentiment)
importlib.reload(utils)

current_dir = path.dirname(path.abspath(__file__))
TRAIN_LABEL_PATH = path.join(current_dir,"data/training_label.txt")
TRAIN_NO_LABEL_PATH = path.join(current_dir,"data/training_nolabel.txt")
TESTING_PATH = path.join(current_dir,"data/testing_data.txt")
DIM_WORD = 1024  # 词向量的维度，若改变，需要手动重新生成词向量库

'''
# 先分词，将三个文档中的所有句子拆为单词
# 然后通过word2vec建立词向量词典
'''
f = open(TRAIN_LABEL_PATH, encoding='utf-8')
lines = f.readlines()
labels_train = np.array([int(line.split(" ")[0]) for line in lines])
sentenses_labeled = np.array(
    [line.strip("\n").split(" ")[2:] for line in lines])

f = open(TRAIN_NO_LABEL_PATH, encoding='utf-8')
lines = f.readlines()
sentenses_nolabeled = np.array(
    [line.strip("\n").split(" ") for line in lines])

f = open(TESTING_PATH, encoding='utf-8')
# 去掉第一行->去掉每一行的标号->去掉\n
lines = f.readlines()[1:]
sentenses_test = np.array([line.split(",", 1)[1].strip("\n").split(" ")
                           for line in lines])

# word embedding，利用所有的语料库，使用word2vec建立词向量，每个key的词向量存储在model.wv[key]中
SAVED_PATH = path.join(current_dir,"wordvec.pkl")
try :  # 若已经保存过，则不需要再训练词向量，因为生成较慢
    wordvecs = Word2Vec.load(SAVED_PATH)
    print("load wordvecs from file.")
except:  # 没有保存过，则生成词向量并存储
    print("generating word vectors...")
    all_sentenses = np.concatenate((sentenses_labeled, sentenses_nolabeled, sentenses_test), axis=0)
    wordvecs = Word2Vec(all_sentenses, size=DIM_WORD, window=5, min_count=1,
                        workers=multiprocessing.cpu_count())
    wordvecs.save(SAVED_PATH)


# 用有标签的训练集进行一次训练
print("begin to train model with labeled sentenses...")
model = sentiment.train_model(
    sentenses_labeled, labels_train, wordvecs.wv, DIM_WORD, epochs = 50)

print("computing the acurracy of training set...")
y_pred = sentiment.predict(model, sentenses_labeled, wordvecs.wv).squeeze()
labels_pred = utils.p2label(y_pred)
print("the accuracy of labeled training set is ", 100 *
      utils.calc_accuracy(labels_train, labels_pred), "%")


# 对没有标签的训练集进行一次预测，当预测值大于阈值thresh的时候，将其加入到再训练集合
thresh = 0.8
print("generating labels from unlabeled train data with threshold ", thresh, " ...")
y_pred_unlabeled = sentiment.predict(
    model, sentenses_nolabeled, wordvecs.wv).squeeze()
# 只取结果概率在区间[0, 1-thresh]和[thresh,1]中的预测为可信的预测
idx = ((y_pred_unlabeled > thresh) | (y_pred_unlabeled < (1-thresh))).numpy()
lables_generated = np.array(utils.p2label(y_pred_unlabeled[idx]))
sentenses_generated = sentenses_nolabeled[idx]
# 用这些通过半监督得到的label加入到原来的有labeled数据集中，继续训练模型
print(len(lables_generated), " labels have been generated.")
print("begin to train model with generated labels...")
model = sentiment.train_model(
    np.concatenate((sentenses_labeled, sentenses_generated),axis=0), 
    np.concatenate((labels_train,lables_generated), axis=0),
    wordvecs.wv, DIM_WORD, epochs=10,
    model = model)#继续之前的模型训练

print("computing the acurracy of training set...")
y_pred = sentiment.predict(model, sentenses_labeled, wordvecs.wv).squeeze()
labels_pred = utils.p2label(y_pred)
print("the accuracy of labeled training set with modified model is ", 100 *
      utils.calc_accuracy(labels_train, labels_pred), "%")

# 测试写入结果文档
print("waiting for testing ...")
y_pred_test = sentiment.predict(
    model, sentenses_test, wordvecs.wv).squeeze().numpy()
label_pred_test = utils.p2label(y_pred_test)
pd.DataFrame({"id": [x for x in range(len(label_pred_test))],
              "label":  label_pred_test}).to_csv(path.join(current_dir,"data/result.csv"), index=False)
print("test result has been written into ./data/result.csv")
