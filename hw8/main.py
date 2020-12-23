import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import datetime
import multiprocessing
import math
import os
import model_manager
import sentense_set
import en2cn_model

importlib.reload(model_manager)
importlib.reload(sentense_set)
importlib.reload(en2cn_model)

MODEL_PATH = "./data/model.pkl"
device = torch.device("cuda" if True & torch.cuda.is_available() else "cpu")

dic = sentense_set.Dictionary()
data = sentense_set.SentenseSet("./data/training.txt", dic)
model = en2cn_model.EN2CN(len(dic.en_ix2word), len(dic.cn_ix2word))


if os.path.exists(MODEL_PATH):
    model = model_manager.load_model(model, MODEL_PATH, device)

for _ in range(20):
    model_manager.train_model(model, data, device, epochs=5, lr=0.1, opt=1, weight_decay=0.0001)
    model_manager.save_model(model, MODEL_PATH)