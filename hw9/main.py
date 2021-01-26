import os
import importlib
import torch
import image_set
import encoder
import model_manager
from sklearn import cluster
import numpy as np

importlib.reload(image_set)
importlib.reload(encoder)
importlib.reload(model_manager)

device = torch.device("cuda" if False & torch.cuda.is_available() else "cpu")

encoder = encoder.AutoEncoder()
MODEL_PATH = "./data/model.pkl"
if os.path.exists(MODEL_PATH):
    model = model_manager.load_model(encoder, MODEL_PATH, device)

if False:
    data_train = image_set.NoLabeledSet(True)
    encoder = model_manager.train_model(
        encoder,
        data_train,
        MODEL_PATH,
        device,
        epochs=20,
        nbatch=32,
        lr=0.00001,
        opt=0,
    )
    # 看一下参数的梯度情况
    for name, param in encoder.named_parameters():
        print(name)
    print(None if param.grad == None else param.grad.abs().mean().item())

# 对验证集进行编码
data_val = image_set.ValidationSet()
codes = model_manager.encode(encoder, device, data_val).numpy()
# 使用内置的kmeans算法进行聚类
kmeans = cluster.KMeans(n_clusters=2, max_iter=100, tol=1e-4, n_init=10).fit(codes)
rigth_num = len((kmeans.labels_ == data_val.labels.numpy()).nonzero()[0])
print(f"the accuracy of kmeas:{100*rigth_num/len(codes)}%")
