import os
import importlib
from pickle import TRUE
import torch
import image_set
import encoder
import model_manager
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans

importlib.reload(image_set)
importlib.reload(encoder)
importlib.reload(model_manager)

device = torch.device("cuda" if True & torch.cuda.is_available() else "cpu")

encoder = encoder.AutoEncoder()
MODEL_PATH = "./data/model.pkl"
if os.path.exists(MODEL_PATH):
    encoder = model_manager.load_model(encoder, MODEL_PATH, device)

if False:
    # train the model
    data_train = image_set.NoLabeledSet(True)
    encoder = model_manager.train_model(
        encoder,
        data_train,
        MODEL_PATH,
        device,
        epochs=200,
        nbatch=32,
        lr=0.0001,
        opt=0,
    )
    # 看一下参数的梯度情况
    for name, param in encoder.named_parameters():
        print(name)
        print(None if param.grad == None else param.grad.abs().mean().item())

# 画出原始和最终得到的图片，看一下encoder的效果
data1 = image_set.NoLabeledSet(False)
img = data1.images[0]
plt.figure()
plt.imshow(image_set.tensor2numpy(img))
img1 = encoder(img.unsqueeze(0).to(device)).cpu().squeeze(0)
plt.figure()
plt.imshow(image_set.tensor2numpy(img1))

# 对验证集进行编码
data_val = image_set.ValidationSet()
codes = model_manager.encode(encoder, device, data_val).numpy()


def predict(latents):
    # First Dimension Reduction
    transformer = KernelPCA(n_components=200, kernel="rbf", n_jobs=-1)
    kpca = transformer.fit_transform(latents)
    print("First Reduction Shape:", kpca.shape)

    # # Second Dimesnion Reduction
    X_embedded = TSNE(n_components=2).fit_transform(kpca)
    print("Second Reduction Shape:", X_embedded.shape)

    # Clustering
    pred = MiniBatchKMeans(n_clusters=2, random_state=0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded


pred1, _ = predict(codes)
rigth_num1 = len((pred1 == data_val.labels.squeeze().numpy()).nonzero()[0])
print(f"the accuracy of kmeas1:{100*rigth_num1/len(codes)}%")

# 使用内置的kmeans算法进行聚类
kmeans = cluster.KMeans(n_clusters=2, max_iter=100, tol=1e-4, n_init=10).fit(codes)
rigth_num = len((kmeans.labels_ == data_val.labels.numpy()).nonzero()[0])
print(f"the accuracy of kmeas2:{100*rigth_num/len(codes)}%")
