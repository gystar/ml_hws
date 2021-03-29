import torch
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
import numpy as np
import random
from sklearn import cluster


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# KMeans
def direct_kpredict(latents):
    # kmeans Clustering
    # kmeans 是基于空间欧式距离来进行聚类的
    pred = cluster.MiniBatchKMeans(n_clusters=2, random_state=0, max_iter=200).fit(latents)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred


# PCA->TSNE->KMeans
def predict(latents):
    # 使用kernelpca进行第一次维度转换
    transformer = KernelPCA(n_components=200, kernel="rbf", n_jobs=-1)
    kpca = transformer.fit_transform(latents)

    # 使用TSNE再次降为成平面点
    X_embedded = TSNE(n_components=2).fit_transform(kpca)

    # kmeans Clustering
    # kmeans 是基于空间欧式距离来进行聚类的
    pred = cluster.MiniBatchKMeans(n_clusters=2, random_state=0, max_iter=200).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred