import os
import importlib
from pickle import TRUE
import torch
import image_set
import encoder
import model_manager
import numpy as np
import matplotlib.pyplot as plt
import utils

importlib.reload(image_set)
importlib.reload(encoder)
importlib.reload(model_manager)
importlib.reload(utils)

device = torch.device("cuda" if True & torch.cuda.is_available() else "cpu")
utils.same_seeds(10)  # 如果使各个部分的随即seed相同，否则测试的时候会受到随即数的影响

encoder_class = encoder.ConvEnoder
encoder = encoder_class().to(device)
MODEL_PATH = f"./data/{str(encoder_class)}.pkl"
if os.path.exists(MODEL_PATH):
    encoder = model_manager.load_model(encoder, MODEL_PATH, device)

if True:
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
print("image0 before encoder:")
plt.figure()
plt.imshow(image_set.tensor2numpy(img))
plt.show()
img1 = encoder(img.unsqueeze(0).to(device)).cpu().squeeze(0)
print("image0 after encoder:")
plt.figure()
plt.imshow(image_set.tensor2numpy(img1))
plt.show()


data_val = image_set.ValidationSet()
# 不使用encoder直接进行聚类
pred1 = utils.predict(data_val.images.cpu().view(data_val.images.shape[0], -1))
rigth_num1 = len((pred1 == data_val.labels.squeeze().numpy()).nonzero()[0])
print(f"the accuracy of clustering without autoencoder:{100*rigth_num1/data_val.__len__()}%")


# 使用encoder后进行聚类
codes = model_manager.encode(encoder, device, data_val).numpy()
pred2 = utils.predict(codes)
rigth_num2 = len((pred2 == data_val.labels.squeeze().numpy()).nonzero()[0])
print(f"tthe accuracy of clustering with autoencoder:{100*rigth_num2/len(codes)}%")
