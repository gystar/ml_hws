"""
Author: gystar
Date: 2020-12-14 10:59:54
LastEditors: gystar
LastEditTime: 2020-12-14 10:59:54
FilePath: /ml_hws/hw7/model_proning.py
Description: 
    参考https://arxiv.org/pdf/1708.06519.pdf中的理论
    对normbatch层中r接近0的连接进行proning，同时减少其前一层的对应neuron（行），
    删除后一层对应的weigth(列)
"""
##test
if __name__ == "__main__":
    import torch
    import model_architecture
    import os

    # 使用distillation训练得到的模型存储路径
    SMART_MODEL_DISTILLATION_SAVE_PATH = "./data/model_distillation.bin"

    smart_model = model_architecture.SmartResnet18()
    if os.path.exists(SMART_MODEL_DISTILLATION_SAVE_PATH):
        smart_model.load_state_dict(torch.load(SMART_MODEL_DISTILLATION_SAVE_PATH))
    print("\n".join(["%s:%s" % item for item in smart_model.__dict__.items()]))

    #### block 中的第一个bn
    # bn proning
    bn = smart_model.resnet18.layer1[0].bn1
    print("\n".join(["%s:%s" % item for item in bn.__dict__.items()]))
    thresh = 1e-6
    idx = bn._parameters["weight"].data > thresh
    num_left = torch.count_nonzero(idx).item()
    bn.num_features = num_left
    bn._parameters["weight"].data = bn._parameters["weight"].data[idx]
    bn._parameters["bias"].data = bn._parameters["bias"].data[idx]
    bn._buffers["running_mean"].data = bn._buffers["running_mean"].data[idx]
    bn._buffers["running_var"].data = bn._buffers["running_var"].data[idx]

    # pre cnn proning
    cnn_pointwise = bn = smart_model.resnet18.layer1[0].conv1[1]
    print("\n".join(["%s:%s" % item for item in cnn_pointwise.__dict__.items()]))
    cnn_pointwise.out_channels = num_left
    # cnn的weight在第一维度:[out,in,w,h]
    cnn_pointwise._parameters["weight"].data = cnn_pointwise._parameters["weight"].data[idx]
