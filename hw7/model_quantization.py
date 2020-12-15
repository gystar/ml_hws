"""
Author: gystar
Date: 2020-12-14 11:02:45
LastEditors: gystar
LastEditTime: 2020-12-14 11:02:46
FilePath: /ml_hws/hw7/model_quantization..py
Description: 
    此处使用简单的quantization方法，将模型中的32位的浮点数字截尾存为16位的，读取模型的时候再恢复为32位
"""
import torch
import numpy as np
import pickle


def save_model_f16(model: torch.nn.Module(), path):
    compressed_dict = {}
    for (name, value) in model.state_dict().items():
        value = np.float64(value.cpu().numpy())
        # 只对矩阵进行压缩
        compressed_dict[name] = np.float16(value) if type(value) == np.ndarray else value
    pickle.dump(compressed_dict, open(path, "wb"))


def load_model_f16(model: torch.nn.Module(), path):
    compressed_dict = pickle.load(open(path, "rb"))
    origin_dict = {}
    for (name, value) in compressed_dict.items():
        value = torch.tensor(value)
        origin_dict[name] = value

    model.load_state_dict(origin_dict)


##test
if __name__ == "__main__":
    import torchvision.models as models

    target_model = models.resnet18(pretrained=False, num_classes=11)

    path = "./data/compressed_model1.bin"
    save_model_f16(target_model, path)
    load_model_f16(target_model, path)