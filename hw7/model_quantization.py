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
    compressed_dict = {}#compressed_dict是字典类型{}
    for (name, value) in model.state_dict().items():
        value = np.float64(value.cpu().numpy())
        # 只对矩阵进行压缩    字典类型compressed_dict[name]，如果没有索引到name，就自动保存，如果已有name则更新 
        compressed_dict[name] = np.float16(value) if type(value) == np.ndarray else value#type判断（）类型
    pickle.dump(compressed_dict, open(path, "wb"))#序列化


def load_model_f16(model: torch.nn.Module(), path):
    compressed_dict = pickle.load(open(path, "rb"))#"rb"读取权限
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