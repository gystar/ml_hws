import torch
import importlib
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
model = en2cn_model.EN2CN(len(dic.en_ix2word), len(dic.cn_ix2word), data.BOS, data.EOS)


if os.path.exists(MODEL_PATH):
    model = model_manager.load_model(model, MODEL_PATH, device)

model_manager.train_model(
    model,
    data,
    save_path=MODEL_PATH,
    sampling=0.5,
    device=device,
    epochs=30,
    lr=0.0001,
    opt=0,
    nbatch=32,
    clip_norm=100.0,
)


# test
en = "i love you ."
ch = model_manager.translate_one(model, device, data, en)
print('en sen: "%s", translation: "%s"' % (en, ch))

# 打印各个层的梯度情况(计算了梯度且没有清0)
for name, param in model.named_parameters():
    print(name)
    print(None if param.grad == None else param.grad.abs().mean().item())
