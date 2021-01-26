import os
import importlib
import torch
import image_set
import encoder
import model_manager

importlib.reload(image_set)
importlib.reload(encoder)
importlib.reload(model_manager)

device = torch.device("cuda" if False & torch.cuda.is_available() else "cpu")

encoder = encoder.AutoEncoder()
MODEL_PATH = "./data/model.pkl"
if os.path.exists(MODEL_PATH):
    model = model_manager.load_model(encoder, MODEL_PATH, device)

data = image_set.NoLabeledSet(True)

encoder = model_manager.train_model(
    encoder,
    data,
    MODEL_PATH,
    device,
    epochs=20,
    nbatch=32,
    lr=0.001,
    opt=0,
)

for name, param in encoder.named_parameters():
    print(name)
    print(None if param.grad == None else param.grad.abs().mean().item())
