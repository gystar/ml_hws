import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np
import os,sys
import importlib
import torch
import math

sys.path.append('../hw3') ##直接使用hw3中的model
import image_classification
import image_set
importlib.reload(image_set)
importlib.reload(image_classification)

current_dir = os.path.dirname(__file__)
MODEL_PATH =  os.path.join(current_dir, "../hw3/<class 'image_classification.GYHF_LetNet5'>.pkl")
TRAIN_DIR = os.path.join(current_dir, "../hw3/data/training")
OUTPUT_DIR = os.path.join(current_dir, "output")
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
    
def plot_filters(filters):
    #输入的layers为[layer_count, channels,w, h]
    fig = plt.figure(figsize=(filters.shape[2], filters.shape[3]))
    cols = 3
    rows = math.ceil(filters.shape[0]/cols)
    
    for i in range(filters.shape[0]):
        rgb_matrix = np.array(transforms.ToPILImage()(filters[i].detach()))
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        # grab layer outputs
        ax.imshow(rgb_matrix)
        ax.set_title('Filter %s' % str(i+1), fontsize=15,color='g')  
    
#指定模型类别
model_class = image_classification.GYHF_LetNet5
data = image_set.LearningSet( TRAIN_DIR, model_class.input_size, False)
#使用已经训练好的model
model = torch.load(MODEL_PATH)

plot_filters(model.features[0].weight)