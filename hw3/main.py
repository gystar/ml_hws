import torch
import numpy as np
import pandas as pd
import os,sys
import multiprocessing
import importlib
import image_set  # 图片数据集实现类
import model_manager
import image_classification  # 分类模型实现类


#处理思路：数据处理（引用函数dataset）->模型实例化、损失函数、优化器->
# dataloader构造->循环训练模型（寻找参数）

# 如果自定义的模块代码改变，需要reload
importlib.reload(image_set)
importlib.reload(model_manager)
importlib.reload(image_classification)

current_dir = os.path.dirname(__file__)
TRAIN_DIR = os.path.join(current_dir ,"data/training")
VALIDATION_DIR = os.path.join(current_dir ,"data/validation")
TEST_DIR = os.path.join(current_dir ,"data/testing")
SAVE_PATH = os.path.join(current_dir ,"model.pkl")
TEST_REULST_PATH = os.path.join(current_dir ,"./data/result.csv")

# 用以对cpu和GPU进行兼容
cuda_ok = torch.cuda.is_available()
print("Cuda is available" if cuda_ok else "There is no gpu available.")

def CalcRightCount(y, t):
    # 计算预测正确的数量，y、t均为numpy一维数组
    count = 0
    for i in range(y.shape[0]):
        if y[i] == t[i]:
            count += 1
    return count

def validate(data_loader, m):
    m.eval()  # 会关闭dropout、batchnorm等  
    right_count = 0    
    with torch.no_grad():  # 不构建计算图
        for _, info in enumerate(data_loader):
            images, labels = info
            if cuda_ok:
                images, labels = images.cuda(), labels.cuda()
            y_pred = m(images).squeeze()            
            right_count += CalcRightCount(
                np.argmax(y_pred.cpu().numpy(), 1), labels.cpu().numpy())
            del images, labels, y_pred
            torch.cuda.empty_cache()            
    return right_count

# 训练数据集
data_train = image_set.LearningSet(
    TRAIN_DIR, image_classification.GYHF_AlexNet.input_size)
class_count = data_train.GetClassNum()  # 获取类别数量 
    
#模型实例化
if os.path.exists(SAVE_PATH):
    print("model has been loaded from file.")
    model = torch.load(SAVE_PATH)
else:
    print("create a new model.")
    model = image_classification.GYHF_AlexNet(class_count)
    
if cuda_ok:
    try:  # 如果显存不够，可能无法用GPU进行计算
        model = model.cuda()
    except:
        print("There is no enough GPU memory for model,use cpu instead.")
        model=model.cpu()
        cuda_ok = False

nbatch_predict = 1024
#验证集数据加载器
#训练数据验证
data_validation1 = image_set.LearningSet(
                TRAIN_DIR, model.input_size, False)
data_loader_validation1 = torch.utils.data.DataLoader(
                data_validation1, nbatch_predict, shuffle=False,
                num_workers=multiprocessing.cpu_count())
#验证集数据验证
data_validation2 = image_set.LearningSet(
                VALIDATION_DIR, model.input_size, False)
data_loader_validation2 = torch.utils.data.DataLoader(
                data_validation2, nbatch_predict, shuffle=False,
                num_workers=multiprocessing.cpu_count())
validate(data_loader_validation1,model)
#训练模型
print("waiting for training...")
for i in range(1):
    model = model_manager.train_model(model, data_train,cuda_ok = cuda_ok, epochs = 10)
    with torch.no_grad():
        #每10轮保存一次模型，同时验证一下正确率
        # 模型保存
        torch.save(model, SAVE_PATH)
        # 用验证集和训练集验证:
        print("waiting for validation...")
        train_accuracy = 100*validate(data_loader_validation1,model)/data_validation1.GetLen()
        print("train accuracy:", train_accuracy, "%")            
        validation_accuracy = 100 * \
        validate(data_loader_validation2,model)/data_validation2.GetLen()
        print("validation accuracy:", validation_accuracy, "%")

# 测试集
model.eval()
data_test = image_set.TestingSet(TEST_DIR, model.input_size)
data_loader_test = torch.utils.data.DataLoader(
    data_test, nbatch_predict, shuffle=False, num_workers=multiprocessing.cpu_count())
y_test = []
print("waiting for testing...")
with torch.no_grad():
    for i, images in enumerate(data_loader_test):
        if cuda_ok:
            images = images.cuda()
        y_pred = model(images).cpu().squeeze()
        # 获得类别，即最大元素下标
        y_test.extend(list(np.argmax(y_pred.numpy(), 1)))
        del images, y_pred
        torch.cuda.empty_cache()
# 测试结果存入文件
pd.DataFrame({"Id": [x for x in range(data_test.GetLen())],
              "Category":  y_test}).to_csv(TEST_REULST_PATH, index=False)
print("test result has been written into ./data/result.csv")
