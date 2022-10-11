'''
Description: 
Date: 2022-08-13 12:34:04
LastEditTime: 2022-10-11 20:19:48
FilePath: /01_test_tvm_onnx/02_tvm_optim_infer.py
'''
import torch
import torch.nn as nn
import cv2
import torchvision
import numpy as np
import os
import shutil
from tqdm import tqdm
import onnx
import onnxruntime
import time
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor


onnx_model_path = "/home/rane/2TDisk/01_Projects/06_hesuan/07_yolov5_det_cls_seg/runs/train-cls/1011_cls/weights/best.onnx"
img_pred_path = "/home/rane/2TDisk/01_Projects/06_hesuan/07_yolov5_det_cls_seg/datasets/01_labeled_data/01_all_0918_1004/croped_zui_images/1/59_正样本_train_24.jpg"


# test image
# device = torch.device('cuda')
device = torch.device('cpu')
mean = np.array([127.0, 127.0, 127.0])
stdev = np.array([128.0, 128.0, 128.0])
normalize = torchvision.transforms.Normalize(mean, stdev)
def preprocess(img):
    global device, normalize
    x = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x 

img = cv2.imread(img_pred_path, cv2.IMREAD_COLOR)
rand_input = preprocess(img)

# tvm convert
onnx_model = onnx.load(onnx_model_path)

# target = "llvm"  # llvm
# target = "llvm -mcpu=skylake"  # 有些提升
# target = "llvm -mcpu=core-avx2"
target = "llvm -mcpu=skylake-avx512"  # 提升深大！

input_name = "images"  # 和onnx第一层名一样
# input_name = "data"

shape_dict = {input_name: rand_input.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# tvm compiler
with tvm.transform.PassContext(opt_level=4):  # opt_level=4 优化更好
    lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))


# tvm infer
dtype = "float32"
module.set_input(input_name, rand_input.numpy())
module.run()
output_shape = (1, 2)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
print("tvm output: ", tvm_output)


# test time
T1 = time.time()
for i in range(1000):
    module.set_input(input_name, rand_input.numpy())
    module.run()
    output_shape = (1, 2)
    tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

T2 = time.time()
print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))


print("finish!")



