'''
Description: 
Date: 2022-08-13 12:34:04
LastEditTime: 2022-10-11 16:45:36
FilePath: /01_test_tvm_onnx/01_onnx_torch_infer.py
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

# onnx infer
ort_session = onnxruntime.InferenceSession(onnx_model_path)
ort_inputs = {ort_session.get_inputs()[0].name: rand_input.numpy()}
ort_outs = ort_session.run(None, ort_inputs)
print("onnx output: ", ort_outs)

# test time
T1 = time.time()
for i in range(1000):
    ort_inputs = {ort_session.get_inputs()[0].name: rand_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

T2 = time.time()
print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))


print("finish!")


# onnx output:  [array([[-7.0989323,  6.9344306]], dtype=float32)]
# 程序运行时间:1576.2107372283936毫秒



