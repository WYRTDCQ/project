import numpy as np
import cv2
import torch.nn as nn
import os
import torch
from torch.nn import functional as F
from PIL import Image
import torchvision.transforms as tr

#python实现卷积操作
#train_img_path=r'..\dataset\Images\spatial_envelope_256x256_static_8outdoorcategories'
train_img_path=r'..\dataset\grayimages'
label_path=r'..\dataset\label'
k = np.array([[-1, -1, -1],
          [-1, 9, -1],
          [-1, -1, -1]])

def convolution(k, data):
    n,m = data.shape
    img_new = []
    for i in range(n-3):
        line = []
        for j in range(m-3):
            a = data[i:i+3,j:j+3]
            line.append(np.sum(np.multiply(k, a)))
        img_new.append(line)
    return np.array(img_new)
names=sorted(os.listdir(train_img_path))
for i in range(len(names)):
    img=cv2.imread(os.path.join(train_img_path, names[i]),0)#0,单通道
    data = np.array(img)
    img_ = convolution(k, data)
    name1= names[i].split(".jpg")
    print(name1[0])
    cv2.imwrite(os.path.join(label_path, name1[0])+".jpg", img_)
    print(i)


