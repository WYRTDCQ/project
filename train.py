import numpy as np
import cv2
import torch.nn as nn
import os
from PIL import Image
import torchvision.transforms as tr
train_img_path=r'..\dataset\Images\spatial_envelope_256x256_static_8outdoorcategories'
train_path=r'..\dataset\grayimages'
names=sorted(os.listdir(train_img_path))
for i in range(len(names)):
    img =cv2.imread(os.path.join(train_img_path, names[i]))
    img_= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    cv2.imwrite(os.path.join(train_path, names[i])+".jpg", img_)
    print(i)
'''
class Trainer(object):
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0,
                               bias=False)
'''