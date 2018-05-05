from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import cv2

from light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from load_imglist import ImageList
from preprocessing import normalize_image
from skimage.color import rgb2gray
from glob import glob
import matplotlib.pyplot as plt

person_A = glob('/home/toby/Documents/HollowFakes/data/donald/*.*')
person_B =  glob('/home/toby/Documents/HollowFakes/data/boris/*.*')
test =  glob('/home/toby/Documents/HollowFakes/data/test_images/*.*')
# print(f'Personal A :{person_A}')
model = LightCNN_29Layers_v2(num_classes=80013)
model.eval()
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load('data/LightCNN_29Layers_V2_checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

with torch.no_grad():
    with torch.no_grad():
        donald_feature = []
        boris_feature = []
        for image in person_A:
            img = normalize_image(image)
            if img is None:
                continue
            img = rgb2gray(img)
            img = torch.Tensor(img).unsqueeze(0).unsqueeze(0).cuda()
            _, feature = model(img)
            donald_feature.append(feature.cpu().squeeze().numpy())
        
        donald_feature = np.stack(donald_feature)
        print(f'donald_feature : shape {donald_feature.shape}, std {donald_feature.std(0).mean(0)}')

        for image in person_B:
            img = normalize_image(image)
            if img is None:
                continue
            img = rgb2gray(img)
            img = torch.Tensor(img).unsqueeze(0).unsqueeze(0).cuda()
            _, feature = model(img)
            boris_feature.append(feature.cpu().squeeze().numpy())
        boris_feature = np.stack(boris_feature)
        print(f'boris_feature : shape {boris_feature.shape}, std {boris_feature.std(0).mean(0)}')

        total_feature = np.concatenate([boris_feature, donald_feature])
        print(f'total_feature : shape {total_feature.shape}, std {total_feature.std(0).mean(0)}')
        average = np.abs((donald_feature.mean(0) - boris_feature.mean(0))).mean()
        print(f'Average diff : {average}')
        donald = donald_feature.mean(0)
        boris = boris_feature.mean(0)
        for test_image in test:
            img = normalize_image(test_image)
            if img is None:
                continue
            _img = rgb2gray(img)
            img = torch.Tensor(_img).unsqueeze(0).unsqueeze(0).cuda()
            _, feature = model(img)
            feature = feature.cpu().squeeze().numpy()
            print(f'Feature {feature[:10]}')
            DonaldDistance = np.abs(feature - donald).mean() 
            BorisDistance = np.abs(feature - boris).mean()
            decision = 'is Trump' if DonaldDistance < BorisDistance else 'is Boris'
            print(f'This image\'s {os.path.basename(test_image)} {decision}, TrumpBoris Ratio : {DonaldDistance}:{BorisDistance}')
            plt.imshow(_img)
            plt.show()

            