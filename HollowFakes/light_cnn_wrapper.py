from __future__ import print_function
import argparse
import os
import shutil
import logging as log
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

from .LightCNN.light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from .LightCNN.load_imglist import ImageList
from .LightCNN.preprocessing import normalize_image
from skimage.color import rgb2gray
from skimage.transform import resize
from glob import glob
import matplotlib.pyplot as plt
import logging as log

import HollowFakes.utils as utils

class LightCNN_Wrapped():
    def __init__(self):
        model = LightCNN_29Layers_v2(num_classes=80013)
        model.eval()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load('data/LightCNN_29Layers_V2_checkpoint.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        self.model = model

    def __call__(self, _input):
        _input = _input.view(-1, 1, 128, 128).cuda()
        _, feature = self.model(_input)
        return feature.squeeze() 

    def preprocess(self, tensor):
        if isinstance(tensor, np.ndarray):
            if image.shape[0] != 128 or image.shape[1] != 128:
                image = resize(image, [128, 128])
            img = rgb2gray(image)
            img = torch.Tensor(img).unsqueeze(0).unsqueeze(0).cuda()
            # print(f'preprocess : {img.shape}')
        elif isinstance(tensor, torch.Tensor):
            if len(tensor.shape) != 4:
                log.error(f'Image shape Unrecognized for LightCNN preprocessing : {image.shape}, has to be BxCxHxW')
                return 
            if tensor.shape[2] != 128 or tensor.shape[3] != 128:
                tensor = utils.downsample(tensor, 128)
            if tensor.shape[1] == 3:
                tensor = utils.rgb2gray(tensor)
            img = tensor.cuda()
        return img
