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



class LightCNN_Wrapper():
    def __init__(self):
        model = LightCNN_29Layers_v2(num_classes=80013)
        model.eval()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load('data/LightCNN_29Layers_V2_checkpoint.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        self.model = model

    def forward(self, _input):
        _input = _input.view(-1, 1, 128, 128).cuda()
        return self.model(_input).squeeze()
