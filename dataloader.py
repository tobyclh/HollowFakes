import torch
from torch.utils.data import Dataset
from skimage import io, transform, img_as_float, color
import sys
import random
import numpy as np
import cv2
from glob import glob
from os.path import join
from light_cnn_wrapper import LightCNN_Wrapped as LightCNN
class FaceDataset(Dataset):
    def __init__(self, data_dir):
        """ a dataset that normalize face according to LightCNN standard
        
        Arguments:
            data_dir {[str]} -- [the output directoy created with processing_dataset script]
        """
        self.data_dir = data_dir
        self.images = glob(join(data_dir, '*.*'))   
        self.identifier = LightCNN()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = io.imread(self.data_dir + "/" + image_path)
        image, mask = image[...,:-1], image[...,-1]//255
        identity = self.identifier(image).view(256, 1, 1)
        image = img_as_float(image)
        if len(image.shape) < 3:
            image = color.grey2rgb(image)
        if np.random.uniform() > 0.5:
            image = np.fliplr(image)
        image = (image * 2) - 1
        image = np.transpose(image, (1, 2, 0))
        image = torch.Tensor(image)
        mask = torch.Tensor(mask)
        return image, mask, identity



