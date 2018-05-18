import random
import sys
from glob import glob
from os.path import join

import cv2
import numpy as np
import torch
from skimage import color, img_as_float, io, transform
from skimage.color import rgb2gray
from torch.utils.data import Dataset

from light_cnn_wrapper import LightCNN_Wrapped as LightCNN


class NormalizedFaceDataset(Dataset):
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
        image = io.imread(image_path)
        image, mask = image[...,:-1], image[...,-1]//255
        mask = 1 - mask
        identity = self.identifier(self.identifier.preprocess(image)).view(256, 1, 1)
        image = img_as_float(image)
        if len(image.shape) < 3:
            image = color.grey2rgb(image)
        if np.random.uniform() > 0.5:
            image = np.fliplr(image)
        image = (image * 2) - 1
        masked = image * mask[...,np.newaxis]
        image = np.transpose(image, (2, 0, 1))
        masked = np.transpose(masked, (2, 0, 1))

        image = torch.Tensor(image)
        masked = torch.Tensor(masked)
        # print(f'Image {image.shape} masked {masked.shape}')
        output = {}
        output['A'] = masked
        output['B'] = image
        output['injection'] = identity
        output['paths'] = image_path
        return output
