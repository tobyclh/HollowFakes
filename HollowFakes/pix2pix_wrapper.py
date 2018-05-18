import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from pix2pix.models import create_model
from pix2pix.models.cycle_gan_model import CycleGANModel
from pix2pix.options.test_options import TestOptions
from pix2pix.options.train_options import TrainOptions
from pix2pix.util import html
from pix2pix.util.util import tensor2im
from pix2pix.util.visualizer import Visualizer

class pix2pix_wrapped:
    def __init__(self):
        opt = TrainOptions().parse()
        opt.input_nc = 3
        opt.loadSize = 256
        opt.which_model_netG = 'unet_256'
        opt.which_direction = 'AtoB'
        opt.batchSize = 2
        opt.norm = 'batch'
        opt.no_lsgan = True
        opt.model = 'pix2pix'
        opt.dataset_mode = 'aligned'
        self.opt = opt
        self.model = create_model(opt)
    
    def __call__(self, image, injection=None):
        if not image.is_cuda:
            image = image.cuda()

        if injection is not None and not injection.is_cuda:
            injection = injection.cuda()

        result = self.model.netG(image, injection)
        return result