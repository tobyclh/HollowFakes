import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from HCIGAN.pix2pix.data import CreateDataLoader
from HCIGAN.pix2pix.models import create_model
from HCIGAN.pix2pix.models.cycle_gan_model import CycleGANModel
from HCIGAN.pix2pix.options.test_options import TestOptions
from HCIGAN.pix2pix.util import html
from HCIGAN.pix2pix.util.util import tensor2im
from HCIGAN.pix2pix.util.visualizer import Visualizer

from torchsummary import summary


class pix2pix_wrapped:
    def __init__(self):
        opt = TestOptions().parse()
        opt.dataroot = 'datasets/edges2shoes/'
        opt.checkpoints_dir = './data/'
        opt.name = 'edges2shoes_2nd'
        opt.no_dropout = False
        opt.model = 'pix2pix'
        opt.dataset_mode = 'aligned'
        opt.loadSize = 256
        opt.nThreads = 1   # test code only supports nThreads = 1
        opt.batchSize = 1  # test code only supports batchSize = 1
        opt.serial_batches = False  # no shuffle
        opt.no_flip = True  # no flip
        opt.display_id = -1  # no visdom display
        opt.which_model_netG = 'unet_256'
        opt.which_direction = 'AtoB'
        opt.norm = 'batch'

        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        
        model = create_model(opt)
        # model.load_networks('30')
        model.netG.module.load_state_dict(torch.load('/home/toby/Documents/HCIGAN/data/30_net_G.pth'))
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.test()
            if i == 100:
                break

        model.netG.train(False)
        self.model = model
        self.data_iter = iter(dataset)
        self.get_next()
        

    def get_next(self):
        data = next(self.data_iter)
        self.model.set_input(data)
        self.model.netG(self.model.real_A,cache=True)
        self.original_intermediate = self.intermediate.clone()

    
    @property
    def intermediate(self):
        return self.model.netG.module.model._submodules[-1].intermediate
    
    @intermediate.setter
    def intermediate(self, value):
        self.model.netG.module.model._submodules[-1].intermediate = value

    def __call__(self, vector, img=False):
        if not vector.is_cuda:
            vector = vector.cuda()
        vector = vector.view(-1, 512, 1, 1)
        self.intermediate = self.original_intermediate.expand(vector.shape[0], *self.original_intermediate.shape[1:]) + vector*5
        result = self.model.netG(None, full=False, cache=False)
        if not img:
            return result
        imgs = []
        for i in range(result.shape[0]):
            imgs.append(tensor2im(result[i,...].unsqueeze(0)))
        return np.stack(imgs)


    def plt_data(self, visuals):
        real_A, fake_B = map(tensor2im, [visuals['real_A'], visuals['fake_B']])
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(real_A)
        axarr[1].imshow(fake_B)
        plt.show()
