import torch
import torch.nn.functional as F
import numpy as np


def downsample(tensor, size):
    x = torch.linspace(-1, 1, size).view(-1, 1).repeat(1, size)
    y = torch.linspace(-1, 1, size).repeat(size, 1)
    grid = torch.cat((y.unsqueeze(2), x.unsqueeze(2)), 2)
    grid.unsqueeze_(0)

    return F.grid_sample(tensor, grid)


def rgb2gray(tensor):
    R = tensor[:, 0,...]
    G = tensor[:, 1,...]
    B = tensor[:, 2,...]
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    return gray.unsqueeze(1)