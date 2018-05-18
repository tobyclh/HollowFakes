import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
img = imread('/home/toby/Documents/HollowFakes/data/HF/cropped_boris/2. 200px-boris_johnson_-opening_bell_at_nasdaq-14sept2009-3c_cropped.png')
img = img[...,:-1] #ditch alpha channel
# plt.imshow(img)
# plt.show()
img_tensor = torch.Tensor(img).unsqueeze(0)
img_tensor = img_tensor.permute(0, 3, 1, 2)

plt.imshow(banana)
plt.show()

out_size = 128
x = torch.linspace(-1, 1, out_size).view(-1, 1).repeat(1, out_size)
y = torch.linspace(-1, 1, out_size).repeat(out_size, 1)
grid = torch.cat((y.unsqueeze(2), x.unsqueeze(2)), 2)
grid.unsqueeze_(0)

image_small = F.grid_sample(img_tensor, grid)
resized = image_small[0].permute(1, 2, 0).numpy()
resized = resized.astype(np.uint8)
plt.imshow(resized)
plt.show()