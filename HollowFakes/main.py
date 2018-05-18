import torch
import cv2
from YFYF.Alignment.BaseAligner import BaseAligner as Aligner
from YFYF.Alignment.BaseAlignerOpenCV import BaseAlignerOpenCV as CVAligner
from YFYF.Rendering.BaseFaceRenderer import BaseFaceRenderer as Renderer
from YFYF.Rendering.PickableFaceRenderer import PickableFaceRenderer as PRenderer
from YFYF.Morphing.BaseModelFitter import BaseFitter as Fitter
from skimage.io import imread, imsave, imshow, show
import matplotlib.pyplot as plt
import numpy as np
from time import time, sleep
source = imread('data/trump.jpg')
target = imread('data/test_images/Boris-Johnson.jpg')

# print(f'ori shape {target.shape}, aspect {aspect}, new {height} {width}')
aligner = Aligner()
imgs = [source,target]
dets = aligner.detect(imgs, 0)
cropped = aligner.crop(imgs, dets, square=True)
print(f'Cropped : {cropped[1].shape}')

aspect = cropped[1].shape[0] / cropped[1].shape[1]
height = int(1000)
width = int(height/aspect)
fitter = Fitter()
scale = cropped[1].shape[0] // fitter.image_size
with torch.no_grad():
    source_semetic = fitter.fit(cropped[0])
    target_semetic = fitter.fit(cropped[1])
    index_to_swap = ['shape', 'tex']
    for index_type in index_to_swap:
        indices = fitter.decoder.get_indices(index_type)
        target_semetic[:,indices] = source_semetic[:,indices]
    mesh = fitter.make_mesh(target_semetic)
    # scale = (target_semetic[:,fitter.decoder.get_indices('scale')].cpu().numpy().squeeze() + 1) / 1000.0
    # mesh.vertices *= scale
    print(f'mesh.vertices : {mesh.vertices.shape}, {mesh.vertices[:,0].mean()}, {mesh.vertices[:,1].mean()}, {mesh.vertices[:,0].max()} {mesh.vertices[:,1].max()}  {mesh.vertices[:,0].min()} {mesh.vertices[:,1].min()}')
    # rot_mat = np.eye(4)
    # rot_mat[:3,:3] = fitter.decoder.get_rotation_matrix(target_semetic[:,0:3]).cpu().numpy().squeeze()
    # trans = target_semetic[:,fitter.decoder.get_indices('trans')].cpu().numpy().squeeze()
    # scale_x, scale_y = scale * width, scale * height
    renderer = PRenderer(mesh, screen_size=[width, height], background=cropped[1], cleanup=False)
    renderer.background_run()
    sleep(2)
    renderer.manual = True
    renderer.set_model(np.eye(4))
    renderer.set_translation(0, 0)
    renderer.set_scale(16, 16) #why is this 16??? 160/10?
    renderer.manual = False
    # print(f'Rot Mat : \n{rot_mat} \n trans {trans} \n scale {scale}, {scale}')
