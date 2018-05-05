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
source = imread('trump.jpg')
target = imread('stephen.jpg')

aspect = target.shape[0] / target.shape[1]
height = 600
width = int(height/aspect)
# print(f'ori shape {target.shape}, aspect {aspect}, new {height} {width}')
aligner = Aligner()
imgs = [source,target]
dets = aligner.detect(imgs, 0)
cropped = aligner.crop(imgs, dets, square=True)

fitter = Fitter()
with torch.no_grad():
    source_semetic = fitter.fit(cropped[0])
    target_semetic = fitter.fit(cropped[1])
    index_to_swap = ['shape', 'tex']
    for index_type in index_to_swap:
        indices = fitter.decoder.get_indices(index_type)
        target_semetic[:,indices] = source_semetic[:,indices]
    mesh = fitter.make_mesh(target_semetic)

    rot_mat = np.eye(4)
    rot_mat[:3,:3] = fitter.decoder.get_rotation_matrix(target_semetic[:,0:3]).cpu().numpy()
    trans_mat = fitter.decoder.get_translation_matrix(target_semetic).cpu().numpy()
    scale_mat = fitter.decoder.get_scale_matrix(target_semetic).cpu().numpy()

    model = trans_mat @ rot_mat @ scale_mat
    # print(f'Rot Mat : {rot_mat}')
    print(f'Model Matrix : {model}')
    renderer = PRenderer(mesh, screen_size=[width, height], background=target, cleanup=True)
    # renderer.manual = True
    # renderer.set_model(model)
    # renderer.manual = False
    renderer.background_run()
    print('mvp ' ,renderer.get_MVP())
