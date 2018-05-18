import argparse
parser = argparse.ArgumentParser(description='Turn image with face to learning ready images')
parser.add_argument('--data_dir',  type=str, required=True,
                    help='pth where all the data is stored')
parser.add_argument('--output_dir', type=str, required=True,
                    help='where to output the files')
parser.add_argument('--worker', type=int, default=8,
                    help='Pool Size')
args = parser.parse_args()
import os
from os.path import *
from skimage.io import imread, imsave
from skimage.color import grey2rgb
from glob import glob
import logging as log
from multiprocessing.pool import Pool
from LightCNN.preprocessing import normalize_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
imgs = glob(join(args.data_dir, '*.*'), recursive=True)
log.info(f'Found {len(imgs)} imgs in {args.data_dir}')

if not exists(args.output_dir):
    os.mkdir(args.output_dir)
    assert exists(args.output_dir), f'Cannot create output directory : {args.output_dir}'  


log.info(f'Starting {args.worker} worker')

def normalize_and_save(img_path):
    try:
        img = imread(img_path)
    except:
        log.error(f'Failed to open image : {img_path}')
        return
    if len(img.shape) < 2 or len(img.shape) > 4:
        return
    if img.shape[0] < 100 or img.shape[1] < 100:
        return
    if img.shape[-1] > 3:
        img = img[..., :-1] # remove alpha channel
    img = grey2rgb(img)
    result = normalize_image(img, mask=True)
    if result is None:
        return
    img, mask = result
    mask = mask[...,np.newaxis]
    # print(f'img, mask : {img.shape}, {mask.shape}')
    img = np.concatenate([img, mask], axis=-1)
    output_filename = join(args.output_dir, basename(img_path))
    filename, file_extension = os.path.splitext(output_filename)
    img = (img*255).astype(np.uint8)
    imsave(filename + '.png', img)
    # f, axarr = plt.subplots(2)
    # axarr[0].imshow(img)
    # axarr[1].imshow(mask)
    # plt.show()
    # except:
    #     log.error(f'Error processing : {img_path}')

with Pool(args.worker) as p:
    r = list(tqdm(p.imap(normalize_and_save, imgs), total=len(imgs)))
