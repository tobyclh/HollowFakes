from torch.utils.data import DataLoader
from YFYF.Alignment.BaseAligner import BaseAligner
from YFYF.Alignment.Aligner2D import Aligner2D
from tqdm import tqdm
from skimage.io import imread, imsave, imshow, show
from skimage.transform import rotate, resize, rescale
from skimage.draw import polygon
import numpy as np
images = ['data/johnson.jpg', 'data/stephen.jpg', 'data/trump.jpg']
output_shape = [128*2, 128*2]
ec_mc_y = 48*2 #check the lightcnn repo&paper for more information
ec_y = 48*2 #check the lightcnn repo&paper for more information
from multiprocessing import Pool
import logging as log
def rotate_pt(origin, points, angle):
    """
    Rotate a point counterclockwise by a given angle(degree) around a given origin.
    """
    oy, ox = origin
    qx = ox + np.cos(angle) * (points[:, 0] - ox) - np.sin(angle) * (points[:, 1] - oy)
    qy = oy + np.sin(angle) * (points[:, 0] - ox) + np.cos(angle) * (points[:, 1] - oy)
    return np.asarray([qx, qy]).transpose()

eye_indices = [36, 45]
mouth_indices = [48, 54]

import face_alignment
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=False)


def normalize_image(img, mask=False):
    img = rescale(img, 512/max(img.shape), preserve_range=True, mode='constant').astype(np.uint8) #for some reason high res image doesn't work in cnn detection (upsampling?)
    lms = fa.get_landmarks(img)[0]
    # _img = aligner.draw_landmark(img.copy(), lms)
    # imshow(_img)
    # show()
    if lms is None:
        return
    left_eye = lms[eye_indices[0]]
    right_eye = lms[eye_indices[1]]
    left_mouth = lms[mouth_indices[0]]
    right_mouth = lms[mouth_indices[1]]

    angle = np.arctan((right_eye[1] - left_eye[1])/(right_eye[0] - left_eye[0]))
    # print(f'Angle : {angle}')
    img = rotate(img, np.degrees(angle), mode='constant')
    # imshow(img)
    # show()

    rotated_lms = rotate_pt(np.asarray(img.shape[:2])//2, lms, -angle)
    # print(f'rotated_lms {rotated_lms[:5]} \n lm {lms[:5]}')
    # _img = aligner.draw_landmark(img.copy(), rotated_lms)
    # imshow(_img)
    # show()

    scale = ec_mc_y/np.linalg.norm((right_eye + left_eye)/2 - (left_mouth + right_mouth)/2, ord=2)
    img = rescale(img, scale, mode='constant')
    rotated_lms *= scale
    left_eye = rotated_lms[eye_indices[0]]
    right_eye = rotated_lms[eye_indices[1]]
    mid_eye = (left_eye + right_eye)/2
    start_y = int(mid_eye[1]) - ec_y
    end_y = start_y + output_shape[0]
    start_x = int(mid_eye[0]) - output_shape[1]//2
    end_x = int(mid_eye[0]) + output_shape[1]//2
    if start_y < 0 or end_y > img.shape[0] or start_x < 0 or end_x > img.shape[1]:
        # log.warn(f'Face too near edge, skipped : {image_pth}')
        return
    # print(f'{start_y} {end_y} {start_x} {end_x}')
    cropped = img[start_y:end_y, start_x:end_x]
    if not mask:
        return cropped
    height, width = int(end_y - start_y), int(end_x - start_x)
    mask = np.zeros(cropped.shape[:-1], np.uint8)
    xs = np.asarray(rotated_lms[:,0])
    ys = np.asarray(rotated_lms[:,1])
    xs = xs[:27]
    ys = ys[:27]
    xs[17:] = xs[17:][::-1]
    ys[17:] = ys[17:][::-1]
    xs -= start_x
    ys -= start_y
    # print(f'xs : {xs}, ys : {ys}')
    xs = np.maximum(np.minimum(xs, width), 0)
    ys = np.maximum(np.minimum(ys, height), 0)
    rr, cc = polygon(ys, xs)
    mask[rr, cc] = 1
    return cropped, mask


    # imshow(cropped)
    # show()


# with Pool(16) as pool:
#     pool.map(normalize_image, images)
