import face_alignment
from skimage import io
from YFYF.Alignment.BaseAligner import BaseAligner
import matplotlib.pyplot as plt
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=False)

input = io.imread('/home/toby/Documents/HollowFakes/data/HF/cropped_boris/51. reutersdominick-reuter.png')
input = input[...,:-1]
preds = fa.get_landmarks(input)
print(f'Preds : {preds}')

aligner = BaseAligner()
lms = aligner.get_landmark([input])[0]
alignered = aligner.draw_landmark(input.copy(), lms)
faed = aligner.draw_landmark(input.copy(), preds[0])
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(alignered)
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(faed)
plt.show()