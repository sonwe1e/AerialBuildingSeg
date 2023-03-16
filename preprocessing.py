import os
import cv2
import numpy as np

label_path = './data/labels/'
window_path = './data/window_labels/'
if not os.path.exists(window_path):
    os.mkdir(window_path)
label_list = os.listdir(label_path)
for label in label_list:
    if label[-1] == 'g':
        img = cv2.imread(label_path+label)
        img = cv2.resize(img, (512, 512))
        index = np.ones((512, 512))
        index[np.where(img[..., 0] != 128)] = 0
        index[np.where(img[..., 1] != 0)] = 0
        index[np.where(img[..., 2] != 0)] = 0
        np.save(window_path+label[:-4], index.astype(np.uint8))