import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'tools')
import warper

import os
files=os.listdir("../test_images/")
#%matplotlib qt

for file in files:
    image = cv2.imread('../test_images/' + file)
    img_size = (image.shape[1], image.shape[0])

    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])

    warped = warper.warp(image, src,dst)
    cv2.imwrite('../test_images_output/' + file,warped)
    cv2.imshow("warped",warped)