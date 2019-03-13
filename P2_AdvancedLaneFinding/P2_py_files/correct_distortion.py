import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'tools')
import undistort as und

import os
files=os.listdir("../test_images/")
#%matplotlib qt

for file in files:
    img = cv2.imread('../test_images/' + file)
    img_unditorted=und.cameraCalibrate(img)

cv2.destroyAllWindows()