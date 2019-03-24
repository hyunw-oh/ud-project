import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'tools')
import undistort
import warper
import find_lr_line
import binary_combo
import time
import os
files=os.listdir("../test_images/")
#%matplotlib qt

for file in files:
    img = cv2.imread('../test_images/' + file)
    img_undistorted = undistort.cameraCalibrate(img)
    img_size = (img.shape[1], img.shape[0])

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

    binary_warped=binary_combo.binary_combo(img_undistorted)

    warped = warper.warp(binary_warped, src, dst)
    window_img, left_curverad, right_curverad, deviation = find_lr_line.search_around_poly(warped)

    window_img=warper.warp(window_img,dst,src)
    window_img=window_img.astype(np.uint8)
    img_undistorted=img_undistorted.astype(np.uint8)
    result = cv2.addWeighted(img_undistorted , 1.0,window_img, 0.3, 0.0)
    cv2.imshow('image',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # View your output
    #plt.imshow(result[...,::-1]) #same - result[:,:,::-1]
    #plt.show()

def print_carinfo(img, left_curvature, right_curvature,deviation):
    left_text = 'left curvature : {0:0.2f}m'.format(left_curvature)
    right_text = 'right curvature : {0:0.2f}m'.format(right_curvature)
    cv2.putText(img, left_text, (30, 70), cv2.QT_FONT_NORMAL, 1.5, (50, 100, 100), 1)
    cv2.putText(img, right_text, (30, 120), cv2.QT_FONT_NORMAL, 1.5, (50, 100, 100), 1)
    cv2.putText(img, deviation, (30, 170), cv2.QT_FONT_NORMAL, 1.5, (50, 100, 100), 1)

files = os.listdir("../project_video/")
print(files)
for file_name in files:
    cap = cv2.VideoCapture("../project_video/"+file_name)
    ret, frame = cap.read()

    fcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter("../project_video_output/"+file_name, fcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            img_undistorted = undistort.cameraCalibrate(frame)
            img_size = (frame.shape[1], frame.shape[0])

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

            binary_warped = binary_combo.binary_combo(img_undistorted)
            warped = warper.warp(binary_warped, src, dst)
            window_img, left_curverad, right_curverad, deviation = find_lr_line.search_around_poly(warped )
            window_img = warper.warp(window_img, dst, src)
            window_img = window_img.astype(np.uint8)
            img_undistorted = img_undistorted.astype(np.uint8)
            result = cv2.addWeighted(img_undistorted, 1.0, window_img, 0.3, 0.0)

            print_carinfo(result,left_curverad,right_curverad,deviation)

            cv2.imshow(file_name,result)
            out.write(result)

        if cv2.waitKey(33) > 0: break

#    out.release()
    cap.release()
    cv2.destroyAllWindows()
