import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os
files=os.listdir("../camera_cal/")
#%matplotlib qt

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images

def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return warped


# Step through the list and search for chessboard corners
for fname in files:
    img = cv2.imread('../camera_cal/'+fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

#        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        offset_x=100
        offset_y=100
        nx=9

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)

        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        dst = np.float32([[offset_x, offset_y], [img.shape[1]-offset_x, offset_y],
                                     [img.shape[1]-offset_x, img.shape[0]-offset_y],
                                     [offset_x, img.shape[0]-offset_y]])
        img=warper(img,src,dst)
        print(fname)
        cv2.imwrite('../camera_cal_output/'+fname,undist)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()