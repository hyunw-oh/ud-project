import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os
files=os.listdir("../camera_cal/")

def accumlateCalibrate():
    # %matplotlib qt

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    for fname in files:
        img = cv2.imread('../camera_cal/'+fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    return objpoints, imgpoints

objpoints, imgpoints = accumlateCalibrate()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (1280,720), None, None)

#cameraCalibrate(img) will be used externally
def cameraCalibrate(img):
    img = cv2.undistort(img, mtx, dist, None, mtx)
    return img