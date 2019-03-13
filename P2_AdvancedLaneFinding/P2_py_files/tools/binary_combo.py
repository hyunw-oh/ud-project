import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os

#enchance yellow line
def enhance_yellow(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_yellow = np.array([5, 100, 100])
    high_yellow = np.array([40,255,255])
    yellow_range=cv2.inRange(hsv, low_yellow, high_yellow)
    yellow_converted_img=img.copy()
    yellow_converted_img[yellow_range == 255] = 255
#    cv2.imshow("test",yellow_converted_img)
    return yellow_converted_img

#enchance white line
def enhance_white(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([131, 255, 255])
    white_range=cv2.inRange(hsv, lower_white, upper_white)
    white_converted_img=img.copy()
    white_converted_img[white_range == 255] = 255
#    cv2.imshow("test",yellow_converted_img)
    return white_converted_img

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # sobel 이미지 추출
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # range 0~1
    abs_sobel = np.absolute(abs_sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Pixels have a value of 1 or 0 based on the strength of the x gradient.
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    binary_output = np.copy(sbinary)
    return binary_output


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # sobel image
    abs_sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0,sobel_kernel))
    abs_sobel_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,sobel_kernel))

    # 3) Calculate the magnitude
    grad_mag= np.sqrt(abs_sobel_x**2+abs_sobel_y **2)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(grad_mag)/255
    grad_mag = (grad_mag/scale_factor).astype(np.uint8)

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(grad_mag)
    binary_output[(grad_mag >= mag_thresh[0]) & (grad_mag <= mag_thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def binary_combo(image ):
    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    yellow_enhanced_img=enhance_yellow(image)
    mixed_enhanced_img=enhance_white(yellow_enhanced_img)
    grady = abs_sobel_thresh(mixed_enhanced_img, orient='y', sobel_kernel=ksize, thresh=(50, 100))
    gradx = abs_sobel_thresh(mixed_enhanced_img, orient='x', sobel_kernel=ksize, thresh=(50, 100))
    mag_binary = mag_thresh(mixed_enhanced_img, sobel_kernel=ksize, mag_thresh=(5, 100))
    dir_binary = dir_threshold(mixed_enhanced_img, sobel_kernel=ksize, thresh=(0.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined
