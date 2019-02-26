#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline

def grayscale(img):
    """Applies the Grayscale transform"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    x1, y1, x2, y2 = lines
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    #img=initial_img * α + img * β + γ
    return cv2.addWeighted(initial_img, α, img, β, γ)


def cv2_fitline(img, f_lines):
    width,height=img.shape[0],img.shape[1]
    lines = f_lines.reshape(f_lines.shape[0] * 2, 2)
    output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((width - 1) - y) / vy * vx + x), width - 1
    x2, y2 = int(((width / 2 + 100) - y) / vy * vx + x), int(width/ 2 + 100)

    result = [x1, y1, x2, y2]
    return result

def process_image(image):
    imshape=image.shape

    gray_image = grayscale(image)

    blur_image = gaussian_blur(gray_image ,5)

    canny_image = canny(blur_image ,100,230)

    vertices = np.array([[(imshape[0]*0.05,imshape[0]),(imshape[1]/2, imshape[0]/2), (imshape[1]/2, imshape[0]/2), (imshape[1]*0.95,imshape[0])]], dtype=np.int32)
    roi_image = region_of_interest(canny_image ,vertices)

    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 50     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments

    lines = hough_lines(roi_image,rho,theta,threshold,min_line_length,max_line_gap)
    lines = np.squeeze(lines)

    # calculate degree
    line_degree = (np.arctan2(lines[:,1] - lines[:,3], lines[:,0] - lines[:,2]) * 180) / np.pi
    line_degree = np.squeeze(line_degree)

    # limit vertical degree
    lines = lines[np.abs(line_degree)>90]
    line_degree = line_degree[np.abs(line_degree)>90]

    # limit horizontal degree
    lines = lines[np.abs(line_degree) < 170]
    line_degree = line_degree[np.abs(line_degree) < 170]

    # Devide the line into the two types
    left_lines = lines[(line_degree>0),:]
    right_lines  = lines[(line_degree<0),:]
    line_img = np.zeros((imshape[0], imshape[1], 3), dtype=np.uint8)

    # find the fitline
    left_fit_line = cv2_fitline(image,left_lines)
    right_fit_line = cv2_fitline(image,right_lines)

    # draw lines
    draw_lines(line_img, left_fit_line,[255,0,0],12)
    draw_lines(line_img, right_fit_line,[255,0,0],12)

    result = weighted_img(line_img, image)
    return result

import os
test=os.listdir("test_images/")
for file_name in test:
    image = mpimg.imread("test_images/"+file_name)
    result=process_image(image)
    fig = plt.gcf()
    plt.imshow(result)
    plt.show()
# then save them to the test_images_output directory.
    fig.savefig("test_images_output/"+file_name)

test=os.listdir("test_videos/")
for file_name in test:
    cap = cv2.VideoCapture("test_videos/"+file_name)
    while (cap.isOpened()):
        if (cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            cap.open(file_name)
        ret, frame = cap.read()
        result = process_image(frame)
        cv2.imshow("Video",result)
        if cv2.waitKey(33) > 0: break

#white_output = 'test_videos_output/solidWhiteRight.mp4'
#yellow_output = 'test_videos_output/solidYellowLeft.mp4'
#challenge_output = 'test_videos_output/challenge.mp4'
