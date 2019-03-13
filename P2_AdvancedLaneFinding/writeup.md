**Advanced Lane Finding Project**

[//]: # (Image References)

[image1]: ./camera_cal_output/calibration2.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./test_images_output/test2.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image7]: ./writeup/result.PNG "Output"
[video1]: ./project_video_output/project_video.mp4 "Video"

My all python file was in P2_py_files.
finding_lines.py is my main project file.
Tools directory has many utility to solve a problem.
My project runs all the test cases sequentially and stores the output in the * _output directory.

------------------------------------------------------
####The last output file is in the project_video_output

------------------------------------------------------

## Camera Calibration
#### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
###### correct_distortion.py
The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)
###### finding_lines.py

#### 1. Provide an example of a distortion-corrected image.
###### tools/undistort.py
#######The all input image was undistorted and saved on the camera_cal_output directory.
undistort.py calculate the distortion information, and has the information  on memory.
I calculated the calibration values ​​in advance and used to undistort the images and videos.
The process was fast because I already used the calculated results.

    import undistort
    img_undistorted = undistort.cameraCalibrate(img)


![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
###### tools/binary_combo.py

I used the two filter and four binary converter.
Two filters are enhancing yellow and white line information.
Yellow line and white line information means lane on the road.

Four converters are converting color images to binary images.
xsobel, ysobel, mag, dir binary filter was mixed to assist find line


![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
###### warp_image.py  and tools/warper.py
#######The all input image was warped and saved on the test_images_output directory.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:
I tried the perspective warp on the binarized videos and images

```python
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

    warped = warper.warp(binary_warped, src, dst)
```
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
###### finding_ilnes.py

To find poly
I call this function

            window_img = find_lr_line.search_around_poly(warped )

in the source code...
First of all.
I set the ROI area for the first time, I specified a sectorial ROI because the lane can turn left and right.

    binary_warped=roi.get_roi(binary_warped,vertices)

find lane pixels : Extract left and right line pixel positions

    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    
fit polynomial : To draw exact internal space between left and right line

    left_fit, right_fit, ploty = fit_polynomial(binary_warped, leftx, lefty, rightx, righty,out_img)

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
Fit a second order polynomial to each using `np.polyfit`
Calculation of R_curve (radius of curvature)
I am using the Formula

f(y)=Ay 2*y^2 +B*y+C

R curve = (1+(2*A*y+B)^2 )^(3/2)/∣2A∣
 
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    y_eval = np.max(ploty)

    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My project did not solve the harder difficult challenge. I could not solve the shadows and the motorbike that was on the line.

I think the shadow can be solved by using the histogram equalization, and the intervening motorcycle seems to be able to be solved by giving the average of line recognition and ignoring the bouncing value.
