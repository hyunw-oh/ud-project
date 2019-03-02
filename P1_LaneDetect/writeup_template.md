# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteCurve.jpg "Grayscale"
[image2]: ./test_images_output/solidWhiteRight.jpg "Grayscale"
[image3]: ./test_images_output/solidYellowCurve.jpg "Grayscale"
[image4]: ./test_images_output/solidYellowLeft.jpg  "Grayscale"
[image5]: ./test_images_output/solidYellowCurve2.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 
First, I converted the images to grayscale
Second, I blurred the screen to remove noise. That will help to find a clear line.
Third, I used Canny Detection to find a strong edge on the screen.
Fourth, I limit the degree of the line 
Fifth, I find the line using cv2.fit_line

*My project test all cases iteratively.

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]


### 2. Identify potential shortcomings with your current pipeline

Other functions have been implemented, 
but saving output_movie function has not been implemented yet. 
That part will be revised.

### 3. Suggest possible improvements to your pipeline

There appears to be an unexpected input during video input. 
I will fix this bug.