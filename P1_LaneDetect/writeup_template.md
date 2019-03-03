# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # "Image References"
[image1]: ./test_images_output/solidWhiteCurve.jpg 
[image2]: ./test_images_output/solidWhiteRight.jpg 
[image3]: ./test_images_output/solidYellowCurve.jpg
[image4]: ./test_images_output/solidYellowLeft.jpg 
[image5]: ./test_images_output/solidYellowCurve2.jpg

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
solidWhiteCurve.jpg
![alt text][image2]
solidWhiteRight.jpg
![alt text][image3]
solidYellowCurve.jpg
![alt text][image4]
solidYellowLeft.jpg
![alt text][image5]
solidYellowCurve2.jpg
<video controls="controls">
  <source type="video/mp4" src="./test_videos_output/challenge.mp4"></source>
</video>
challenge.mp4
<video controls="controls">
  <source type="video/mp4" src="./test_videos_output/solidWhiteRight.mp4"></source>
</video>
solidWhiteRight.mp4
<video controls="controls">
  <source type="video/mp4" src="./test_videos_output/solidYellowLeft.mp4"></source>
</video>
solidYellowLeft.mp4

#### @Depending on the viewer, these videos may not be visible. If so, videos can be found in the directory './test_videos_output/*'.

### 2. Identify potential shortcomings with your current pipeline

The left line is not well recognized. in challeg.mp4 
I have to remember ten frames and draw an average line to ignore the overflowing value.

### 3. Suggest possible improvements to your pipeline

My source contains a lot of hyper parameters.
If possible, I want the program to judge the appropriate parameter values and set the values itself.