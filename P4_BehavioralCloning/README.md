# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_network.jpeg "Model Visualization"
[image2]: ./examples/center_image.png "Grayscaling"
[image3]: ./examples/from_center.png "Recovery Image"
[image4]: ./examples/right_bias.png "Recovery Image"
[image5]: ./examples/recover_center.png "Recovery Image"
[image6]: ./examples/left_car.PNG "Normal Image"
[image7]: ./examples/left_flip_car.PNG "Flipped Image"
[image8]: ./examples/normal_car.PNG "Normal Image"
[image9]: ./examples/darken_car.PNG "Darken Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 Successful driving video

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on nvidia model.
some parameters were slightly adjusted.

#### 2. Attempts to reduce overfitting in the model

I used about 15 percent of the data as verification data.

    train_samples, validation_samples = train_test_split(images_db, test_size=0.15,shuffle=True)

In the simulator, there were many cases of turning left.
So I created a flip image so that the model would not be overfit towards the left.
This also helped to increase data.

#### 3. Model parameter tuning

I used MSE for the loss function and I used adam for the optimizer.
Loss Function
MSE is suitable as a loss fuction in regression problems.

Optimizer
There are optimizer, Momentum, and Nag, which improved performance by modifying the gradient.
There are Adagrad, RMSProp and AdaDelta optimizers that have improved by modifying Learning Rate.
I finally decided to apply Adam, which combines these two strengths.
Good performance in most cases.

#### 4. Appropriate training data

The model I made was constantly falling into the river around the river.

At first it worked well in most cases. However, it seems that the model did not work well only around the river, because there are few images around the river in the scenario.
I tried to drive more precisely when driving around the river. And when I tried to get out of the lane in the vicinity of the river, I made the data repeatedly by returning to work again.
This action was also learned so that when the car moved around the river, it actually came back out of the lane.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As input, I received the image of the vehicle, the steering wheel information, the brake information, and the speed information.

This simulator does not have a traffic light or other vehicles, so that it does not need much brake information or speed information if it can only run.

 I decided to do lane detection on the image and use the corresponding steering angle as a label.

 Data was obtained while driving. But I did augmentation the data because the amount of data was not enough.
 We randomly created shadows, randomly adjusted the brightness, shifted the image, and controlled the position of the image to create multiple images.

#### 2. Final Model Architecture

Here is a visualization of the architecture (this Network is Nvidia Network)

![alt text][image1]

I customized the nvidia network. To be suited to learning my data

My model's summary is below

Layer (type)              |   Output Shape          |    Param #   
--------------------------|-------------------------|-------------
lambda_1 (Lambda)         |  (None, 160, 320, 3)    |  0         
cropping2d_1 (Cropping2D) |  (None, 65, 320, 3)     |  0         
conv2d_1 (Conv2D)         |  (None, 31, 158, 24)    |  1824      
activation_1 (Activation) |  (None, 31, 158, 24)    |  0         
conv2d_2 (Conv2D)         |  (None, 14, 77, 36)     |  21636     
activation_2 (Activation) |  (None, 14, 77, 36)     |  0         
conv2d_3 (Conv2D)         |  (None, 5, 37, 48)      |  43248     
activation_3 (Activation) |  (None, 5, 37, 48)      |  0         
conv2d_4 (Conv2D)         |  (None, 3, 35, 64)      |  27712     
activation_4 (Activation) |  (None, 3, 35, 64)      |  0         
conv2d_5 (Conv2D)         |  (None, 1, 33, 64)      |  36928     
activation_5 (Activation) |  (None, 1, 33, 64)      |  0         
flatten_1 (Flatten)       |  (None, 2112)           |  0         
dense_1 (Dense)           |  (None, 100)            |  211300    
activation_6 (Activation) |  (None, 100)            |  0         
dropout_1 (Dropout)       |  (None, 100)            |  0         
dense_2 (Dense)           |  (None, 50)             |  5050      
activation_7 (Activation) |  (None, 50)             |  0         
dense_3 (Dense)           |  (None, 10)             |  510       
activation_8 (Activation) |  (None, 10)             |  0         
dense_4 (Dense)           |  (None, 1)              |  11        
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0

Input : Images
output : steering degree

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, 
flipped images and angles thinking 
darkend images and angles thinking 

|![alt text][image6] | ![alt text][image7]|
----------------------|-------------------------
left side image       | flip image 
![alt text][image8] |![alt text][image9]
normal image         | darken image

and randomly make a shadow on image, and shift images by randomized x offset

After the preprocessing, I can get a enough data
So I can train my model very well
Finally I was able to complete the course successfully. and the output is saved to run1.mp4