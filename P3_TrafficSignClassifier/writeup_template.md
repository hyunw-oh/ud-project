# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./data_visualization.png "Visualization"
[image1]: ./data_graph.png "Visualization"
[image4]: ./german_data/1.jpg "Traffic Sign 1"
[image5]: ./german_data/18.jpg "Traffic Sign 2"
[image6]: ./german_data/25.png "Traffic Sign 3"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

Number of classes = 43

* The size of training set is ?
Number of training examples = 34799

* The size of the validation set is ?
Number of validation examples = 4410

* The size of test set is ?
Number of testing examples = 12630

* The shape of a traffic sign image is ?
Image data shape = (32, 32, 3)

* The number of unique classes/labels in the data set is ?
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image0]

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, 
---

I've always shuffled the data to avoid the same result.

Shuffling sometimes improves the performance of the model.

    X_train, y_train = shuffle(X_train, y_train)

so the normalized image is in [ -1< normal_data <1 ]

    normalized_train =np.divide(np.subtract(np.average(X_train,axis=3),128.0),128.0)   

and reshape the data for training.

    normalized_train =np.reshape(normalized_train,(normalized_train.shape[0],normalized_train.shape[1],normalized_train.shape[2],1))

this data type is suitable for softmax_cross_entropy_with_logits

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:
This is LeNet

| Layer             	|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 normalized data for RGB image   		| 
| Convolution 3x3      	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|        										|
| Max pooling           | 2x2 stride,  outputs 14x14x6   				|
| Convolution 3x3       | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU	                | 												|
| Max pooling           | 2x2 stride,  outputs 5x5x16   				|
| Flatten       		| output 400   									|
| Fully Connected		| output 120        							|
| RELU					|       										|
| Fully Connected		| output 84         							|
| RELU					|       										|
| Fully Connected		| output 10            							|
| RELU					| logists										|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an LeNet based on Udacity Source
logists LeNet
cost function softmax_cross_entropy_with_logits
optimizer adam_optimizer
*Using Adam Optimizer, I can expect all the advantages of Momentum Optimizer and RMSProp Optimizer
batch size 128
epochs 43
learning late 0.001
*If the learning late is too small, it will take too long or stop, and if it is too large, it will not be found.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.93
* test set accuracy of 0.93

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
KNN is very simple and this is the image recognition model.

* What were some problems with the initial architecture?
Initially, KNN was also considered for application. However, KNN was very dependent on the input data and terribly slow.
If the input data is reddish, the test data is more likely to be recognized by red light. And if the left-leaning image and the right-leaning image come in evenly, the recognition rate of the class is lowered.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
In network, I used a way to shuffle data, because the order in which data is learned also affects performance.
Although this does not guarantee the same result every time, I can get a good model once or twice with this method many times.

* Which parameters were tuned? How were they adjusted and why?
I used flatten
Using flatten, I can do linear operation for image.
and I used this form
 
      loss_func = tf.matmul(flattend_img, truncated_norm) + bias
      
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  I used truncated_norm
  in Tensorflow API
  Outputs random values from a truncated normal distribution.
  The generated values follow a normal distribution with specified mean and standard deviation, except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
  So, rather than using a generic random function, the results do not always converge, and I can get closer to the layer more and more.

If a well known architecture was chosen:
* What architecture was chosen?
LeNet

* Why did you believe it would be relevant to the traffic sign application?
LeNet is the first model to have a meaningful deep learning recognition.
This model successfully analyzed the zip code and later became the base model of AlexNet, which revolutionized recognition in the CNN domain.
Our subject is recognizing the sign, so it is considered to be a relatively simple object, so it can be expected to achieve high performance even with relatively simple LeNet.
  
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Yeah, I used a validation set to ensure accuracy.
Every Epochs, I print validation data accuracy.
EPOCH 1 ...
Validation Accuracy = 0.727
EPOCH 2 ...
Validation Accuracy = 0.835
...
EPOCH 43 ...
Validation Accuracy = 0.933
model saved.
So I am sure that the model has good accuracy.
If the test data is not too different

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 

The third image might be difficult to classify because the data has black margine upside and downside.
but our train data has no image with margin.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.jpg (Velocty) 		| Velocty	    								| 
| 2.jpg (Warn)    		| Warn   										|
| 3.png	(Road construct)| Yield											|

The model was able to correctly guess 2 of the 3 traffic signs, which gives an accuracy of 66.6%. 
This compares favorably to the accuracy on the test set of tarin data.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.
      
For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were
1.jpg 
| Probability         	|     Prediction (class index) 					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99995708e-01        | 24       	    								| 
| 4.27906161e-06     	| 2      										|
| 1.32989247e-12		| 1     										|
| 1.14198050e-12  		| 21        					 				|
| 5.30826337e-16	    | 5                   							|

For the second image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were
18.jpg
| Probability         	|     Prediction (class index) 		    		| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00       	| 18           									| 
| 3.64476129e-08   		| 11      										|
| 2.15828917e-11		| 27    										|
| 9.35578413e-18	    | 1         					 				|
| 4.18892017e-20		| 38                  							|

For the thith image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were
25.png
| Probability         	|     Prediction (class index) 					| 
|:---------------------:|:---------------------------------------------:| 
| 8.50478053e-01        | 25           									| 
| 1.15288205e-01    	| 37       										|
| 3.26423496e-02		| 31   											|
| 1.54759665e-03	    | 24        					 				|
| 1.91868021e-05	    | 29                  							|

