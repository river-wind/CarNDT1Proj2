#**Traffic Sign Recognition** 

##Writeup for Chris Lawrence's second project for CarND Term1 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/german-road1.jpg "Traffic Sign 1"
[image5]: ./examples/german-road2.jpg "Traffic Sign 2"
[image6]: ./examples/german-road3.jpg "Traffic Sign 3"
[image7]: ./examples/german-road4.jpg "Traffic Sign 4"
[image8]: ./examples/german-road5.jpg "Traffic Sign 5"
[image9]: ./examples/german-road6.jpg "Traffic Sign 6"
[image10]: ./examples/german-road7.jpg "Traffic Sign 7"
[image11]: ./examples/german-road9.jpg "Traffic Sign 9"
[image12]: ./examples/german-road91.jpg "Traffic Sign 10"
[image13]: ./examples/german-road91_gray.jpg "Traffic Sign 11"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup

The Sign Classifier project was a difficult one, due to the shear number of possible options available to us which were not covered directly in the class.  Image manipulation options and how we might modify the Nueral Network were overwhelming, and most seemed to do nothing to improve the results of the classifier.  My [project code](https://github.com/river-wind/CarNDT1Proj2/blob/master/Traffic_Sign_Classifier-Copy1.ipynb) reflects the best output I was able to produce, using a mostly untouched LeNet model with greyscaled images.  I have left in place much of the code which was tried unsuccessfully, including my LeNet implementation which I assumed was not working given the behavior I was seeing; all of this unused code has been commented out.

###Data Set Summary & Exploration

The code for this step is contained in the second code cell of the IPython notebook.    The data set contained over 50,000 images of German traffic signs, sized into 32x32x3 RGB color images.  These had been split into three collections, Train, Validate, and Test.

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

I viewed some of the images and began a pre-processing of those images by converting them to grayscale next.  This preprocessing was done prior to the defined "Preprocessing" section so that I could check the output as part of the data visualization.  Before and after the grayscale conversion:

![alt text][image12]
![alt text][image13]

I attempted a dozen other preprocessing options, from sharpening to blurring to gamma adjustments, but all seemed to make the later stages of the process worse.  I have left some of that code intact in case I can make sense of why it wasn't working later, but for now it is all commented out.

I also attempted to normalize the image data because the Udacity Computer Vision course suggested it was a good idea.  That said, the few methods I attempted all failed to produce better results, and most made things drastically worse.  For example:

>corrected = cv2.normalize(corrected, corrected, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

made predictions slightly worse, while

>corrected = (corrected*255.0/corrected.max()) 

cause the predictions to be so bad as to be worthless, lower than 0.01.  It was not clear why this occured.


###Design and Test a Model Architecture

The code for this step is contained in the code cells under the heading Model Architecture of the IPython notebook.  The first sets the epoch and batch size, while the second contains my own code taken from the LeNet lab, now commented out.  The third cell contains the LeNet code nearly verbatim to try and eliminate it as the cause of the problems I encountered.  The only difference between this block on the LeNet solution is the addition of a fully connected layer at the bottom, which seemed to improve accuracy slightly.

####Training and Validation data

There was some confusion in the assignment over the training and validation data sets, as even though the instructions state that no validation data would be available as a part of the dataset, there was in fact a valid.p file containing validation data.  As such, there is no code to split the data in this project, as cross-validation was possible with the validation data provided.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB images   							| 
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6    				|
| Flatten   	      	| Input 5x5x16, Output 400      				|
| Fully connected		| Input = 400. Output = 120						|
| RELU					|												|
| Fully connected		| Input = 120. Output = 84						|
| RELU					|												|
| Fully connected		| Input = 84. Output = 43						|
| RELU					|												|
| Fully connected		| Input = 43. Output = 43						|
|						|												|
|						|												|
 
I did not include any dropout, despite trying the method in numerous locations.  It seems that the dropout did the least damage right after a fully connected layer, but it didn't seem to provide any benefit.  Since the model didn't seem to be overfitting, I assume the dropout wasn't able to function as intended.

####4. Training 

In the section titled "Train, Validate and Test the Model", I describe some of the methods attempted to improve the initial accuracy rate returned by the vanilla LeNet model.  

From there, I generated the one\_hot array needed for validation purposes, then set up the training pipeline.  Relying on the softmax cross entroy and the Adam Optimizer, the training peration was set to minimize on the reduced mean of the loss operator.

The evaluation function compares the predictions with the known labels, and the actual training sessions begin.  By batching the data into smaller chunks of 128 examples at a time and running on those batches, out of memory errors can be avoided.


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the last two cells of the Ipython notebook before "Step 3: Test a Model on New Images".

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.936
* test set accuracy of 0.908

The LeNet architecture was used as a starting point for this project because it was a logical choice; having just finished identifying 2D image data with letters, it was a natural selection for identifying 2D image data of non-letters.  I found that the LeNet design did an acceptable, but not supurb, job with the training and validation data.  I attempted to alter the training rate (lowering it), epoch size (increasing it), mean (increasing and decreasing), and stddev (increasing an decreasing) hyperparameters, but the results were substantially worse each time.  I attempted altering the LeNet model itself, by removing layers, altering the activation functions, changing layer orders, and adding dropout layers, but all changes I made decreased the accuracy of the validation testing.

It was not clear what steps should be taken to improve the model, and a number of threads on the forum suggested others were having similar issues.  A few members claimed that the LeNet model without image preprocessing was able to return above 0.93 without changes, but I was unable to reproduce that myself.

After roughly 40 hours of attempting to alter the LeNet model, I started over, and tried to add a single fully-connected layer after just changing the images to grayscale.  This produced somewhat better result than the LeNet alone, finally cresting the 0.93 thrshold.

Had I been able to identify a model which would overfit, I would have added dropout to produce a better result.  As it was, I never managed to find a suitable model to apply dropout to.


###Test a Model on New Images

####1. I chose 9 German traffic signs found on the web. I tried not to pick obviously clear signs, and some with different shading.

Here are the 9 German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11] ![alt text][image12]

One of the difficulties of selecting the images is that not just any German traffic sign would work.  Because the training set only identifies 43 types of signs, the signs which could be successfully tested must fall into that set of 43.  MAny possible signs would not be possible to match, and could not be included in the test.  For that reason, the first step was to learn what those 43 German Traffic signs looked like, and save a handful of them from the web.  I then opened each in Paint.net and modified them to 32x32 square color images (I later realized I could have done this in code with cv2, but having the images pre-cut made later processing faster).

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on the downloaded images is located in the cells under "Step 3: Test a Model on New Images".  The results seem to not match up with the results of the evaluation function, however.  Evaluation gives a prediction accuracy  of 0.300 against the new images, however the softmax specifics seems to suggest 0% accuracy.

Here are the specific results of the prediction:

| Image     			        |     Prediction	        					| 
|:-----------------------------:|:---------------------------------------------:| 
| Priority road         		| Vehicles over 3.5 metric tons prohibited		| 
| 30MPH zone        			| Vehicles over 3.5 metric tons prohibited		|
| Dangerous curve to the right	| Vehicles over 3.5 metric tons prohibited		|
| 30MPH zone        			| Vehicles over 3.5 metric tons prohibited		|
| Traffic signal    			| Vehicles over 3.5 metric tons prohibited		|
| No Entry           			| Keep left                             		|
| Bumpy Road        			| Vehicles over 3.5 metric tons prohibited		|
| Bicycle crossing    			| Vehicles over 3.5 metric tons prohibited		|
| Watch Ice and Snow   			| Vehicles over 3.5 metric tons prohibited		|

The model was able to correctly guess 0 of the 9 traffic signs, which gives an accuracy of 0%.

####3. Describe how certain the model is when predicting on each of the new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the first block under "Output Top 5 Softmax Probabilities For Each Image Found on the Web"

For the first image, the model has a good confidence in its prediction (probability of 0.99), The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Vehicles over 3.5 metric tons prohibited 		| 
| .0005     			| Road work 									|
| .00006				| General caution								|
| .0000002	      		| Stop      					 				|
| .00000007			    | Keep left         							|


For the second image, the model has a good confidence in its prediction (probability of 0.99), The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Vehicles over 3.5 metric tons prohibited 		| 
| .0000006     			| Priority road									|
| .000000008			| Speed limit (20km/h)							|
| .000000000008	      	| Stop      					 				|
| .000000000001		    | Road work         							| 

For the third image, the model has a good confidence in its prediction (probability of 0.99), The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Vehicles over 3.5 metric tons prohibited 		| 
| .0000001     			| Priority road									|
| .000000001			| Stop              							|
| .0000000001	      	| Slippery road					 				|
| .00000000001		    | Road work         							| 

For the fourth image, the model has a good confidence in its prediction (probability approaching 1.0), The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Vehicles over 3.5 metric tons prohibited 		| 
| ~0        			| Priority road									|
| ~0         			| General caution            					|
| ~0         	      	| Speed limit (70km/h)			 				|
| ~0         		    | Traffic signals         						| 

For the fifth image, the model has a good confidence in its prediction (probability of 0.98), The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.98         			| Vehicles over 3.5 metric tons prohibited 		| 
| .01        			| Keep left  									|
| ~0         			| Stop                      					|
| ~0         	      	| General caution    			 				|
| ~0         		    | Traffic signals         						| 

The sixth image is the only one different than the others. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.85         			| Keep left  									|
| .14        			| Vehicles over 3.5 metric tons prohibited 		|
| .001         			| General caution    			 				|
| ~0         	      	| Road work         			 				|
| ~0         		    | Speed limit (70km/h)     						| 

The remaining image predictions are very similar to the first 5.  Clearly, the model has an issue with the web images, which may be related to the limitede ability of the accuracy of the model to top 0.9, or it could be that the model doesn't generalize well.  given that some of the images are very clear and simple geometrically, the model should have faired better.

It is possible that the same data has a small number of examples of particular signs.  These could be effectiing the model, and could be handled by adding more data for those particular signs, or cleaning up the training data's examples of those signs.

In attempting to visualize the Network State with "Step 4: Visualize the Neural Network's State with Test Images", I found myself at a complete roadblock based on arameter two of the provided function.  the description says:

> tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer

However, nothing I provided to that variable restulted in the same error:
NameError                                 Traceback (most recent call last)
<ipython-input-98-0b3f80d94e6b> in <module>()
----> 1 outputFeatureMap(images_array,conv1_W)

NameError: name 'conv1_W' is not defined

In this case, conv1_W _is_ defined - it is the weights tensor for the first convolution layer of the network.  I'm not sure what should be provided here to make this function run based on the provided description.

###Conclusion
There was a significant degree of uncertainty in this project as how to procede.  Through forum posts, conversations with my mentor and dozens of hours of trial and error, I was able to produce an accuracy which hit the validation threshold targeted, however the tests against the web images and subsequest sections seem to execute very poorly.  Further investigation into how these items should function would be very helpful.