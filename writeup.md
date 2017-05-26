[//]: # (Image References)

[image1]: ./images/lenet_architecture.png "Vanilla LeNet Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./images/left.jpg "Front-left Camera Image"
[image4]: ./images/center.jpg "Center Camera Image"
[image5]: ./images/right.jpg "Front-right Camera Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


---
# Behavioral Cloning
## 1 Project introduction

The aim of this project is to use behavioral cloning to enable a simulated car to drive at least one lap around a race track in a manner which safe for human passengers. This is to be done by recording driving the car around the track, using the recorded data to train a neural network to drive the car around the same track.
 
### Behavioral Cloning Technique

Behavioral cloning is a technique that tries to capture behavior in a model to apply the modelled behaviour to a similar situation. In this project the copied behaviour is a human driving a car around a race track in a simulator is captured through recording of car telemetry data while driving around the course. 

## 2 Project Setup

My github repository for this project contains five main files:
* model.py is the script to load, normalize and augment the training data as well as to create and train the model
* drive.py connects to the simulator and uses the model given through a parameter to drive the car in the simulator
* model.h5 contains the convolutional neural network trained by model.py
* report.md, the writeup of the project
* video.mp4 - a recording of the model driving around the first track from the perspective of the center camera

## 3 Generation of Training Data

To train the neural network training data was recorded by driving the car in the simulation around the track. The simulator provided by udacity captures the following information:
1. camera images from the<br>
    a) front center <br>
    b) front left<br>
    c) front right<br> 
    of the car.
2. steering angles
3. throttle
4. break
5. speed

From these only 1. and 2. were use as input for training.

As a start a basic dataset was recorded where the focus was put on driving the car as much as possible in the center of the lane. 

Doing this with a keyboard results in very stark steering angles as the only possible steering alngles are -25, 0 and +25 degrees. This results in quickly changing steering angles heavily biased towards these three values and not much in between. In curve this causes these stark angles to be applied intermittently. To avoid this unrealistic steering angle pattern behaviour the car was steered with an XBox360 controller which allows smoother, continuous steering angles through the analogue joysticks.
After recording of basic center of the road driving. With this data the model should be able to learn how to stay in the middle of the road. However the model is not perfect and should be able to respond to situations other than the ideal center road driving. If the car strays to far from the center it will not now how to recover and return to the center.
To enable the model to steer back to the middle of the road recovery data was generated where the recording was started in an undesirable situation, at the side of the road, and driven back to the center. This was repeated several times. Furthermore extra recordings were made of particularly challenging parts of the track. Those include a bridge with different road texture, a dirt lay by starting in a curve and sometimes strong shadows or dark trees.

As the simulated track is a round course the steering angle distribution is naturally biased towards one side which biases the model to steer the car to one side. To avoid that all existing training images have been flipped by the vertical axis and the steering angles, which range from -1 (leftmost steering angle) to +1 (rightmost steering angle), have been inverted. This has the added benefit of doubling the available training data. 
The training data was further augmented using not only the image of the simulated center camera but as well the images of the left and right camera. A set of images from all three cameras is provided below. However the steering angles used for images from the left and right camera have to be adjusted. I.e. the recorded steering angle, which belongs to the center images, is not applicable to the left/right image. For the left image the car should steer more to the right to get to the center of the road than the center steering angle suggest and the opposite applies for the right camera images. To accomodate for this to the steering angle used for the left images 0.2 was added and the same amount subtracted for the right images.
Combining flipping and using all three camera inputs equates a 6-fold increase in training data leading to a final dataset size of 52230 images. This helps the model to generalize better as well as it adds some recovery data through the side cameras as they often will be close the edges of the road teaching the model corrective steering angles for these situations.

| ![alt text][image3] | ![alt text][image4] | ![alt text][image5] | 
|:------------:|:-----------------:|:-----------------:|
| Left Camera | Center Camera| Right Camera |
 

The images were preprocessed using adapative histogram normalization to emphasize detail in low contrast areas, such as shadow. In form of lambda layers the image data was normalized to range from -0.5 to +0.5 and the top 43% and bottom 15% were cropped as they mostly contain irrelevant information such as the sky and the hood of the car. The essential information, relative location to the edge of the road as well as the angle to the edge can be obtained with the remaining section of the image. This reduces the data that needs to be analyzed without losing much relevant information and hence allows quicker training of the neural network.

## 4 Model Architecture

The first neural network implemented for this project was an unmodified version of LeNet.
 ![alt text][image1]
This the input, after passing through a cropping and a lambda layer for normalization go through a 5x5 convolutional layer with 6 filters and Relu activation followed by 2x2 Max Pooling. The it enters another 5x5 convolutional layer with 16 filters and Relu activation followed by 2x2 max pooling.
The resulting matrices are flattened and the number of features get funnelled through fully connected/dense layers with 120, 84 and 1 nodes (the predicted steering angle). 

To avoid overfitting the dataset was divided into training and validation set and an early-termination callback method was used to terminate training before overfitting occurred. The 6-fold data augmentation mentioned in "3. Generation of Training Data" as well prevented overfitting.

After unsuccessful experiments with the [Nvidia end-to-end autonomous driving neural network](https://arxiv.org/pdf/1604.07316v1.pdf) I reverted to the LeNet architecture but with increased number of filters of the convolutional layers. Instead of 6 and 16 filters 32 and 64 filters were used. The rational behind this was that the original number of features (6, 16) were insufficient to classify the different situations with adequate granularity. This architecture has as well the benefit of being very simple and fast to train, which aid the development of the model through shorter iterations and might as well be beneficial for inference in an autonomous vehicle which could be limited by available computational power.
The outcome of this was the model successfullt driving the car around the track as it can be seen in the linked video.
[Modell driving successfully around track](https://youtu.be/mANc1VkiWEc) 

## 5 Model Parameter Tuning

This project involved a lengthy trial and error phase that caused the implementation of the early termination and the checkpoint callback. This was benefitial as now instead of saving the model in it's state after last epoch the model was saved whenever it improved. Since it was observed that models do not always perform better on the track with a better meas squared error. A reason for this could be that once the car is off the track due to maybe very few wrong steering anlges it does not matter anymore how perfect the remaining ones would have been predicted.
The early termination callback function was not strictly necessary with the checkpoint callback function but helped cut training time.

The default learning rate of the [Adam optimizer](https://keras.io/optimizers/#adam) (0.001) was found to be too high leading to validation losses that were inconsistently jumping around and often not improving consistently. Therefore it was lowered to 0.0001. This lead to a more continuos decline of validation loss.

However the key change that enabled the model to succeed driving around the track was the increase in filters of the convolutional layers. This seemed to have an immediate impact on the models performance. However it could be the case that everything else was contributing to a good standard but the neural network was acting as a bottleneck of performance on the track.

## 6 Potential Improvements

I was very happy to finally make the car drive the course successfully. However it took me a long experiementation process to get there so there are some improvements that could help the model become more generalized and lead to even better results.
1. The steering angles in tight curves seem to be barely enough. To keep the car more in the center of the road it could benefit from stronger steering. Potentially multiplying the steering angles of the training data could help. But as well removing the strong bias towards straight steering angles and steering angles of -0.2 and +0.2 could help. Furthermore this depends as well on recorded training data which is to some extent subject to human racing game bias which contradicts staying in the center of the road.
2. Through more augmentation the model could become more generalizing and robust so that it drives better on track 2 of the simulator. Specifically adding random shaddows and randomly altering the brightness could help the performance on the second track with varying lighting conditions.
3. The Nvidia end-to-end driving network could be introduced again.
