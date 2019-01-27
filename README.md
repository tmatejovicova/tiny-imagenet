# Tiny Imagenet: Image Classification
Author: Tatiana Matejovicova  
University of Cambridge
14 December 2018

## Abstract
The ImageNet Large Scale Visual Recognition Challenge was started in 2010 to compare performance of object detection algorithms across a large variety of objects and has been used as an object recognition benchmark. Tiny ImageNet Visual Recognition Challenge is an equivalent challenge of a smaller size with 200 classes. Each class has 500 training images, 50 validation and testing images, all of a size 64x64 pixels. In this project the Tiny ImageNet database is used to design, train and test a classification deep neural network. Multi-class logistic regression and one hidden layer fully connected networks are used as a baseline and this is compared to two variations of convolutional neural networks.

## Results
The highest top-1 accuracy of 24.5% was achieved for a CNN that was based on the VGG model. The highest top-5 accuracy of 48.8%  was achieved with the same model which means that it produced the correct label in the top five guesses for about half of the image samples. The performance could be improved significantly by adding more convolution blocks and increasing the number of filters. Furthermore the training procedures could be improved by tuning the optimisation parameters and using more complex techniques such as scheduled learning rate and momentum. A completely alternative approach that would be likely to yield high accuracy would be to use transfer learning.

## Code
Code is organised in the following files.
- script.py - Run training and testing for all the models
- train_test.py - Train and test the specified model
- load_data.py - Load the images
- models.py - Definition of models
- plot_history.py - Plot training and validation loss and accuracy evolution
- constants.py - Define constants for the whole project

## Note
See report.pdf for more information, analysis and conclusions.
