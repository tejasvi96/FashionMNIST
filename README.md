# FashionMNIST

# Background

This project considers the problem of classification of 10 clothing items using deep models like MLP and CNN on the [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.


# Experiment

We make use of two models-

1) Multi Layer Perceptron- Here the image feature of 28 *28  is flattened to 784 and a multi layer perceptron is used to model this classification problem. Softmax activation is used in the last layer and Cross Entropy Loss is used as the error function. 
2) CNN- A convolutional Neural Network is used which takes an image as in input and a combination of 2D convolutional layers and max pooling layers are used to arrive at a robust hidden representation for each input. Fully connected layer with softmax activation is used and Cross Entropy Loss is used as the error function.

Refer main.py for both the models.

Accuracy scores of 88.2% was achieved using MLP and 90.1% using CNN model on the test set.
