# Tutorial Part 1
We'll be looking at some introductory material with the Iris (the flower)
dataset.

## Starting with Data Visualization
We want to visualize the data to see how well our solution is suited for the
problem.

In this example, we want to classify the irises using a linear model. For
a linear model to be viable, our data has to be linearly separable in
some way (whether naturally or after some feature transformations).

Because our data samples are >3-dimensional, we will apply Principle
Component Analysis (PCA) to project the points onto their three most
"important" axes. 

Although you can code up your own PCA implementation, we will be using
sklearn.decomposition.PCA.

After doing PCA, we can plot the new dimensionally-reduced samples using
matplotlib. After doing so, we see that the data is very well separated; almost
as if we can take two straight lines to seperate these clusters. So that is what
we will do.

## A Linear Model
To classify the iris data, we will use sklearn.svm.LinearSVC.

Sklearn makes this relatively simple. We can create a new classifier
by calling the LinearSVC function with the max number of iterations we 
want to "train" our model AND the type of loss function we would like to use.

It is possible to leave these blank as they probably have default values.

We fit our model on the training data -- then we run our model with the test 
samples.

We see that we can get 100% accuracy on both our training and test set. 

## Extra step
To see what we can do, we retry our experiment using a non-linear model -- 
a very simple neural network. 

Network:
```
input -> flatten
dense layer 3
- RELU activation
- flatten
output layer: dense layer 3
- softmax activation

After simple training for 100 epochs, we see that our final validation accuracy
is also 100%.

We can see that by understanding and exploring the distribution of the data, we
can achieve the same performance as a neural network with less overhead.
