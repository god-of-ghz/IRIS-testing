# The ML libary we kinda love
import os
import tensorflow as tf
# Keras -> Wrapper for TF
from tensorflow import keras
# Layers -> Contains types of layers to use in our NN
from tensorflow.keras import layers
from tensorflow.python.client import device_lib

# Other useful library for statistics and learning 
import sklearn
# Using to download the iris dataset
import sklearn.datasets
# Using to separate our model into a training and test set
from sklearn.model_selection import train_test_split as tts

import numpy as np

# some CPU/GPU stuff
tf.config.set_visible_devices([], 'GPU')

# Load the dataset
dataset = sklearn.datasets.load_iris()
X = np.array(dataset.data[:, :], dtype=np.float32)
y = np.array(dataset.target, dtype=np.int32)
input_shape = [4,]

# Split and reshape the dataset
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3)
X_train = np.array(X_train)
X_test = np.array(X_test)
shape = [-1] + input_shape
X_train = np.reshape(X_train, shape)
X_test = np.reshape(X_test, shape)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


# Define and compile model
model = keras.Sequential(       # computer vision often uses sequential
    [
    # ML doesnt HAVE to be an nn
    # example: 'layers are sequential in nature'
	#   - input --> each layer --> output
	
    # non-sequential
    #   - information flow may loop
	#   - often revolves around time dependence
    # two layers
        keras.Input(shape=input_shape),
        layers.Flatten(),       # if multi-D data, turns larger dim matrix into vector
        layers.Dense(3, activation="relu", name="MID"),     # a 'dense' layer has connections to every node inthe previous and next layer
        # 3 nodes: that obtain value from input
        # activation: thresholding scheme- if a value incoming to a node is less, than it can just ignore it
        # relu is just a standard for each layer- 
        # name, just for organization
        layers.Flatten(),
        layers.Dense(3
        , activation="softmax", name="OUT")
        # use softmax to put values between 0 and 1, makes it easier to assign probability/classify input
    ]
)
# optimizer, different kinds. how to update the learning coefficients are changed
# Adam is fairly standard
optim = keras.optimizers.Adam(learning_rate = 0.01)
# defining the model's "loss"
loss='categorical_crossentropy'
metrics = ['accuracy']
model.compile(optimizer=optim,
              loss=loss,
              metrics=metrics)


# Train the model
model.fit(X_train, 
          y_train, 
          batch_size=5,
          epochs=50,
          validation_data = (X_test, y_test))
          
 # this wont actually save the model, there is some function in keras that does this