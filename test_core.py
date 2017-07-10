from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import scipy
import numpy as np
import argparse  

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


# To make user-friendly command line interface
parser = argparse.ArgumentParser(description='Decide if an image is a picture of a logo')  
parser.add_argument('image', type=str, help='The image file to check')
args = parser.parse_args()


# Same network definition as before, Packages imported from TFLearn
img_prep = ImagePreprocessing()            
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()

# Passing the input parameters, 32x32 3 channel image with it's mean and standard deviation to the network
network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)

# Setting layer 1 with 64 nodes with activation function as rectified linear unit:
network = conv_2d(network, 64, 3, activation='relu')

# Max pooling of layer 1
network = max_pool_2d(network, 2)

# Setting layer 2 with 128 nodes with activation function as rectified linear unit:
network = conv_2d(network, 128, 3, activation='relu')

# Setting layer 3 with 256 nodes with activation function as rectified linear unit:
network = conv_2d(network, 256, 3, activation='relu')

# Again Max pooling
network = max_pool_2d(network, 2)

# Setting 512 node for fully connected layer with activation function as rectified linear unit.
network = fully_connected(network, 512, activation='relu')

# To prevent over-fitting ( Dropout - throw away some data randomly during training. )
network = dropout(network, 0.5)

# Creating a fully connected layer for output with activation function as Softmax as we are dealing with multiclass classification.
network = fully_connected(network, 10, activation='softmax')

# To train the network we will use adaptive moment estimation (ADAM) and categorical_crossentropy  
# to determine loss in learning process and optimization.
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.01)

# Covering the network in a model object and calling the trained dataset file.
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='model\logo-classifier.tfl.ckpt-2737')

#Loading the trained dataset file
model.load("model\logo-classifier.tfl.ckpt-2737")

# Load the image file
img = scipy.ndimage.imread(args.image, mode="RGB")

# Scale it to 32x32
img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')

# Predict
prediction = model.predict([img])

# Check the result.
is_logo = np.argmax(prediction[0]) ==1
print (is_logo)

if is_logo:
    print("That's a logo!")
else:
    print("That's not a logo!")