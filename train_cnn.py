# Training acurracy of 0.95-0.99 and validation/test accuracy of 0.95-0.99 is achieved.

from __future__ import division, print_function, absolute_import

# Import tflearn and some other packages
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import tensorflow as tf
import pickle
import numpy as np
from six.moves import range

# Reading data from pickle file
def read_data():									
	with open("logo_dataset.pickle", 'rb') as f:
	    save = pickle.load(f)
	    X = save['train_dataset']					# assign X as train dataset
	    Y = save['train_labels']					# assign Y as train labels 
	    X_test = save['test_dataset']				# assign X_test as test dataset 
	    Y_test = save['test_labels']				#assign Y_test as test labels
	    del save
   
	return [X, X_test], [Y, Y_test]

def reformat(dataset, labels):   
    dataset = dataset.reshape((-1, 32, 32,3)).astype(np.float32) 	# Reformatting shape array to give a scalar value for dataset.  
    labels = (np.arange(10) == labels[:, None]).astype(np.float32)	 
    return dataset, labels

dataset, labels = read_data()
X,Y = reformat(dataset[0], labels[0])
X_test, Y_test = reformat(dataset[1], labels[1])
print('Training set', X.shape, Y.shape)
print('Test set', X_test.shape, Y_test.shape)            

# Shuffle the data
X, Y = shuffle(X, Y)					# Imported from TFLearn.

# Make sure the data is normalized
img_prep = ImagePreprocessing()    		#  Packages are imported from TFLearn.
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()


# Define the convolutional network architecture:

# 32x32 image is the input with 3 color channels (red, green and blue) with it's mean and standard deviation.
network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)


# Convolution 1 with 64 nodes with activation function as rectified linear unit:
network = conv_2d(network, 64, 3, activation='relu')

# Max pooling 1:
network = max_pool_2d(network, 2)

# Convolution 2 with 128 nodes with activation function as rectified linear unit:
network = conv_2d(network, 128, 3, activation='relu')

# Convolution 3 with 256 nodes with activation function as rectified linear unit:
network = conv_2d(network, 256, 3, activation='relu')

# Max pooling 2:
network = max_pool_2d(network, 2)

# Fully-connected 512 node neural network with activation function as rectified linear unit
network = fully_connected(network, 512 , activation='relu')

# Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Fully-connected neural network for outputs with activation function as softmax as we are dealing with multiclass classification.
network = fully_connected(network, 10, activation='softmax')

# To train the network we will use adaptive moment estimation (ADAM) and categorical_crossentropy  
# to determine loss in learning process and optimization.
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

# Covering the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path="model\logo-classifier.tfl.ckpt")

# Training! n_epoch will tell how many iterations the network has to go through, here it is kept 15 training passes and monitor it as it goes.
model.fit(X,Y, n_epoch=15, shuffle=True, validation_set=(X_test, Y_test), show_metric=True, batch_size=128, snapshot_epoch=True,
          run_id='logo-classifier')

# Save model when training is complete to a file
model.save("logo-classifier.tfl")
print("Network trained and saved as logo-classifier.tfl!")
