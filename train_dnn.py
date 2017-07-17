from __future__ import print_function
import numpy as np

import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras import regularizers
from keras.models import model_from_json
from keras.utils import np_utils
import pickle
import os

from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, BinaryConv2D   # we have used Bianry weights during propagations.

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


def binary_tanh(x):
    return binary_tanh_op(x)


H = 1.
kernel_lr_multiplier = 'Glorot' # For finding a good variance for the uniform distribution from which the initial parameters are drawn.

# Neural Netwotrk Architecture
batch_size = 128 # Batch size
epochs = 1 # Number of epochs
channels = 3 # Number of channel
img_rows = 32 # Width
img_cols = 32 # Height
filters = 32  # Number of pooling filter
kernel_size = (3, 3) # Size of feature detector
pool_size = (2, 2) # Size of pool matrix 
hidden_units = 128
classes = 10  # Number of classes
use_bias = False # Boolean, whether the layer uses a bias vector. 

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# Binary network
epsilon = 1e-4  # Fuzz factor
momentum = 0.9  # Parameter updates Momentum

# dropout
p1 = 0.25
p2 = 0.5

# the data split between train and test sets

with open("logo_dataset.pickle", 'rb') as f:
  save = pickle.load(f)
  X_train = save['train_dataset']         # assign X as train dataset
  y_train = save['train_labels']          # assign Y as train labels 
  X_test = save['test_dataset']       # assign X_test as test dataset 
  y_test = save['test_labels']        #assign Y_test as test labels
  del save
 

X_train = X_train.reshape(70000, 3, 32, 32).astype('float32')   # Reformatting shape array to give a scalar value for dataset.  
X_test = X_test.reshape(7000, 3, 32, 32).astype('float32')   # Reformatting shape array to give a scalar value for dataset.  

X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, classes) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, classes) * 2 - 1


model = Sequential()  # This model is a linear stack of layers.

# conv1
model.add(BinaryConv2D(128, kernel_size=kernel_size, input_shape=(channels, img_rows, img_cols),
                       data_format='channels_first',kernel_regularizer=regularizers.l2(0.01),
                       activity_regularizer=regularizer.l1(0.01),H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias, name='conv1'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn1'))
model.add(Activation(binary_tanh, name='act1'))   

# conv2
model.add(BinaryConv2D(128, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       data_format='channels_first',kernel_regularizer=regularizers.l2(0.01),
                       activity_regularizer=regularizer.l1(0.01)
                       ,padding='same', use_bias=use_bias, name='conv2'))
model.add(MaxPooling2D(pool_size=pool_size, name='pool2'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn2'))
model.add(Activation(binary_tanh, name='act2'))

# conv3
model.add(BinaryConv2D(256, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',kernel_regularizer=regularizers.l2(0.01),
                       activity_regularizer=regularizer.l1(0.01),
                       padding='same', use_bias=use_bias, name='conv3'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn3'))
model.add(Activation(binary_tanh, name='act3'))

# conv4
model.add(BinaryConv2D(256, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',kernel_regularizer=regularizers.l2(0.01),
                       activity_regularizer=regularizer.l1(0.01),
                       padding='same', use_bias=use_bias, name='conv4'))
model.add(MaxPooling2D(pool_size=pool_size, name='pool4'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn4'))
model.add(Activation(binary_tanh, name='act4'))
model.add(Flatten())
# dense1
model.add(BinaryDense(1024, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense5'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn5'))
model.add(Activation(binary_tanh, name='act5'))
# dense2
model.add(BinaryDense(classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense6'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn6'))

opt = Adam(lr=lr_start)  # Define optimizer to use.
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
#model.summary()

lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1 , validation_data=(X_test, Y_test),
                    callbacks=[lr_scheduler])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print("InitialiZing save to C:/Users/AchmN/Desktop/logo/model: ")
#serialze model to JSON
model_json = model.to_json()
with open('C:/Users/AchmN/Desktop/logo/model/model.json', "w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
model.save_weights("model.h5")
print("Model saved to disk")
