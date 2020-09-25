'''
Trains a simple Keras Adapted CNN on the CIFAR-10 dataset using FP implementation.
Gets to 83.27% test accuracy after 89 epochs using tensorflow backend

Example:
Training Mode:
python cifar_keras.py -w weights.hdf5

'''

from __future__ import print_function
import numpy as np
np.random.seed(0)  # for reproducibility
import tensorflow as tf

import keras.backend as K
import csv
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Lambda, Dense, Dropout, Activation, BatchNormalization, MaxPooling2D, Conv2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.utils import np_utils
from keras.models import Model
from keras.regularizers import l2
from sec_ops import relu_layer as relu_layer_op
from sec_ops import softmax_layer as softmax_layer_op
import argparse
import math

import matplotlib.pyplot as plt

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", type=str,
        help="(optional) path to weights file")
ap.add_argument("-l", "--load-model", type=int, default=-1,
        help="(optional) whether or not pre-trained model should be loaded")

args = vars(ap.parse_args())

def relu_layer(x):
    return relu_layer_op(x)

def softmax_layer(x):
    return softmax_layer_op(x)

batch_size = 32
epochs = 100
channels = 3
img_rows = 32
img_cols = 32
classes = 10
use_bias = False

# Batch Normalization
epsilon = 1e-6
momentum = 0.9
weight_decay = 0.0004

######################
from data.cifar_test_yarray import y_test
from data.cifar_train_yarray import y_train
from data.cifar_test_xarray import X_test
from data.cifar_train_xarray import X_train

Y_train = np_utils.to_categorical(y_train, classes) # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, classes)

print(X_train.shape, 'train samples')
print(X_test.shape, 'test samples')
print(Y_train.shape, 'train samples values')
print(Y_test.shape, 'test samples values')

model = Sequential()

#Conv1 and ReLU1
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(channels, img_rows, img_cols), data_format='channels_first', kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv1'))
model.add(Activation(relu_layer, name='act_conv1'))

#Conv2 and ReLU2
model.add(Conv2D(32, kernel_size=(3,3), data_format='channels_first', kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv2'))
model.add(Activation(relu_layer, name='act_conv2'))

#Pool1
model.add(MaxPooling2D(pool_size=(2,2), name='pool1', data_format='channels_first'))

model.add(Dropout(0.25))

#Conv3 and ReLU3
model.add(Conv2D(64, kernel_size=(3,3), data_format='channels_first', kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv3'))
model.add(Activation(relu_layer, name='act_conv3'))

#Conv4 and ReLU4
model.add(Conv2D(64, kernel_size=(3,3), data_format='channels_first', kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv4'))
model.add(Activation(relu_layer, name='act_conv4'))

#Pool2
model.add(MaxPooling2D(pool_size=(2,2), name='pool2', data_format='channels_first'))

model.add(Dropout(0.25))

model.add(Flatten())

#FC1, Batch Normalization and ReLU5
model.add(Dense(512, use_bias=True, name='FC1', kernel_initializer='he_normal'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn1'))
model.add(Activation(relu_layer, name='act_fc1'))

model.add(Dropout(0.5))

#FC2, Batch Normalization and ReLU6
model.add(Dense(classes, use_bias=True, name='FC2', kernel_initializer='he_normal'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn2'))
model.add(Activation(softmax_layer, name='act_fc2'))

#Optimizers
opt = RMSprop(lr=0.0001, decay=1e-6)
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
model.summary()

WEIGHTS_FNAME = args["weights"]

#model.load_weights(WEIGHTS_FNAME, by_name=True)


if args["load_model"] > 0:
    print('Loading existing weights')
    model.load_weights(WEIGHTS_FNAME)
else:
    checkpoint_scheduler = ModelCheckpoint('output/weights.{epoch:02d}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='max', period=1)

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size, epochs=(epochs),
                        verbose=1, validation_data=(X_test, Y_test),
                        callbacks=[checkpoint_scheduler], shuffle=True)
    print("[INFO] dumping weights to file...")
    model.save_weights(args["weights"], overwrite=True)


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Top-1 accuracy:', score[1])

# #plot training acc
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()


#print output to txt

# outputs = []
# keras_function = K.function([model.input], [model.layers[0].output])
# outputs.append(keras_function([X_train, 1]))
#
# print(outputs)
# print(np.shape(outputs))
# output_reshape = np.reshape(outputs, 1638400000)
# print(output_reshape)
# print(np.shape(output_reshape))
# output_clip = output_reshape[:100000000]
# with open("output2.txt" , "w") as f:
#     for item in output_clip:
#         f.write("%s\n" % item)

#82.59% acc weight 84








