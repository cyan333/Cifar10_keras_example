'''
Peforms 8-bit INT inference of a simple Keras Adapted CNN on the CIFAR-10 dataset.
Gets to 82.46% test accuracy after 89 epochs using tensorflow backend

Example:
  Inference Mode:
  python cifar_keras.py -w weights_int_8bit_signed.hdf5
  #load quantized weights

  Inference Mode and print intermediate layer pkl files
  python cifar_keras.py -w weights_int_8bit_signed.hdf5 -p 1
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

from binary_ops import relu_layer as relu_layer_op
from binary_ops import softmax_layer as softmax_layer_op
from binary_ops import floor_func as floor_func_op
from binary_layers import BinaryDense, BinaryConv2D
from matplotlib import pyplot as plt
import argparse
import math

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", type=str,
        help="(optional) path to weights file")
ap.add_argument("-p", "--print_layers", type=int, default=-1,
        help="(optional) To print intermediate layer pkl files")


args = vars(ap.parse_args())

def relu_layer(x):
    return relu_layer_op(x)

def softmax_layer(x):
    return softmax_layer_op(x)

def floor_func(x,divisor):
    return floor_func_op(x,divisor)

def clip_func(x):
    low_values_flags = x < -127
    x[low_values_flags] = 0

    high_values_flags = x > 127
    x[high_values_flags] = 128
    return x

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

layers_array = ["scaling1", "scaling2", 'scaling3','scaling4']

######################
#load scale
with open('max_dict.csv', mode='r') as infile:
    reader = csv.reader(infile, delimiter=',')
    data_read = [row for row in reader]

conv_scale = []
for i in range(0,4):
    conv_scale.append(math.floor(127/float(data_read[i*2][1])))
    print("conv scale" + str(i) + "====" + str(conv_scale[i]))

######################
from data.cifar_test_yarray import y_test
from data.cifar_test_xarray import X_test

Y_test = np_utils.to_categorical(y_test, classes)

print(X_test.shape, 'test samples')
print(Y_test.shape, 'test samples values')



model = Sequential()

#Conv1, Scaling1 and ReLU1
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(channels, img_rows, img_cols), data_format='channels_first', kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv1'))
model.add(Lambda(lambda x: floor_func(x,conv_scale[0]),name='scaling1'))  ## Dividing by 27 (MAV) and 18.296 (Instead of 128), so need to multiply by factor of 7 in gain stage
# model.add(Lambda(lambda x: K.clip(x, clip1_min, clip1_max), name='clip1'))
model.add(Activation(relu_layer, name='act_conv1'))


#Conv2, Scaling2 and ReLU2
model.add(Conv2D(32, kernel_size=(3,3), data_format='channels_first', kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv2'))
model.add(Lambda(lambda x: floor_func(x,conv_scale[1]),name='scaling2'))  ## Dividing by 288 (MAV) and 1 (Instead of 128), so need to multiply by factor of 128 in gain stage
# model.add(Lambda(lambda x: K.clip(x, clip2_min, clip2_max), name='clip2'))
model.add(Activation(relu_layer, name='act_conv2'))

#Pool1
model.add(MaxPooling2D(pool_size=(2,2), name='pool1', data_format='channels_first'))

#Conv3, Scaling3 and ReLU3
model.add(Conv2D(64, kernel_size=(3,3), data_format='channels_first', kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv3'))
model.add(Lambda(lambda x: floor_func(x,conv_scale[2]),name='scaling3'))  ## Dividing by 288 (MAV) and 2 (Instead of 128), so need to multiply by factor of 64 in gain stage
# model.add(Lambda(lambda x: K.clip(x, clip3_min, clip3_max), name='clip3'))
model.add(Activation(relu_layer, name='act_conv3'))

#Conv4, Scaling4  and ReLU4
model.add(Conv2D(64, kernel_size=(3,3), data_format='channels_first', kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv4'))
model.add(Lambda(lambda x: floor_func(x,conv_scale[3]),name='scaling4'))  ## Dividing by 576 (MAV) and 1 (Instead of 128), so need to multiply by factor of 128 in gain stage
# model.add(Lambda(lambda x: K.clip(x, clip4_min, clip4_max), name='clip4'))
model.add(Activation(relu_layer, name='act_conv4'))

#Pool2
model.add(MaxPooling2D(pool_size=(2,2), name='pool2', data_format='channels_first'))
model.add(Flatten())


# model.add(Lambda(lambda x: x*6, name='scaling_fc'))

#FC1, Batch Normalization and ReLU5
model.add(Dense(512, use_bias=True, name='FC1', kernel_initializer='he_normal'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn1'))
model.add(Activation(relu_layer, name='act_fc1'))

#FC2, Batch Normalization and ReLU6
model.add(Dense(classes, use_bias=True, name='FC2', kernel_initializer='he_normal'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn2'))
model.add(Activation(softmax_layer, name='act_fc2'))

#Optimizers
opt = RMSprop(lr=0.0001, decay=1e-6)
model.compile(loss='squared_hinge', optimizer=opt, metrics=['accuracy', 'top_k_categorical_accuracy'])
# model.compile('adam', 'categorical_crossentropy', ['accuracy', 'top_k_categorical_accuracy'])
model.summary()

WEIGHTS_FNAME = args["weights"]

model.load_weights(WEIGHTS_FNAME, by_name=True)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Top-1 accuracy:', score[1])
print ('Top-5 accuracy:', score[2])

## LAYER OUTPUTS TO DUMP
if args["print_layers"] > 0:
    for i in layers_array:
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(i).output)
        intermediate_output = intermediate_layer_model.predict([X_test])

        file_name = "output/" + i + ".pkl"

        print("Dumping layer {} outputs to file {}".format(i,file_name))
        intermediate_output.dump(file_name)





