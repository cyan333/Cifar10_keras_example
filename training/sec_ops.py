# -*- coding: utf-8 -*-
from __future__ import absolute_import
import keras.backend as K
import tensorflow as tf

def relu_layer(x):
    return K.relu(x, alpha=0.0, max_value=None)

def softmax_layer(x):
    return K.softmax(x, axis=-1)
