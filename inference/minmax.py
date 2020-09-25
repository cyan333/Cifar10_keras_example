import os
import h5py
import numpy as np
import _pickle as pickle
from shutil import copyfile
import csv
import math

weight_hdFile_from_training = 'weight_0923.hdf5'
#define a name for the quantized weight file
weight_for_quantize = 'quantized_0923.hdf5'
#define a nme for clipped weight after quantization
weight_for_clip = 'clipped_0923.hdf5'

#copy training un-quantized weight to new file and overwrite with quantized weight
copyfile(weight_hdFile_from_training, weight_for_quantize)
# copyfile(weight_for_quantize, weight_for_clip)

def get_min_max(thisValue):
    print("get_minmax function")
    """Quantization based on linear rescaling over min/max range.
    """
    min_val, max_val = np.min(thisValue), np.max(thisValue)
    return min_val,max_val
    
def get_conv_minmax(hdf5_f):
    print("get_conv_minmax function")
    """quantize method.
    Strategy for extracting the weights is adapted from the
    load_weights_from_hdf5_group method of the Container class:
    see https://github.com/keras-team/keras/blob/master/keras/engine/topology.py#L2505-L2585
    """
    #Init dictionary to save max value
    conv_layer_max_value_dict = {}
    hdf5_file = h5py.File(hdf5_f, mode='r')
    # print("Keys: %s" % hdf5_file.keys())
    f= open("min_max_fine_2by2.txt", 'w')

    for layer_name in hdf5_file.keys(): #attrs['layer_names']:
        g = hdf5_file[layer_name]
        # print(layer_name)
        layer_name_str = str(layer_name)
        # loop thru all the attributes in every layer
        # for var in g.attrs:
        #     print("attribute name = " + str(var))
        for weight_name in g.attrs['weight_names']:
            weight_value = g[weight_name][()]
            #print("weight_name = " + str(g.attrs['weight_names']) + str(weight_value))
            min_val,max_val = get_min_max(weight_value)
            f.write(layer_name_str + " " + str(weight_name) + " " + str(min_val) + " " + str(max_val) + "\n")
            #save value into dictionary
            if(layer_name_str.startswith("conv")):
                conv_layer_max_value_dict[layer_name_str] = max(abs(max_val),abs(min_val))

    hdf5_file.close()
    #print(conv_layer_max_value_dict)
    return conv_layer_max_value_dict

# quantize weight functions
def quantize_singleweight(thisValue,alpha):
    print("quantize_singleweight function")
    """Quantization based on linear rescaling over min/max range.
    """
    gamma = 127/alpha;
    quantized = np.round(gamma*thisValue)
    #else:
    #    quantized = np.zeros(arr.shape)
    quantized = quantized.astype(np.int16)
    # print(quantized)
    return quantized

def quantize_weights(hdf5_f, max_dict):
    print("quantize_weights function")
    """quantize method.
    Strategy for extracting the weights is adapted from the
    load_weights_from_hdf5_group method of the Container class:
    see https://github.com/keras-team/keras/blob/master/keras/engine/topology.py#L2505-L2585
    """
    hdf5_file = h5py.File(hdf5_f, mode='a')
    # alpha = {'conv1': 0.60103804, 'conv2': 0.35357526, 'conv3': 0.5367581, 'conv4': 0.29506233}
    # array_updated_weights = {"conv1", "conv2", "conv3", "conv4"}

    for layer_name in max_dict.keys():  # attrs['layer_names']:
        g = hdf5_file[layer_name]

        for weight_name in g.attrs['weight_names']:
            weight_value = g[weight_name][()]
            quantized = quantize_singleweight(weight_value, max_dict[layer_name])
            weight_name_trim = str(weight_name)[2:-1]
            #print("weight name = " + str(weight_name)[2:-1])
            # print("quantize = " + str(quantized))
            # print(str(layer_name)+"/"+str(weight_name))
            del hdf5_file[str(layer_name)+"/"+weight_name_trim]
            print(str(layer_name)+"/"+weight_name_trim)
            hdf5_file[str(layer_name)+"/"+weight_name_trim] = quantized

    hdf5_file.close()

def clip(value, lowerBound, upperBound, max, min):
    if(value < lowerBound):
        output = min
    elif (value > upperBound):
        output = max
    else:
        output = value

    return output

def clip_weights(hdf5_f, max_dict):
    """quantize method.
    Strategy for extracting the weights is adapted from the
    load_weights_from_hdf5_group method of the Container class:
    see https://github.com/keras-team/keras/blob/master/keras/engine/topology.py#L2505-L2585
    """
    hdf5_file = h5py.File(hdf5_f, mode='a')
    # alpha = {'conv1': 0.60103804, 'conv2': 0.35357526, 'conv3': 0.5367581, 'conv4': 0.29506233}
    # array_updated_weights = {"conv1", "conv2", "conv3", "conv4"}

    for layer_name in max_dict.keys():  # attrs['layer_names']:
        g = hdf5_file[layer_name]

        for weight_name in g.attrs['weight_names']:
            weight_value = g[weight_name][()]
            # print('weight_value = ' + str(weight_value))
            low_values_flags = weight_value < -127
            weight_value[low_values_flags] = 0
            # print(low_values_flags)
            high_values_flags = weight_value > 127
            weight_value[high_values_flags] = 128
            # print(high_values_flags)
            weight_name_trim = str(weight_name)[2:-1]

            #print("weight name = " + str(weight_name)[2:-1])
            # print("quantize = " + str(quantized))
            # print(str(layer_name)+"/"+str(weight_name))

            del hdf5_file[str(layer_name)+"/"+weight_name_trim]
            print(str(layer_name)+"/"+weight_name_trim)
            hdf5_file[str(layer_name)+"/"+weight_name_trim] = weight_value

    hdf5_file.close()

print('start here -------> ')
conv_layer_max_value_dict = get_conv_minmax(weight_hdFile_from_training)

print(conv_layer_max_value_dict)

# generate max dictionary for inference scaling factor
fout = 'max_dict.csv'
f = open(fout, 'w')
for key, value in conv_layer_max_value_dict.items():
    f.write(str(key) + "," + str(value) + '\n\n')
f.close()

quantize_weights(weight_for_quantize, conv_layer_max_value_dict)

# clip_weights(weight_for_clip, conv_layer_max_value_dict)

print("FINISHED")







