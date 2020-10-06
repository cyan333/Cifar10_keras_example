import numpy as np
import h5py
import math
import argparse

## Example usage: python convert_hdf5_to_txt.py -w output/weights_exp.hdf5.orig -o bin

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", type=str,
        help="(optional) path to weights file")
ap.add_argument("-o", "--output", type=str,
        help="(optional) path to output weights file")

args = vars(ap.parse_args())

f1 = h5py.File(args ["weights"], 'r+')     # open the file
f = open(args["output"],'w')

conv1_data = f1['/conv1/conv1/kernel:0']       # load the data


f.write('Conv Layer 1 Weights')

for l in range(conv1_data.shape[3]):
    for i in range(conv1_data.shape[0]):
        for j in range(conv1_data.shape[1]):
            for k in range(conv1_data.shape[2]):
                orig_data = conv1_data[i,j,k,l]
                conv1_data[i,j,k,l] = round(orig_data,0)
                f.write('\n' + "({},{},{},{}): {}".format(i, j, k, l, conv1_data[i,j,k,l]))

print('[INFO] Binarized Convolution layer 1 weights')

conv2_data = f1['/conv2/conv2/kernel:0']       # load the data

f.write('\n\n' + 'Conv Layer 2 Weights')
 
for l in range(conv2_data.shape[3]):
    for i in range(conv2_data.shape[0]):
        for j in range(conv2_data.shape[1]):
            for k in range(conv2_data.shape[2]):
                orig_data = conv2_data[i,j,k,l]
                conv2_data[i,j,k,l] = round(orig_data,0)
                f.write('\n' + "({},{},{},{}): {}".format(i, j, k, l, conv2_data[i,j,k,l]))



print('[INFO] Binarized Convolution layer 2 weights')

conv3_data = f1['/conv3/conv3/kernel:0']       # load the data

f.write('Conv Layer 3 Weights')

for l in range(conv3_data.shape[3]):
    for i in range(conv3_data.shape[0]):
        for j in range(conv3_data.shape[1]):
            for k in range(conv3_data.shape[2]):
                orig_data = conv3_data[i,j,k,l]
                conv3_data[i,j,k,l] = round(orig_data,0)
                f.write('\n' + "({},{},{},{}): {}".format(i, j, k, l, conv3_data[i,j,k,l]))

print('[INFO] Binarized Convolution layer 3 weights')

conv4_data = f1['/conv4/conv4/kernel:0']       # load the data

f.write('\n\n' + 'Conv Layer 4 Weights')
 
for l in range(conv4_data.shape[3]):
    for i in range(conv4_data.shape[0]):
        for j in range(conv4_data.shape[1]):
            for k in range(conv4_data.shape[2]):
                orig_data = conv4_data[i,j,k,l]
                conv4_data[i,j,k,l] = round(orig_data,0)
                f.write('\n' + "({},{},{},{}): {}".format(i, j, k, l, conv4_data[i,j,k,l]))



print('[INFO] Binarized Convolution layer 4 weights')



f1.close()                          # close the file


print("[INFO] Generated output binary weight file {}".format(args ["output"]))

f.close()
