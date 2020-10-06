'''Prints intermediate layer output
Example:
Inference: python print_intermediate_layer.py
'''

import numpy as np
np.random.seed(1437)  # for reproducibility
import tensorflow as tf
import csv
import argparse
import math
import h5py

## Variables ###
layers_array = ["scaling1", 'scaling2', 'scaling3','scaling4']
#####


for layer in layers_array:
	file_name = "output/" + layer + ".pkl"
	output_file = "output/" + layer + ".txt"
	data = np.load(file_name, allow_pickle=True)

	f = open(output_file, 'w')

	f.write('Format: Input Number, Filter, Row Number, Column Number\n')
	f.write("Size: {}\n".format(data.shape))
				
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			for k in range(data.shape[2]):
				for l in range(data.shape[3]):
				        f.write('\n' + "({},{},{},{}): {}".format(i, j, k, l, data[i,j,k,l]))
				
	print("[INFO] Generated output layer {} file {}".format(layer,output_file))


