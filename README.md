# Cifar10_keras_example
training: python cifar10_keras_training.py -w weight_FP.hdf5

try inference with FP weight:
1. comment out the scaled (lambda) layer for FP weight inference 
2. python cifar10_keras_inference.py -w weight_FP.hdf5

quantize weight: 
1. change weight name in the minmax.py file
2. python minmax.py

inference with quantized weight:
1. add lambda scaling layer in for quantized weight inference
2. python cifar_10_keras_inference.py -w weight_quantized.hdf5

