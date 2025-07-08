from __future__ import print_function
import keras 
import numpy as np

np.random.seed(42)

class Layer:      # Dummy class, allows to do .forward() and .backward()
    
    def __init__(self):
        pass
        
    def forward(self, input):       # Takes input data of shape [batch, input_units] and returns [batch, output_units]
        return input 

    def backward(self, input, grad_output):     # Performs back propagation step
        num_units = input.shape[1]      # Gets number of 'neurons' in the layer
        d_layer_d_input = np.eye(num_units)     # Creates an identity matrix of num_units x num_units
        return np.dot(grad_output, d_layer_d_input)     # Finds the dot product of the incoming gradient from the next layer(grad_output) with the local gradient of this layer(d_layer_d_input)
                                                        # Computes the gradient of loss w.r.t input of this layer
                                        
                                        
class ReLU(Layer):      # Applies element wise function to all inputs
    
    def __init__(self):
        pass
        
    def forward(self, input):
        relu_forward = np.maximum(0, input)         # 