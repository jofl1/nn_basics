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
                                        
                                        
class ReLU(Layer):      # Applies f(x) = max(0, x) element-wise    
    
    def forward(self, input):   # Apply ReLU activation Args: input: numpy array of shape [batch_size, features]
            return np.maximum(0, input) # Returns: output: numpy array of shape [batch_size, features] with ReLU applied
    
    def backward(self, input, grad_output):     # Compute gradient of ReLU. Args: input: numpy array of shape [batch_size, features] - original input
                                                # grad_output: numpy array of shape [batch_size, features] - gradient from next layer
        # ReLU derivative is 1 for positive inputs, 0 for negative
        relu_gradient = input > 0
        
        return grad_output * relu_gradient  # Returns: 'grad_input': numpy array of shape [batch_size, features] - gradient * ReLU derivative

        
class Dense(Layer):
    
    def __init__(self, input_units, output_units, learning_rate = 0.1): # Input_units: number of features in input data - i.e 28x28 -> 784
                                                                        # Output_units: Int showing number of neurons in layer
        
        self.learning_rate = learning_rate
        self.input_units = input_units
        self.output_units = output_units
        
        self.biases = np.zeros(self.output_units)       # Initial biases: size of output_units (1D array of floats)
        
        self.weights = np.random.normal(                # Initialises weights using Xavier initialisation: https://365datascience.com/tutorials/machine-learning-tutorials/what-is-xavier-initialization/
            loc = 0.0,
            scale = np.sqrt(2 / (self.input_units + self.output_units)),    # Essentially the variance from which weights are drawn on the normal distribution
            size = (self.input_units, self.output_units)
        )
                                    
    
    def forward(self, input):       # Forward pass: output = input • weights + biases
        return np.dot(input, self.weights) + self.biases 
        
    def backward(self, input, grad_output):     # grad_output is gradient of the loss with respect to the layer's output. Passed from the next layer in the network
        
        grad_input = np.dot(grad_output, self.input.T)      # 
        