from __future__ import print_function
import keras 
import numpy as np

np.random.seed(42)

class Layer:      # Dummy class, allows to do .forward() and .backward()
    
    def forward(self, input):       # Takes input data of shape [batch, input_units] and returns [batch, output_units]
        return input 

    def backward(self, input, grad_output):     # Performs back propagation step
        
        num_units = input.shape[1]      # Gets number of 'neurons' in the layer
        
        d_layer_d_input = np.eye(num_units)     # Creates an identity matrix of num_units x num_units
        
        return np.dot(grad_output, d_layer_d_input)     # Finds the dot product of the incoming gradient from the next layer(grad_output) with the local gradient of this layer(d_layer_d_input)
                                                        # Computes the gradient of loss w.r.t input of this layer
                                        
                                        
class ReLU(Layer):      # Applies f(x) = max(0, x) element-wise    
    
    def forward(self, input):        # Apply ReLU activation Args: input: numpy array of shape [batch_size, features]
            return np.maximum(0, input)        # Returns: output: numpy array of shape [batch_size, features] with ReLU applied
    
    def backward(self, input, grad_output):     # Compute gradient of ReLU. Args: input: numpy array of shape [batch_size, features] - original input
                                                # grad_output: numpy array of shape [batch_size, features] - gradient from next layer
        # ReLU derivative is 1 for positive inputs, 0 for negative
        relu_gradient = input > 0
        
        return grad_output * relu_gradient       # Returns: 'grad_input': numpy array of shape [batch_size, features] - gradient * ReLU derivative

        
class Dense(Layer):
    
    def __init__(self, input_units, output_units, learning_rate = 0.1):     # Input_units: number of features in input data - i.e 28x28 -> 784
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
        
        grad_input = np.dot(grad_output, self.weights.T)      # Find out how we contributed to grad_output error, multiply grad_output by transpose of weights matrix. Transpose ensures matrix dimensions align 
       
        grad_weights = np.dot(input.T, grad_output)      # Gradient w.r.t weights - finds out how much each weight contributed to overall loss
                                                         # Transpose here 'pairs' each input feature with corresponding output gradient across entire batch
                                                      
        # grad_biases = grad_output.mean(axis=0) * input.shape[0]   
        grad_biases = grad_output.sum(axis = 0)      # Easier way of achieving above line, grad_output is a 2d array representing the gradient of loss function w.r.t layers output
                                                     # i.e how much final loss would change if output neuron changed. Shape is [batch_size, number_of_neurons]
                                                     # .sum(axis = 0) sums elements of this array along x=0 axis ie down the rows. Whole operation finds total influence of bias on batchs loss
                                                
        # assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        
        self.weights = self.weights - self.learning_rate * grad_weights     # Updates weights, learning rate ensure step is small and controlled
        self.biases = self.biases - self.learning_rate * grad_biases    # Updates biases, ""                                                  ""
        
        return grad_input   # Becomes grad_output for previous layer
        

#==============================================================================
#                            Loss Functions
#==============================================================================
class Loss:
    
    @staticmethod
    def softmax_crossentropy_with_logits(logits, reference_answers):        # Logits are the raw scores, array of shape [batch_size, num_classes], each row is set of scores for one input sample, each column a class 
        
        logits_for_answers = logits[np.arange(len(logits)), reference_answers]      # np.arrange(len(logits)) creates array of row indices - scores for that image effectively, reference_answers provides column indices ie 1-9
        
        cross_entropy_loss = -logits_for_answers + np.log(np.sum(np.exp(logits), axis = -1))     # Essentially does softmax of cross entropy but allows for very large numbers that would break softmax.
                                                                                             # Cross entropy loss is a measure of how different the models predicted probabilities are from actual label 
                                                                                            
        return cross_entropy_loss
       
    @staticmethod
    def grad_softmax_crossentropy_with_logits(logits, reference_answers):     # Gradient of xentropy loss w.r.t raw outpt (logits) - starting point for backpropogation
            
        ones_for_answers = np.zeros_like(logits)    # Creates an array with same shape as logits but all initialised to 0
            
        ones_for_answers[np.arange(len(logits)), reference_answers] = 1     # FOr each row (sample image), places a 1 in the column responding to correct class index - creates a 100% probability truth
            
        softmax = np.exp(logits) / np.exp(logits).sum(axis = -1, keepdims = True)   # Creates predicted probabilities, makes all values between 0 and 1. keepdims keeps dimensions of matrices
            
        return (-ones_for_answers + softmax) / logits.shape[0]    # (-ones_for_answers + softmax) is perdicted - actual, / logits.shape[0] divides by batch size to find avg gradient across the batch  
    
            
class NeuralNetwork:        # Capture networks state (layers) amd behaviour( training, predicting) into single object
    
    def __init__(self, layers):     # layers = list of layers objects, when NN instance is created it is passed a list of layer objects
        self.layers = layers
        
    def forward(self, inputs):     # inputs is a numpy array [batch_size, input_features]
        
        activations = []    # Empty list to store output from each layer
        
        current_input = inputs
        
        for layer in self.layers:   # Loops through each layer in self.layers, returns activations list with full history of ouputs
            output = layer.forward(current_input)   # Performs layer specific computation
            activations.append(output)
            
            current_input = output      # output becomes input for next layer in loop
            
        return activations
        
    
        