#!/usr/bin/env python

from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class Layer:      # Dummy class, allows to do .forward() and .backward()
    
    def forward(self, input):
        raise NotImplementedError

    def backward(self, input, gradient_from_next_layer):
        raise NotImplementedError   # Finds the dot product of the incoming gradient from the next layer(gradient_from_next_layer) with the local gradient of this layer(d_layer_d_input)
                                    # Computes the gradient of loss w.r.t input of this layer
                                        
                                        
class ReLU(Layer):      # Applies f(x) = max(0, x) element-wise    
    
    def forward(self, input):        # Apply ReLU activation Args: input: numpy array of any shape (works with both 2D and 4D tensors)
        return np.maximum(0, input)  # Returns: output: numpy array of same shape as input with ReLU applied
    
    def backward(self, input, gradient_from_next_layer):     # Compute gradient of ReLU. Args: input: numpy array of any shape - original input
                                                             # gradient_from_next_layer: numpy array of same shape as input - gradient from next layer
        # ReLU derivative is 1 for positive inputs, 0 for negative
        relu_gradient = input > 0
        
        return gradient_from_next_layer * relu_gradient       # Returns: gradient to pass to previous layer - gradient * ReLU derivative


class Flatten(Layer):    # Converts multi-dimensional input (like Conv2D output) into 1D vector per sample
                        # Essential for connecting convolutional layers to dense layers
    
    def forward(self, input):       # Flattens all dimensions except batch dimension
                                   # Args: input: numpy array of shape [batch_size, height, width, channels] or any other shape
        self.original_shape = input.shape      # Store original shape for backward pass - needed to unflatten gradients
        return input.reshape(input.shape[0], -1)    # Returns: output: numpy array of shape [batch_size, flattened_features]
                                                    # -1 tells numpy to calculate this dimension automatically
    
    def backward(self, input, gradient_from_next_layer):     # Reshape gradient back to original input shape
                                                            # Args: gradient_from_next_layer: numpy array of shape [batch_size, flattened_features]
        return gradient_from_next_layer.reshape(self.original_shape)    # Returns: gradient with same shape as original input
                                                                       # This ensures gradient flows correctly through Conv2D layers


class Conv2D(Layer):     # Performs 2D convolution operation - fundamental building block of CNNs
    
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, learning_rate=0.1):
        # input_channels: number of input channels (1 for greyscale, 3 for RGB)
        # output_channels: number of output channels (number of filters)
        # kernel_size: size of the convolutional kernel (assumes square kernel)
        # stride: step size for sliding the kernel across the image
        # padding: number of zeros to add around the image border
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.learning_rate = learning_rate
        
        # Initialise weights using random initialisation adapted for Conv layers
        # Shape: [output_channels, input_channels, kernel_height, kernel_width]
        self.weights = np.random.normal(
            loc=0.0,
            scale=np.sqrt(2 / (input_channels * kernel_size * kernel_size)),    # Fan-in for conv layers includes spatial dimensions
            size=(output_channels, input_channels, kernel_size, kernel_size)
        )
        
        self.biases = np.zeros(output_channels)    # One bias per output channel
    
    def forward(self, input):      # Performs the convolution operation
                                  # Args: input: numpy array of shape [batch_size, height, width, channels]
        batch_size, height, width, channels = input.shape
        
        # Apply padding if specified - adds zeros around the border of the image
        if self.padding > 0:
            input = np.pad(input, 
                         ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 
                         mode='constant')
        
        # Calculate output dimensions using the convolution formula
        output_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Initialise output tensor
        output = np.zeros((batch_size, output_height, output_width, self.output_channels))
        
        # Perform convolution - slide kernel across the image
        # This is the naive implementation for clarity, not optimised for speed
        for batch_idx in range(batch_size):                          # For each image in the batch
            for output_channel_idx in range(self.output_channels):   # For each output channel (filter)
                for height_idx in range(output_height):              # For each output height position
                    for width_idx in range(output_width):            # For each output width position
                        # Extract the patch from input that the kernel will operate on
                        height_start = height_idx * self.stride
                        width_start = width_idx * self.stride
                        input_patch = input[batch_idx, 
                                          height_start:height_start+self.kernel_size, 
                                          width_start:width_start+self.kernel_size, :]
                        
                        # Convolve: element-wise multiply and sum
                        # input_patch shape: [kernel_size, kernel_size, input_channels]
                        # weights[output_channel_idx] shape: [input_channels, kernel_size, kernel_size]
                        # We need to transpose weights to align dimensions properly
                        convolution_result = np.sum(input_patch * self.weights[output_channel_idx].transpose(1, 2, 0))
                        output[batch_idx, height_idx, width_idx, output_channel_idx] = convolution_result + self.biases[output_channel_idx]
        
        return output
    
    def backward(self, input, gradient_from_next_layer):     # Backpropagation through convolution 
                                                            # gradient_from_next_layer shape: [batch_size, output_height, output_width, output_channels]
        batch_size, height, width, channels = input.shape
        _, output_height, output_width, _ = gradient_from_next_layer.shape
        
        # Apply padding to input if needed (same as forward pass)
        if self.padding > 0:
            input_padded = np.pad(input,
                                ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                                mode='constant')
        else:
            input_padded = input
        
        # Initialise gradients
        gradient_to_previous_layer = np.zeros_like(input_padded)
        gradient_weights = np.zeros_like(self.weights)
        gradient_biases = np.zeros_like(self.biases)
        
        # Compute gradients - essentially "reverse" convolution
        for batch_idx in range(batch_size):
            for output_channel_idx in range(self.output_channels):
                for height_idx in range(output_height):
                    for width_idx in range(output_width):
                        height_start = height_idx * self.stride
                        width_start = width_idx * self.stride
                        
                        # Gradient w.r.t weights: correlate input patches with output gradients
                        input_patch = input_padded[batch_idx, 
                                                 height_start:height_start+self.kernel_size, 
                                                 width_start:width_start+self.kernel_size, :]
                        gradient_weights[output_channel_idx] += gradient_from_next_layer[batch_idx, height_idx, width_idx, output_channel_idx] * input_patch.transpose(2, 0, 1)
                        
                        # Gradient w.r.t input: each position influenced by multiple output positions
                        gradient_to_previous_layer[batch_idx, 
                                                 height_start:height_start+self.kernel_size, 
                                                 width_start:width_start+self.kernel_size, :] += \
                            gradient_from_next_layer[batch_idx, height_idx, width_idx, output_channel_idx] * self.weights[output_channel_idx].transpose(1, 2, 0)
                
                # Gradient w.r.t biases: sum over all spatial positions
                gradient_biases[output_channel_idx] = np.sum(gradient_from_next_layer[:, :, :, output_channel_idx])
        
        # Remove padding from gradient_to_previous_layer if padding was used
        if self.padding > 0:
            gradient_to_previous_layer = gradient_to_previous_layer[:, self.padding:-self.padding, self.padding:-self.padding, :]
        
        # Update parameters using gradient descent
        self.weights -= self.learning_rate * gradient_weights
        self.biases -= self.learning_rate * gradient_biases
        
        return gradient_to_previous_layer

        
class Dense(Layer):
    
    def __init__(self, input_units, output_units, learning_rate):     # Input_units: number of features in input data - i.e 28x28 -> 784
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
        
    def backward(self, input, gradient_from_next_layer):     # gradient_from_next_layer is gradient of the loss with respect to the layer's output. Passed from the next layer in the network
        
        gradient_to_previous_layer = np.dot(gradient_from_next_layer, self.weights.T)      # Find out how we contributed to gradient_from_next_layer error, multiply gradient_from_next_layer by transpose of weights matrix. Transpose ensures matrix dimensions align 
       
        gradient_weights = np.dot(input.T, gradient_from_next_layer)      # Gradient w.r.t weights - finds out how much each weight contributed to overall loss
                                                                          # Transpose here 'pairs' each input feature with corresponding output gradient across entire batch
                                                      
        gradient_biases = gradient_from_next_layer.sum(axis = 0)      # gradient_from_next_layer is a 2d array representing the gradient of loss function w.r.t layers output
                                                                      # i.e how much final loss would change if output neuron changed. Shape is [batch_size, number_of_neurons]
                                                                      # .sum(axis = 0) sums elements of this array along axis=0 ie down the rows. Whole operation finds total influence of bias on batch's loss
                                                
        # Update parameters using gradient descent
        self.weights = self.weights - self.learning_rate * gradient_weights     # Updates weights, learning rate ensures step is small and controlled
        self.biases = self.biases - self.learning_rate * gradient_biases        # Updates biases, ""                                                  ""
        
        return gradient_to_previous_layer   # Becomes gradient_from_next_layer for previous layer
        
        
class Loss:
    
    @staticmethod
    def softmax_crossentropy_with_logits(raw_output_scores, reference_answers):        # raw_output_scores are the raw scores, array of shape [batch_size, num_classes], each row is set of scores for one input sample, each column a class 
        
        scores_for_correct_answers = raw_output_scores[np.arange(len(raw_output_scores)), reference_answers]      # np.arange(len(raw_output_scores)) creates array of row indices - scores for that image effectively, reference_answers provides column indices ie 0-9
        
        cross_entropy_loss = -scores_for_correct_answers + np.log(np.sum(np.exp(raw_output_scores), axis = 1))     # Essentially does softmax of cross entropy but allows for very large numbers that would break softmax.
                                                                                                                   # Cross entropy loss is a measure of how different the model's predicted probabilities are from actual label 
                                                                                            
        return cross_entropy_loss
       
    @staticmethod
    def grad_softmax_crossentropy_with_logits(raw_output_scores, reference_answers):     # Gradient of xentropy loss w.r.t raw output (raw_output_scores) - starting point for backpropagation
            
        ones_for_correct_answers = np.zeros_like(raw_output_scores)    # Creates an array with same shape as raw_output_scores but all initialised to 0
            
        ones_for_correct_answers[np.arange(len(raw_output_scores)), reference_answers] = 1     # For each row (sample image), places a 1 in the column corresponding to correct class index - creates a 100% probability truth
            
        softmax_probabilities = np.exp(raw_output_scores) / np.exp(raw_output_scores).sum(axis = 1, keepdims = True)   # Creates predicted probabilities, makes all values between 0 and 1. keepdims keeps dimensions of matrices
            
        return (-ones_for_correct_answers + softmax_probabilities) / raw_output_scores.shape[0]    # (-ones_for_correct_answers + softmax_probabilities) is predicted - actual, / raw_output_scores.shape[0] divides by batch size to find avg gradient across the batch  
    
            
class NeuralNetwork:        # Capture network's state (layers) and behaviour (training, predicting) into single object
    
    def __init__(self, layers):     # layers = list of layer objects, when NN instance is created it is passed a list of layer objects
        self.layers = layers
        
    def forward(self, inputs):     # inputs is a numpy array [batch_size, ...] - can be 2D for Dense or 4D for Conv
        
        activations = []    # Empty list to store output from each layer
        
        current_input = inputs
        
        for layer in self.layers:   # Loops through each layer in self.layers, returns activations list with full history of outputs
            output = layer.forward(current_input)   # Performs layer specific computation
            activations.append(output)
            
            current_input = output      # output becomes input for next layer in loop
            
        return activations
        
    def predict(self, inputs):     # Used to get the final answers from neural network after it has been trained
        
        raw_output_scores = self.forward(inputs)[-1]   # Gets list of activations (self.forward(inputs)) taking the last element of that list
        
        return raw_output_scores.argmax(axis=1)   # Finds index of the highest score for each input sample
    
    def train_batch(self, inputs, targets):     # Performs a single forward and backward pass for one batch of data to update network weights. 
                                                # inputs: shape can vary - (batch_size, 784) for dense or (batch_size, 28, 28, 1) for conv
                                                # targets: shape (batch_size,) - true labels [3, 7, 2, 9, ...]
            
            # Forward pass
            activations = self.forward(inputs)                  # Propagates input through the network, collecting outputs from all layers - returns a list of outputs
            layer_inputs = [inputs] + activations                # Concatenates the initial input with layer activations to get a list of inputs for each layer, for backpropagation, each layer needs to know what its input was to calculate gradients. 
            raw_output_scores = activations[-1]                  # The raw output scores from the final layer, before any activation function 
            
            # Calculate loss and its initial gradient
            loss = Loss.softmax_crossentropy_with_logits(raw_output_scores, targets)           # Computes the loss for each sample in the batch
            loss_gradient = Loss.grad_softmax_crossentropy_with_logits(raw_output_scores, targets) # Computes the gradient of the loss w.r.t the raw_output_scores, the starting point for backpropagation
            
            # Backward pass
            for layer_index in range(len(self.layers))[::-1]:     # Iterates backward through the layers, from the output layer to the input layer. [::-1] reverses range
                layer = self.layers[layer_index]                  # Selects the current layer to process
                layer_input = layer_inputs[layer_index]           # Retrieves the input that was fed to this layer during the forward pass
                
                loss_gradient = layer.backward(layer_input, loss_gradient) # Computes parameter gradients for the current layer and passes the gradient back to the previous layer
            
            return np.mean(loss)     # Returns the average loss value across all samples in the batch
    
class Netmain:
    #==============================================================================
    #                            Data Loading
    #==============================================================================
    def load_mnist_data_for_dense_network(self):     # Loads the dataset from keras, preprocesses it for dense networks, and splits it into training, validation, and test sets
        
        (training_images, training_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()     # Loads the raw dataset, which is split into training and testing sets
        
        # Normalise pixel values to be between 0 and 1
        training_images = training_images.astype(float) / 255.     # Converts integer pixel values (0-255) to float and scales them down
        test_images = test_images.astype(float) / 255.
        
        # Carve out a validation set from the training data
        validation_size = 10000
        validation_images = training_images[-validation_size:]                # Takes the last 10,000 images for validation
        validation_labels = training_labels[-validation_size:]
        training_images = training_images[:-validation_size]                  # The rest of the images remain as the training set
        training_labels = training_labels[:-validation_size]
        
        # Flatten images into one-dimensional vectors
        training_images = training_images.reshape([training_images.shape[0], -1])     # Reshapes the 28x28 images into 784-element vectors for the dense layers
        validation_images = validation_images.reshape([validation_images.shape[0], -1])
        test_images = test_images.reshape([test_images.shape[0], -1])
        
        return training_images, training_labels, validation_images, validation_labels, test_images, test_labels
    
    
    def load_mnist_data_for_conv_network(self):     # Loads the dataset from keras, preprocesses it for convolutional networks, and splits it into training, validation, and test sets
        
        (training_images, training_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()     # Loads the raw dataset, which is split into training and testing sets
        
        # Normalise pixel values to be between 0 and 1
        training_images = training_images.astype(float) / 255.     # Converts integer pixel values (0-255) to float and scales them down
        test_images = test_images.astype(float) / 255.
        
        # Carve out a validation set from the training data
        validation_size = 5000
        validation_images = training_images[-validation_size:]                # Takes the last 10,000 images for validation
        validation_labels = training_labels[-validation_size:]
        training_images = training_images[:5000]                  # The rest of the images remain as the training set
        training_labels = training_labels[:5000]
        
        # Add channel dimension for convolutional networks
        # MNIST is greyscale, so add a single channel dimension
        training_images = training_images.reshape([training_images.shape[0], 28, 28, 1])    # Shape becomes [batch, height, width, channels]
        validation_images = validation_images.reshape([validation_images.shape[0], 28, 28, 1])
        test_images = test_images.reshape([test_images.shape[0], 28, 28, 1])
        
        return training_images, training_labels, validation_images, validation_labels, test_images, test_labels
    
    
    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False): # A generator function that yields batches of data
        assert len(inputs) == len(targets)
        indices = np.arange(len(inputs))       # Creates an array of indices for the dataset
        if shuffle:
            np.random.shuffle(indices)         # Shuffles the indices to randomise the data order, important for training
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            batch_indices = indices[start_idx:start_idx + batchsize] # Selects a slice of indices for the current minibatch
            yield inputs[batch_indices], targets[batch_indices]      # Yields the data and labels for the current batch
    
    
    #==============================================================================
    #                        Training and Testing
    #==============================================================================
    def train_network(self, network, training_images, training_labels, validation_images, validation_labels, epochs=25, batch_size=32):     # Manages the overall training process over multiple epochs
        
        train_log = []     # List to store training accuracy for each epoch
        val_log = []       # List to store validation accuracy for each epoch
        
        for epoch in range(epochs): # An epoch is one full pass through the entire training dataset
            
            for batch_images, batch_labels in self.iterate_minibatches(training_images, training_labels, batch_size, shuffle=True):    # batch_images: array of training images, batch_labels: their labels
                network.train_batch(batch_images, batch_labels)     # Performs a training step (forward/backward pass) on the minibatch
            
            # Calculate and record accuracy at the end of each epoch
            train_accuracy = np.mean(network.predict(training_images) == training_labels)     # Computes accuracy on the full training set
            val_accuracy = np.mean(network.predict(validation_images) == validation_labels)   # Computes accuracy on the validation set to monitor for overfitting
            
            train_log.append(train_accuracy)
            val_log.append(val_accuracy)
            
            print(f"Epoch {epoch}: train_acc = {train_accuracy:.3f}, val_acc = {val_accuracy:.3f}")
        
        return train_log, val_log
    
    
    def test_network(self, network, test_images, test_labels, show_example=True):     # Evaluates the final performance of the network on the unseen test set
        
        predictions = network.predict(test_images)     # Gets the model's predictions for the entire test set
        accuracy = np.mean(predictions == test_labels) # Calculates the final test accuracy
        
        print(f"\nTest accuracy: {accuracy:.4f}")
        
        np.random.seed(None)
        random_index = np.random.randint(0, len(test_images))     # Selects a random index from the test set to visualise
        
        # Handle different input shapes for display
        if len(test_images.shape) == 2:  # Flattened data for dense networks
            display_image = test_images[random_index].reshape(28, 28)         # Reshapes the flat vector back into a 28x28 image for display
        else:  # Conv data [batch, height, width, channels]
            display_image = test_images[random_index, :, :, 0]               # Extract the single channel for greyscale display
            
        true_label = test_labels[random_index]
        
        # Get the network's raw output for the selected image
        raw_output_scores = network.forward(test_images[random_index:random_index+1])[-1][0]     # Slicing with [random_index:random_index+1] keeps the dimensions, [-1][0] extracts the final layer's output
        predicted_label = raw_output_scores.argmax()                                             # The predicted class is the index of the highest score
        
        plt.figure(figsize=(6, 3))
        
        plt.subplot(1, 2, 1)
        plt.imshow(display_image, cmap='grey')
        plt.title(f"True: {true_label}, Predicted: {predicted_label}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.bar(range(10), raw_output_scores)
        plt.xlabel('Digit')
        plt.ylabel('Score')
        plt.title('Network Output Scores')
        
        plt.tight_layout()
        
        print(f"Raw output scores: {raw_output_scores}")     # Prints the raw output scores for each class, rounded for readability
        print(f"True label: {true_label}, Predicted: {predicted_label}")
    
        return accuracy
        
    # --- Main Execution ---
    def run(self):
        # Define different network layer configurations in a dictionary for easy experimentation
        architectures = {
            "Agrawal": [        # Original fully connected architecture
                Dense(784, 100, 0.1),    # ie accept an input vector of size 784 (28x28) and output a vector of size 100. self.weights is array of (784,100), self.biases is (100,)
                ReLU(),
                Dense(100, 200, 0.2),
                ReLU(),
                Dense(200, 10, 0.1)
            ],
            
            "JoflNetV1": [      # Deeper fully connected architecture
                Dense(784, 200, 0.1),
                ReLU(),
                Dense(200, 400, 0.2),
                ReLU(),
                Dense(400, 100, 0.1),
                ReLU(),
                Dense(100, 10, 0.2)
            ],
            
            "ConvNet": [        
                Conv2D(1, 4, kernel_size=3, stride=2, padding=1, learning_rate=0.1),     # 4 filters of size 3x3, maintains spatial size with padding
                ReLU(),
                Flatten(),      # Convert from 4D to 2D for Dense layers 
                Dense(784, 128, 0.1),   # 784 = 14*14*4 (the flattened size from previous conv layer)
                ReLU(),
                Dense(128, 10, 0.1)      
            ],
        }
        
        # Select which architecture to use
        selected_architecture = "ConvNet"
        
        # Check if we need convolutional data format by looking at the first layer - returns True if first element is conv2d
        uses_convolution = isinstance(architectures[selected_architecture][0], Conv2D)
        
        # Load data in appropriate format
        if uses_convolution:
            training_images, training_labels, validation_images, validation_labels, test_images, test_labels = self.load_mnist_data_for_conv_network()
        else:
            training_images, training_labels, validation_images, validation_labels, test_images, test_labels = self.load_mnist_data_for_dense_network()
        
        print(f"Data loaded: {training_images.shape[0]} training, {validation_images.shape[0]} validation, {test_images.shape[0]} test samples")
        print(f"Data shape: {training_images.shape}")
        
        # Create, train, and test the network
        print(f"\nTraining {selected_architecture} network")
        model = NeuralNetwork(architectures[selected_architecture])   # Instantiates the network with the chosen set of layers, ie creates neural network object and self.layers is list containing x layer objects
        train_log, val_log = self.train_network(model, training_images, training_labels, validation_images, validation_labels, epochs=10)    # Begins training
        accuracy = self.test_network(model, test_images, test_labels)
        
        # Plot training and validation accuracy over epochs to visualise performance
        plt.figure(figsize=(6, 4))
        plt.plot(train_log, label='Training')
        plt.plot(val_log, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Training Progress - {selected_architecture} network')
        plt.legend()
        plt.grid(True)
        plt.show()
        
n = Netmain()
n.run()
