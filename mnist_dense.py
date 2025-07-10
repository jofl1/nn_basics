import keras 
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class Layer:      # Dummy class, allows to do .forward() and .backward()
    
    def forward(self, input):
        raise NotImplementedError

    def backward(self, input, grad_output):
        raise NotImplementedError   # Finds the dot product of the incoming gradient from the next layer(grad_output) with the local gradient of this layer(d_layer_d_input)
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
        
    def predict(self, inputs):     # Used to get the final answers from neural network after it has been trained
        
        logits = self.forward(inputs)[-1]   # Gets list of activations (self.forward(inuts)) taking the last element of that list
        
        return logits.argmax(axis=-1)   # Finds index of the highest score for each input sample
    
    def train_batch(self, inputs, targets):     # Performs a single forward and backward pass for one batch of data to update network weights. 
                                                # inputs: shape (batch_size , 784) - batch_size flattened images
                                                # targets: shape (batch_size,) - true labels [3, 7, 2, 9, ...]
            
            # Forward pass
            activations = self.forward(inputs)                  # Propagates input through the network, collecting outputs from all layers - returns a list of outputs
            layer_inputs = [inputs] + activations             # Concatenates the initial input with layer activations to get a list of inputs for each layer, for backpropagation, each layer needs to know what its input was to calculate gradients. 
            logits = activations[-1]                          # The raw output scores from the final layer, before any activation function 
            
            # Calculate loss and its initial gradient
            loss = Loss.softmax_crossentropy_with_logits(logits, targets)           # Computes the loss for each sample in the batch
            loss_grad = Loss.grad_softmax_crossentropy_with_logits(logits, targets) # Computes the gradient of the loss w.r.t the logits, the starting point for backpropagation
            
            # Backward pass
            for layer_index in range(len(self.layers))[::-1]:     # Iterates backward through the layers, from the output layer to the input layer.[::1] reverses range
                layer = self.layers[layer_index]                  # Selects the current layer to process
                layer_input = layer_inputs[layer_index]           # Retrieves the input that was fed to this layer during the forward pass
                
                loss_grad = layer.backward(layer_input, loss_grad) # Computes parameter gradients for the current layer and passes the gradient back to the previous layer
            
            return np.mean(loss)     # Returns the average loss value across all samples in the batch
    
    
#==============================================================================
#                            Data Loading
#==============================================================================
def load_mnist_data():     # Loads the dataset from keras, preprocesses it, and splits it into training, validation, and test sets
    
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()     # Loads the raw dataset, which is split into training and testing sets
    
    # Normalise pixel values to be between 0 and 1
    X_train = X_train.astype(float) / 255.     # Converts integer pixel values (0-255) to float and scales them down
    X_test = X_test.astype(float) / 255.
    
    # Carve out a validation set from the training data
    val_size = 10000
    X_val = X_train[-val_size:]                # Takes the last 10,000 images for validation
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]              # The rest of the images remain as the training set
    y_train = y_train[:-val_size]
    
    # Flatten images into one-dimensional vectors
    X_train = X_train.reshape([X_train.shape[0], -1])     # Reshapes the 28x28 images into 784-element vectors for the dense layers
    X_val = X_val.reshape([X_val.shape[0], -1])
    X_test = X_test.reshape([X_test.shape[0], -1])
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def iterate_minibatches(inputs, targets, batchsize, shuffle=False): # A generator function that yields batches of data
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))       # Creates an array of indices for the dataset
    if shuffle:
        np.random.shuffle(indices)         # Shuffles the indices to randomize the data order, important for training
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize] # Selects a slice of indices for the current minibatch
        yield inputs[excerpt], targets[excerpt]            # Yields the data and labels for the current batch


#==============================================================================
#                        Training and Testing
#==============================================================================
def train_network(network, X_train, y_train, X_val, y_val, epochs=25, batch_size=32):     # Manages the overall training process over multiple epochs
    
    train_log = []     # List to store training accuracy for each epoch
    val_log = []       # List to store validation accuracy for each epoch
    
    for epoch in range(epochs): # An epoch is one full pass through the entire training dataset
        
        for x_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):    # x_batch: array shape (batch_size, 784) - random training images, y_batch: array shape (batch_size,) - their labels
            network.train_batch(x_batch, y_batch)     # Performs a training step (forward/backward pass) on the minibatch
        
        # Calculate and record accuracy at the end of each epoch
        train_accuracy = np.mean(network.predict(X_train) == y_train)     # Computes accuracy on the full training set
        val_accuracy = np.mean(network.predict(X_val) == y_val)           # Computes accuracy on the validation set to monitor for overfitting
        
        train_log.append(train_accuracy)
        val_log.append(val_accuracy)
        
        print(f"Epoch {epoch}: train_acc = {train_accuracy:.3f}, val_acc = {val_accuracy:.3f}")
    
    return train_log, val_log


def test_network(network, X_test, y_test, show_example=True):     # Evaluates the final performance of the network on the unseen test set
    
    predictions = network.predict(X_test)     # Gets the model's predictions for the entire test set
    accuracy = np.mean(predictions == y_test) # Calculates the final test accuracy
    
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    np.random.seed(None)
    idx = np.random.randint(0, len(X_test))     # Selects a random index from the test set to visualise
    image = X_test[idx].reshape(28, 28)         # Reshapes the flat vector back into a 28x28 image for display
    true_label = y_test[idx]
    
    # Get the network's raw output (logits) for the selected image
    logits = network.forward(X_test[idx:idx+1])[-1][0]     # Slicing with [idx:idx+1] keeps the dimensions, [-1][0] extracts the final layer's output
    predicted_label = logits.argmax()                     # The predicted class is the index of the highest logit score.
    
    plt.figure(figsize=(6, 3))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='grey')
    plt.title(f"True: {true_label}, Predicted: {predicted_label}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(10), logits)
    plt.xlabel('Digit')
    plt.ylabel('Logit')
    plt.title('Network Output')
    
    plt.tight_layout()
    
    print(f"Logits: {logits}")     # Prints the raw output scores for each class, rounded for readability
    print(f"True label: {true_label}, Predicted: {predicted_label}")

    return accuracy
    
# --- Main Execution ---

# Load data once
X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()
print(f"Data loaded: {X_train.shape[0]} training, {X_val.shape[0]} validation, {X_test.shape[0]} test samples")

# Define different network layer configurations in a dictionary for easy experimentation
architectures = {
    "Agrawal": [
        Dense(784, 100, 0.1),    # ie accept an input vector of size 784 (28x28) and output a vector of size 100. self.weights is array of (784,100), self.biases is (100,0)
        ReLU(),
        Dense(100, 200, 0.2),
        ReLU(),
        Dense(200, 10, 0.1)
    ],
    
    "Josh": [
        Dense(784, 200, 0.1),
        ReLU(),
        Dense(200, 400, 0.2),
        ReLU(),
        Dense(400, 100, 0.1),
        ReLU(),
        Dense(100, 10, 0.2)
    ]
}

selected_architecture = "Josh"

# Create, train, and test the network
print(f"\nTraining {selected_architecture} network")
model = NeuralNetwork(architectures[selected_architecture])   # Instantiates the network with the chosen set of layers, ie creates neural network object and self.layer is list containing x layer objects
train_log, val_log = train_network(model, X_train, y_train, X_val, y_val, epochs=10)    # Begins training
accuracy = test_network(model, X_test, y_test)

# Plot training and validation accuracy over epochs to visualise performance.
plt.figure(figsize=(6, 4))
plt.plot(train_log, label='Training')
plt.plot(val_log, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Training Progress - {selected_architecture} network')
plt.legend()
plt.grid(True)
plt.show()