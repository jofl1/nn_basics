# MNIST Digit Classification with a Dense Neural Network

## Overview

This project provides a from-scratch implementation of a dense neural network in Python using only the NumPy library. The network is designed and trained to classify handwritten digits from the MNIST dataset. The primary goal of this project is to demonstrate the fundamental concepts of neural networks, including forward and backward propagation, layer-based architecture, and loss calculation, without relying on high-level deep learning frameworks.

## How It Works

The implementation is structured in an object-oriented manner, with distinct classes for different components of the neural network.

### Core Components

*   **`Layer`**: A base class that defines the interface for all network layers. It establishes the requirement for `forward()` and `backward()` methods, ensuring a consistent structure.

*   **`ReLU`**: A non-linear activation layer that applies the Rectified Linear Unit function, `f(x) = max(0, x)`. This introduces non-linearity into the network, allowing it to learn more complex patterns.

*   **`Dense`**: A fully connected layer that performs a linear transformation on its input using a weights matrix and a bias vector. It is the primary building block of this network. The weights are initialised using Xavier initialisation to maintain a healthy variance in activations throughout the network.

*   **`Loss`**: A utility class containing static methods for calculating the loss and its gradient. It uses `softmax_crossentropy_with_logits` for both the loss calculation and its gradient, which is a standard approach for multi-class classification problems.

*   **`NeuralNetwork`**: This class encapsulates the entire network. It manages the sequence of layers, orchestrates the forward and backward passes, and provides methods for training (`train_batch`), prediction (`predict`), and evaluation.

### Data Handling and Training

1.  **Data Loading**: The `load_mnist_data()` function fetches the MNIST dataset using Keras, normalises the pixel values to a range of 0 to 1, and splits the data into training, validation, and test sets. The images are also flattened from 28x28 pixels to 784-element vectors to be fed into the dense layers.

2.  **Minibatch Iteration**: The `iterate_minibatches()` function is a generator that yields batches of data and their corresponding labels. This is crucial for stochastic gradient descent, as it allows the network to update its weights based on a small subset of the data at a time.

3.  **Training Loop**: The `train_network()` function orchestrates the training process over a specified number of epochs. In each epoch, it iterates through the minibatches, performs a forward and backward pass for each, and updates the network's weights. At the end of each epoch, it calculates and logs the training and validation accuracy to monitor performance and prevent overfitting.

4.  **Testing**: The `test_network()` function evaluates the final performance of the trained network on the unseen test set. It also provides a visual example of a prediction, showing the input image, the true label, the predicted label, and a bar chart of the network's output logits.

## Usage

To run the script, simply execute it from your terminal:

```bash
python mnist_dense.py
```

The script will automatically load the data, train the network, and then test it. The training progress, including the accuracy on the training and validation sets, will be printed to the console for each epoch. Finally, the test accuracy will be displayed, along with a plot showing a random test image and the network's prediction.

## Architectures

The script defines two network architectures, "Agrawal" and "Josh", which can be easily selected by changing the `selected_architecture` variable. This allows for experimentation with different network depths and layer sizes.

### Agrawal
*   Dense(784, 100) -> ReLU
*   Dense(100, 200) -> ReLU
*   Dense(200, 10)

### Josh
*   Dense(784, 200) -> ReLU
*   Dense(200, 400) -> ReLU
*   Dense(400, 100) -> ReLU
*   Dense(100, 10)

## Dependencies

*   [NumPy](https://numpy.org/)
*   [Keras](https://keras.io/) (for dataset loading)
*   [Matplotlib](https://matplotlib.org/)
