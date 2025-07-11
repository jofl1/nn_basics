import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

mnist_data = datasets.MNIST(
    root = '~/src/nn_basics/mnist_pytorch',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)