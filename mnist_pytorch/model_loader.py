import sys
import torch
from torchvision import datasets, transforms
import numpy as np 

model_path = 'MNIST_model.pt'
directory = './data'
mnist_mean = 0.1307
mnist_std = 0.3081

def load_model(model_path):
    model = torch.jit.load(model_path)
    model.eval()
    print('model loaded')
    return model
    
def get_test_image():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mnist_mean,), (mnist_std,))
    ])
    
    test_dataset = datasets.MNIST(
        root = directory,
        train = False,
        download = True,
        transform = transform
    )
    
    idx = np.random.randint(0, len(test_dataset))
    image, label = test_dataset[idx]
    
    return image, label
    
def predict(model, image):
    input_tensor = image.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(1).item()
        
    return prediction
    
def main():
    model = load_model(model_path)
    
    image, true_label = get_test_image()
    
    predicted_label = predict(model, image)
    
    print(f"True number: {true_label}")
    print(f"Model number: {predicted_label}")
    
    