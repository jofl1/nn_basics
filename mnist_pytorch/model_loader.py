import sys
import torch
from torchvision import datasets, transforms
import numpy as np 
import matplotlib.pyplot as plt

model_path = 'MNIST_model.pt'
directory = './data'
mnist_mean = 0.1307
mnist_std = 0.3081

def load_model(model_path):
    model = torch.jit.load(model_path)
    model.eval()
    
    device = torch.device('cuda')
    
    model = model.to(device)
    
    print('model loaded')
    return model, device
    
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
    
    print('got test image')
    return image, label
    
def predict(model, image, device):
    input_tensor = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(1).item()
        
    return prediction
    
def main():
    model, device = load_model(model_path)
    
    image, true_label = get_test_image()
    
    predicted_label = predict(model, image, device)
    
    print(f"True number: {true_label}")
    print(f"Model number: {predicted_label}")
    
    plt.imshow(image.permute(1, 2, 0))
    plt.show()