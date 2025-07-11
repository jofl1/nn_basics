import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

class MNISTClassifier:
    def __init__(self):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load MNIST data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        # Load full training data
        full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        
        # Split into train and validation
        train_size = 50000
        val_size = 10000
        self.train_data, self.val_data = torch.utils.data.random_split(
            full_train, [train_size, val_size]
        )
        
        # Load test data
        self.test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        # Define architectures using PyTorch layers
        self.architectures = {
            "JoflNetV1": nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 200),
                nn.ReLU(),
                nn.Linear(200, 400),
                nn.ReLU(),
                nn.Linear(400, 100),
                nn.ReLU(),
                nn.Linear(100, 10)
            ),
            
            "ConvNet": nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=5, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(12*12*32, 256),
                nn.ReLU(),
                nn.Linear(256, 100),
                nn.ReLU(),
                nn.Linear(100, 10)
            )
        }
    
    def train_network(self, model, epochs=10, batch_size=128, lr=0.001):
        # Move model to device
        model = model.to(self.device)
        
        # Create data loaders
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        train_log = []
        val_log = []
        
        model.train()
        for epoch in range(epochs):
            # Training
            for images, labels in train_loader:
                # Move data to device
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Calculate accuracies
            train_acc = self.evaluate(model, train_loader)
            val_acc = self.evaluate(model, val_loader)
            
            train_log.append(train_acc)
            val_log.append(val_acc)
            
            print(f"Epoch {epoch}: train_acc = {train_acc:.3f}, val_acc = {val_acc:.3f}")
        
        return train_log, val_log
    
    def evaluate(self, model, loader):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in loader:
                # Move data to device
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        model.train()
        return correct / total
    
    def test_network(self, model, show_example=True):
        test_loader = DataLoader(self.test_data, batch_size=256, shuffle=False)
        
        model.eval()
        test_acc = self.evaluate(model, test_loader)
        print(f"\nTest accuracy: {test_acc:.4f}")
        
        if show_example:
            # Get a random test sample
            idx = np.random.randint(0, len(self.test_data))
            image, true_label = self.test_data[idx]
            
            # Get prediction
            with torch.no_grad():
                # Move to device and add batch dimension
                image_gpu = image.unsqueeze(0).to(self.device)
                output = model(image_gpu)
                raw_scores = output[0].cpu().numpy()
                predicted_label = output.argmax(1).item()
            
            # Visualize
            plt.figure(figsize=(6, 3))
            
            plt.subplot(1, 2, 1)
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title(f"True: {true_label}, Predicted: {predicted_label}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.bar(range(10), raw_scores)
            plt.xlabel('Digit')
            plt.ylabel('Score')
            plt.title('Network Output Scores')
            
            plt.tight_layout()
            plt.show()
            
            print(f"Raw output scores: {raw_scores}")
            print(f"True label: {true_label}, Predicted: {predicted_label}")
        
        return test_acc
    
    def run(self):
        # Select architecture
        selected_architecture = "ConvNet"
        
        print(f"Data loaded: {len(self.train_data)} training, {len(self.val_data)} validation, {len(self.test_data)} test samples")
        print(f"Data shape: {self.train_data[0][0].shape}")
        
        # Create and train network
        print(f"\nTraining {selected_architecture} network")
        model = self.architectures[selected_architecture]
        
        train_log, val_log = self.train_network(model, epochs=10, batch_size=128, lr=0.001)
        accuracy = self.test_network(model)
        
        # Plot training progress
        plt.figure(figsize=(6, 4))
        plt.plot(train_log, label='Training')
        plt.plot(val_log, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Training Progress - {selected_architecture} network')
        plt.legend()
        plt.grid(True)
        plt.show()


classifier = MNISTClassifier()
classifier.run()