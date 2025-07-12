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
        # Set device: torch.device object 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Data preprocessing: torchvision.transforms.Compose object 
        transform = transforms.Compose([
            transforms.ToTensor(),                          # PIL Image -> torch.Tensor (float32, [0,1] range)
            transforms.Normalize((0.1307,), (0.3081,))      # Normalise using MNIST dataset statistics (tuple of floats)
        ])
        
        # Load full training dataset: torchvision.datasets.MNIST object 
        full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        
        # Split training data: torch.utils.data.random_split returns tuple of Subset objects
        train_size = 50000  # int: number of training samples
        val_size = 10000    # int: number of validation samples
        self.train_data, self.val_data = torch.utils.data.random_split(
            full_train, [train_size, val_size]
        )
        
        # Test dataset: torchvision.datasets.MNIST object 
        self.test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        # Network architectures: dict containing nn.Sequential objects 
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
            
            # Convolutional network: Conv2d layers followed by fully connected
            "ConvNet": nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=1),       # Input: 1 channel, Output: 64 feature maps (torch.Tensor [B,64,H,W])
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=5, stride=2, padding=1),      # 64 -> 32 channels, stride 2 reduces spatial dimensions : [(W−K+2P)/S]+1, uses floor function
                nn.ReLU(),
                nn.Flatten(),                    # 3D feature maps -> 1D vector for fully connected layers
                nn.Linear(12*12*32, 256),        # Flattened conv output (4608 values) -> 256 features
                nn.ReLU(),
                nn.Linear(256, 100),             # 256 -> 100 fully connected
                nn.ReLU(),
                nn.Linear(100, 10)               # Final classification layer: 100 -> 10 class scores
            )
        }
    
    def train_network(self, model, epochs=10, batch_size=128, lr=0.001):
        # Move model parameters to GPU/CPU: returns same nn.Sequential object on specified device
        model = model.to(self.device)
        
        # Data loaders: torch.utils.data.DataLoader objects (iterate over batches)
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)      # Shuffled training batches
        val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)        # Fixed validation order
        
        # Training components: optimiser and loss function 
        optimizer = optim.Adam(model.parameters(), lr=lr)    # torch.optim.Adam object for gradient updates
        criterion = nn.CrossEntropyLoss()                    # Loss function: expects logits and class indices
        
        # Training logs: lists to store accuracy values (floats) per epoch
        train_log = []
        val_log = []
        
        model.train()  # Set model to training mode (enables dropout, batch norm updates)
        for epoch in range(epochs):
            # Training loop: iterate through batches of (images, labels) tuples
            for images, labels in train_loader:
                # Move batch data to device: torch.Tensor objects [batch_size, channels, height, width] and [batch_size]
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()            # Clear gradients from previous iteration
                outputs = model.forward(images)          # Forward pass: torch.Tensor [batch_size, 10] logits
                loss = criterion(outputs, labels)  # Compute loss: scalar torch.Tensor
                loss.backward()                  # Backward pass: compute gradients
                optimizer.step()                 # Update model parameters using gradients
            
            # Calculate epoch accuracies: float values between 0 and 1
            train_acc = self.evaluate(model, train_loader)
            val_acc = self.evaluate(model, val_loader)
            
            # Store accuracy history: append float values to lists
            train_log.append(train_acc)
            val_log.append(val_acc)
            
            print(f"Epoch {epoch}: train_acc = {train_acc:.3f}, val_acc = {val_acc:.3f}")
        
        # Return training history: tuple of lists containing float accuracy values
        return train_log, val_log
    
    def evaluate(self, model, loader):
        model.eval()  # Set to evaluation mode (disables dropout, batch norm in eval mode)
        correct = 0   # int: count of correct predictions
        total = 0     # int: total number of samples
        
        # Evaluation without gradient computation (saves memory and computation)
        with torch.no_grad():
            for images, labels in loader:  # Iterate through batches: torch.Tensor objects
                # Move batch to device: same shapes as training
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)                    # Model predictions: torch.Tensor [batch_size, 10]
                _, predicted = torch.max(outputs, 1)      # Get class with highest score: torch.Tensor [batch_size]
                total += labels.size(0)                    # Update sample count: int
                correct += (predicted == labels).sum().item()  # Count correct predictions: int
        
        model.train()  # Return to training mode
        return correct / total  # Return accuracy as float between 0 and 1
    
    def test_network(self, model, show_example=True):
        # Test data loader: DataLoader object for final evaluation
        test_loader = DataLoader(self.test_data, batch_size=256, shuffle=False)
        
        model.eval()  # Set to evaluation mode
        test_acc = self.evaluate(model, test_loader)  # Get test accuracy: float
        print(f"\nTest accuracy: {test_acc:.4f}")
        
        if show_example:
            # Select random test sample: int index and tuple of (torch.Tensor, int)
            idx = np.random.randint(0, len(self.test_data))
            image, true_label = self.test_data[idx]  # image: torch.Tensor [1,28,28], label: int
            
            # Generate prediction for single sample
            with torch.no_grad():
                # Prepare for model: add batch dimension and move to device
                image_gpu = image.unsqueeze(0).to(self.device)      # torch.Tensor [1,1,28,28]
                output = model(image_gpu)                           # Model output: torch.Tensor [1,10]
                raw_scores = output[0].cpu().numpy()                # Convert to numpy array: shape [10]
                predicted_label = output.argmax(1).item()          # Get predicted class: int
            
            plt.figure(figsize=(6, 3))
            
            # Display original image: 2D numpy array from torch.Tensor
            plt.subplot(1, 2, 1)
            plt.imshow(image.squeeze(), cmap='gray')  # Remove channel dimension for display
            plt.title(f"True: {true_label}, Predicted: {predicted_label}")
            plt.axis('off')
            
            # Display prediction scores: bar chart of 10 class probabilities
            plt.subplot(1, 2, 2)
            plt.bar(range(10), raw_scores)  # x: list of ints [0-9], y: numpy array of floats
            plt.xlabel('Digit')
            plt.ylabel('Score')
            plt.title('Network Output Scores')
            
            plt.tight_layout()
            plt.show()
            
            print(f"Raw output scores: {raw_scores}")      # numpy array [10] of float logits
            print(f"True label: {true_label}, Predicted: {predicted_label}")  # int values
        
        return test_acc  # Return final test accuracy: float
    
    def run(self):
        selected_architecture = "ConvNet"  
        
        print(f"Data loaded: {len(self.train_data)} training, {len(self.val_data)} validation, {len(self.test_data)} test samples")
        print(f"Data shape: {self.train_data[0][0].shape}")  # torch.Size object showing tensor dimensions
        
        # Create and train selected network: nn.Sequential object
        print(f"\nTraining {selected_architecture} network")
        model = self.architectures[selected_architecture]
        
        # Training process: returns tuple of lists containing float accuracy values
        train_log, val_log = self.train_network(model, epochs=10, batch_size=128, lr=0.001)
        accuracy = self.test_network(model)  # Final test accuracy: float
        

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