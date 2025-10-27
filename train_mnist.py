import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Use the same SimpleCNN definition from your Backend.py, but adjusted for MNIST
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2) # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2) # 14x14 -> 7x7

        self.flatten_size = 32 * 7 * 7 # 1568
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_and_save_model():
    print("--- Starting MNIST Model Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard MNIST transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Mean and std dev for MNIST
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = SimpleCNN(in_channels=1).to(device) # MNIST is grayscale (1 channel)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- INCREASED EPOCHS ---
    epochs = 15 # Train for more epochs for potentially better accuracy
    print(f"Training for {epochs} epochs...")

    model.train() # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Iterate over batches of data
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device) # Move data to the selected device
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item() * inputs.size(0) # Accumulate loss scaled by batch size
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Print progress every 100 batches
            if (i + 1) % 100 == 0:
                batch_loss = loss.item()
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {batch_loss:.4f}")

        # Calculate epoch statistics
        epoch_loss = running_loss / total_samples
        epoch_accuracy = 100.0 * correct_predictions / total_samples
        print(f"--- Epoch {epoch+1}/{epochs} Complete --- Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}% ---")


    print("--- Training Complete ---")

    # Save the model's state dictionary (the learned weights)
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Model saved successfully as 'mnist_model.pth'")

# Run the training function if the script is executed directly
if __name__ == '__main__':
    train_and_save_model()