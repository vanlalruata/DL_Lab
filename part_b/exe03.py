import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Hyperparameters & Device Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Data Preparation (MNIST Dataset)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST mean & std
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. CNN Architecture Definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Feature Extraction Block
        # Input: [Batch, 1, 28, 28] -> Output: [Batch, 16, 28, 28] -> MaxPool: [Batch, 16, 14, 14]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Input: [Batch, 16, 14, 14] -> Output: [Batch, 32, 14, 14] -> MaxPool: [Batch, 32, 7, 7]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Classification Head (Fully Connected Layers)
        # Flattened size: 32 channels * 7 height * 7 width = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)

    def forward(self, x):
        # First Conv Block
        x = self.pool1(self.relu1(self.conv1(x)))
        
        # Second Conv Block
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Flatten for Dense Layers
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)  # Raw logits output (CrossEntropyLoss applies Softmax internally)
        return x

# 4. Model Initialization, Loss Function & Optimizer
model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. Training Loop
print(f"Training on: {DEVICE}")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Step A: Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Step B: Backward Pass & Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# 6. Evaluation / Testing Loop
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nFinal Test Accuracy: {accuracy:.2f}%")