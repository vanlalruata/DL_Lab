import torch
import torch.nn as nn
import torch.optim as optim

# 1. Minimal CNN Architecture (1 Conv Layer + 1 Linear Layer)
class MinimalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv2d: 1 input channel -> 4 filters, 3x3 kernel (no padding)
        # Input: (1, 8, 8) -> Output: (4, 6, 6)
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        self.relu = nn.ReLU()
        # MaxPool2d: 2x2 window -> Output: (4, 3, 3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Linear: 4 channels * 3 * 3 = 36 flattened features -> 2 classes
        self.fc = nn.Linear(4 * 3 * 3, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))  # Shape: (Batch, 4, 3, 3)
        x = x.view(x.size(0), -1)               # Flatten to (Batch, 36)
        return self.fc(x)                       # Output logits: (Batch, 2)

# 2. Dummy Data: 5 grayscale images (1x8x8) and 5 binary labels (0 or 1)
X = torch.randn(5, 1, 8, 8)
y = torch.tensor([0, 1, 1, 0, 1])

# 3. Model, Loss, Optimizer
model = MinimalCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Training Loop (5 Steps)
for step in range(5):
    optimizer.zero_grad()               # Reset gradients
    predictions = model(X)              # Forward pass
    loss = criterion(predictions, y)    # Compute loss
    loss.backward()                     # Backprop
    optimizer.step()                    # Update weights
    
    print(f"Step {step+1}, Loss: {loss.item():.4f}")