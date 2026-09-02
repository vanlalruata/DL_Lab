import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Build the CNN Model Architecture
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv Layer: 4 filters, 3x3 kernel (valid padding -> output: 6x6x4)
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        # MaxPool Layer: 2x2 pool window (output: 3x3x4)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        # Fully Connected Output Layer: 2 classes (logits)
        self.fc = nn.Linear(3 * 3 * 4, 2)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = CNN()

# 2. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Display architecture and parameter counts
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# 3. Create Synthetic Dummy Data
# 5 images of shape (1, 8, 8) and 5 integer class labels (0 or 1)
torch.manual_seed(0)
X_train = torch.randn(5, 1, 8, 8)
y_train = torch.tensor([0, 1, 1, 0, 1])

# 4. Train the Model (5 Epochs)
print("\nStarting Training:")
for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()
    preds = logits.argmax(dim=1)
    acc = (preds == y_train).float().mean().item()
    print(f"Epoch {epoch+1}/5 - loss: {loss.item():.4f} - acc: {acc:.4f}")