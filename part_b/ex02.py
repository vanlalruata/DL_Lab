import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Build the Conv3D Architecture
class Conv3DNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv3D: 4 filters, 3x3x3 kernel (valid padding)
        # Input: (batch, 1, 8, 16, 16) -> Output: (batch, 4, 6, 14, 14)
        self.conv = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3)
        # MaxPool3D: 2x2x2 pooling window
        # Output: (batch, 4, 3, 7, 7)
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.flatten = nn.Flatten()
        # Dense Classification Head: 2 output classes
        self.fc = nn.Linear(3 * 7 * 7 * 4, 2)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = Conv3DNet()

# 2. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Display architecture and parameter breakdown
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# 3. Create Synthetic 3D Volumetric Data
# 4 video samples of shape (8 frames, 16 height, 16 width, 1 channel)
torch.manual_seed(0)
X_train = torch.randn(4, 1, 8, 16, 16)
y_train = torch.tensor([0, 1, 1, 0])

# 4. Train the Model (5 Epochs)
print("\nStarting Training:")
for epoch in range(5):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0.0
    correct = 0
    for i in range(0, X_train.size(0), 2):
        idx = permutation[i:i+2]
        batch_x, batch_y = X_train[idx], y_train[idx]
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)
        correct += (logits.argmax(dim=1) == batch_y).sum().item()
    n = X_train.size(0)
    print(f"Epoch {epoch+1}/5 - loss: {epoch_loss/n:.4f} - acc: {correct/n:.4f}")