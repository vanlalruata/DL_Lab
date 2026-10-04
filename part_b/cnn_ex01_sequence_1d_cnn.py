import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Build the Conv1D Architecture
class Conv1DNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv1D: 4 filters, kernel_size 3 (sliding window of 3 time steps)
        # Input: (batch, 1, 10) -> Output: (batch, 4, 8)
        self.conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3)
        # MaxPool1D: pooling window of 2 -> Output: (batch, 4, 4)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        # Dense Classification Head: 2 output classes (logits)
        self.fc = nn.Linear(4 * 4, 2)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = Conv1DNet()

# 2. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Display architecture and dimensions
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# 3. Create Synthetic Sequential Data
# 6 sequence samples of length 10, with 1 feature per step
torch.manual_seed(0)
X_train = torch.randn(6, 1, 10)
y_train = torch.tensor([0, 1, 0, 1, 1, 0])

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