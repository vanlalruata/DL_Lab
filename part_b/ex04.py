import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# 1. Train a basic model on CIFAR-10 (RGB images) for 1 epoch
transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

x_train = torch.stack([train_set[i][0] for i in range(5000)])  # (5000, 3, 32, 32)
y_train = torch.tensor([train_set[i][1] for i in range(5000)], dtype=torch.long)

# 2. Build simple CNN
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(16 * 16 * 16, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

model.train()
optimizer.zero_grad()
logits = model(x_train)
loss = criterion(logits, y_train)
loss.backward()
optimizer.step()

# 2. Extract Learned Weights from First Conv Layer
weights = model[0].weight.detach().numpy()  # Shape: (16, 3, 3, 3)

# 3. Visualize the 16 Learned Filters (Averaged across RGB channels)
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
axes = axes.flatten()

for i in range(16):
    filter_kernel = weights[i, :, :, :].mean(axis=0)  # Mean across input channels
    axes[i].imshow(filter_kernel, cmap='viridis')
    axes[i].set_title(f"Filter {i+1}")
    axes[i].axis('off')

plt.suptitle("16 Learned 3x3 Kernels After Backpropagation", fontsize=14)
plt.tight_layout()
plt.show()