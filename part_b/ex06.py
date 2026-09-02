import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# 1. Load CIFAR-10
transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

x_train_all = torch.stack([train_set[i][0] for i in range(len(train_set))])
y_train_all = torch.tensor([train_set[i][1] for i in range(len(train_set))], dtype=torch.long)

x_test_all = torch.stack([test_set[i][0] for i in range(len(test_set))])
y_test_all = torch.tensor([test_set[i][1] for i in range(len(test_set))], dtype=torch.long)

# CIFAR-10 class indices: 3 = Cat, 5 = Dog
CAT_CLASS = 3
DOG_CLASS = 5

# 2. Filter for only Cats and Dogs
train_mask = (y_train_all == CAT_CLASS) | (y_train_all == DOG_CLASS)
test_mask = (y_test_all == CAT_CLASS) | (y_test_all == DOG_CLASS)

x_train = x_train_all[train_mask]
y_train = y_train_all[train_mask]

x_test = x_test_all[test_mask]
y_test = y_test_all[test_mask]

# Convert labels: Cat -> 0, Dog -> 1
y_train = (y_train == DOG_CLASS).float()
y_test = (y_test == DOG_CLASS).float()

print(f"Train samples: {x_train.shape[0]} images, shape: {tuple(x_train.shape[1:])}")
print(f"Test samples:  {x_test.shape[0]} images, shape: {tuple(x_test.shape[1:])}")

# 3. Build Binary Classification CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = CNN()

# 4. Compile and Train
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

BATCH_SIZE = 64
for epoch in range(5):
    model.train()
    perm = torch.randperm(x_train.size(0))
    train_loss, train_correct, n = 0.0, 0, 0
    for i in range(0, x_train.size(0), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        batch_x = x_train[idx]
        batch_y = y_train[idx].unsqueeze(1)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
        train_correct += ((logits > 0).float() == batch_y).sum().item()
        n += batch_x.size(0)

    model.eval()
    with torch.no_grad():
        test_logits = model(x_test)
        test_loss = criterion(test_logits, y_test.unsqueeze(1)).item()
        test_correct = ((test_logits > 0).float() == y_test.unsqueeze(1)).sum().item()

    print(f"Epoch {epoch+1}/5 - "
          f"loss: {train_loss/n:.4f} - acc: {train_correct/n:.4f} - "
          f"val_loss: {test_loss:.4f} - val_acc: {test_correct/x_test.size(0):.4f}")

# 5. Visualize Sample Predictions
class_names = ['Cat', 'Dog']
model.eval()
with torch.no_grad():
    sample_logits = model(x_test[:4])
    probabilities = torch.sigmoid(sample_logits).flatten().numpy()
    predictions = (probabilities > 0.5).astype(int)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    img_np = x_test[i].numpy().transpose(1, 2, 0)
    axes[i].imshow(img_np)
    true_label = class_names[int(y_test[i].item())]
    pred_label = class_names[predictions[i]]
    confidence = probabilities[i] if predictions[i] == 1 else (1 - probabilities[i])

    axes[i].set_title(f"True: {true_label}\nPred: {pred_label} ({confidence*100:.1f}%)")
    axes[i].axis('off')

plt.tight_layout()
plt.show()