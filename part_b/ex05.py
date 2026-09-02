import os
import urllib.request
import zipfile
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Hyperparameters
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5

# 2. Download and Extract Dataset Zip
dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_path = "cats_and_dogs_filtered.zip"
data_root = "cats_and_dogs_filtered"

if not os.path.isdir(data_root):
    print("Downloading dataset...")
    urllib.request.urlretrieve(dataset_url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    os.remove(zip_path)

train_dir = os.path.join(data_root, 'train')
val_dir = os.path.join(data_root, 'validation')

# 3. Create DataLoaders using ImageFolder
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_set = datasets.ImageFolder(train_dir, transform=train_transform)
val_set = datasets.ImageFolder(val_dir, transform=val_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 4. Build CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = CNN()

# 5. Compile and Train
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct, n = 0.0, 0, 0
    for imgs, labels in train_loader:
        labels = labels.float().unsqueeze(1)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        train_correct += ((logits > 0).float() == labels).sum().item()
        n += imgs.size(0)

    model.eval()
    val_loss, val_correct, vn = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            labels = labels.float().unsqueeze(1)
            logits = model(imgs)
            loss = criterion(logits, labels)
            val_loss += loss.item() * imgs.size(0)
            val_correct += ((logits > 0).float() == labels).sum().item()
            vn += imgs.size(0)

    print(f"Epoch {epoch+1}/{EPOCHS} - "
          f"loss: {train_loss/n:.4f} - acc: {train_correct/n:.4f} - "
          f"val_loss: {val_loss/vn:.4f} - val_acc: {val_correct/vn:.4f}")

# 6. Predict and Visualize
class_names = ['Cat', 'Dog']

model.eval()
for images, labels in val_loader:
    with torch.no_grad():
        logits = model(images)
        probabilities = torch.sigmoid(logits).flatten().numpy()
        predictions = (probabilities > 0.5).astype(int)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for i in range(4):
        img_np = images[i].numpy().transpose(1, 2, 0)
        axes[i].imshow(img_np)
        true_label = class_names[int(labels[i].item())]
        pred_label = class_names[predictions[i]]
        confidence = probabilities[i] if predictions[i] == 1 else (1 - probabilities[i])

        axes[i].set_title(f"True: {true_label}\nPred: {pred_label} ({confidence*100:.1f}%)")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    break