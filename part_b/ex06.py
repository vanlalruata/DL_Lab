import os
import zipfile
import urllib.request
import tensorflow as tf

# Download Microsoft's Cats vs Dogs archive with custom User-Agent
url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
zip_path = "cats_and_dogs.zip"

if not os.path.exists(zip_path):
    print("Downloading dataset...")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("cats_and_dogs_data")
    print("Extraction complete.")


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# 1. Load CIFAR-10
(x_train_all, y_train_all), (x_test_all, y_test_all) = datasets.cifar10.load_data()

# CIFAR-10 class indices: 3 = Cat, 5 = Dog
CAT_CLASS = 3
DOG_CLASS = 5

# 2. Filter for only Cats and Dogs
train_mask = np.isin(y_train_all, [CAT_CLASS, DOG_CLASS]).flatten()
test_mask = np.isin(y_test_all, [CAT_CLASS, DOG_CLASS]).flatten()

x_train = x_train_all[train_mask]
y_train = y_train_all[train_mask]

x_test = x_test_all[test_mask]
y_test = y_test_all[test_mask]

# Convert labels: Cat -> 0, Dog -> 1
y_train = (y_train == DOG_CLASS).astype(np.float32)
y_test = (y_test == DOG_CLASS).astype(np.float32)

print(f"Train samples: {x_train.shape[0]} images, shape: {x_train.shape[1:]}")
print(f"Test samples:  {x_test.shape[0]} images, shape: {x_test.shape[1:]}")

# 3. Build Binary Classification CNN
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Rescaling(1./255),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1)  # Logits: <=0 Cat, >0 Dog
])

# 4. Compile and Train
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,
    batch_size=64
)

# 5. Visualize Sample Predictions
class_names = ['Cat', 'Dog']
logits = model.predict(x_test[:4])
probabilities = tf.nn.sigmoid(logits).numpy().flatten()
predictions = (probabilities > 0.5).astype(int)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    axes[i].imshow(x_test[i])
    true_label = class_names[int(y_test[i][0])]
    pred_label = class_names[predictions[i]]
    confidence = probabilities[i] if predictions[i] == 1 else (1 - probabilities[i])
    
    axes[i].set_title(f"True: {true_label}\nPred: {pred_label} ({confidence*100:.1f}%)")
    axes[i].axis('off')

plt.tight_layout()
plt.show()