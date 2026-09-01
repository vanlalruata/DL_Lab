import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Hyperparameters
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5

# 2. Download and Extract Dataset Zip directly via Keras
dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
data_dir = tf.keras.utils.get_file('cats_and_dogs.zip', origin=dataset_url, extract=True)
data_dir = tf.io.gfile.join(tf.io.gfile.dirname(data_dir), 'cats_and_dogs_filtered')

train_dir = tf.io.gfile.join(data_dir, 'train')
val_dir = tf.io.gfile.join(data_dir, 'validation')

# 3. Create tf.data Pipelines using image_dataset_from_directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False
)

# Optimize performance with prefetching
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# 4. Build CNN Model
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Rescaling(1./255),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1)  # Logits: <=0 Cat, >0 Dog
])

# 5. Compile and Train
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# 6. Predict and Visualize
class_names = ['Cat', 'Dog']

for images, labels in val_ds.take(1):
    logits = model(images)
    probabilities = tf.nn.sigmoid(logits).numpy().flatten()
    predictions = (probabilities > 0.5).astype(int)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for i in range(4):
        axes[i].imshow(images[i].numpy().astype("uint8"))
        true_label = class_names[int(labels[i].numpy()[0])]
        pred_label = class_names[predictions[i]]
        confidence = probabilities[i] if predictions[i] == 1 else (1 - probabilities[i])
        
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label} ({confidence*100:.1f}%)")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()