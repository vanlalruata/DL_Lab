import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# 1. Train a basic model on CIFAR-10 (RGB images) for 1 epoch
(x_train, y_train), _ = datasets.cifar10.load_data()
x_train = x_train[:5000] / 255.0  # Subset for fast classroom execution
y_train = y_train[:5000]

model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_layer_1'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=0)

# 2. Extract Learned Weights from First Conv Layer
weights, _ = model.get_layer('conv_layer_1').get_weights()  # Shape: (3, 3, 3, 16)

# 3. Visualize the 16 Learned Filters (Averaged across RGB channels)
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
axes = axes.flatten()

for i in range(16):
    filter_kernel = weights[:, :, :, i].mean(axis=-1)  # Mean across color channels
    axes[i].imshow(filter_kernel, cmap='viridis')
    axes[i].set_title(f"Filter {i+1}")
    axes[i].axis('off')

plt.suptitle("16 Learned 3x3 Kernels After Backpropagation", fontsize=14)
plt.tight_layout()
plt.show()