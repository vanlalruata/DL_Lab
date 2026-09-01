import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Build the Conv3D Architecture
model = models.Sequential([
    # Input shape: (Depth/Frames, Height, Width, Channels)
    # Example: A video clip of 8 frames, 16x16 resolution, 1 grayscale channel
    layers.Input(shape=(8, 16, 16, 1)),
    
    # Conv3D: 4 filters, 3x3x3 kernel (valid padding)
    # Output shape: (8-3+1, 16-3+1, 16-3+1, 4) = (6, 14, 14, 4)
    layers.Conv3D(filters=4, kernel_size=(3, 3, 3), activation='relu'),
    
    # MaxPool3D: 2x2x2 pooling window
    # Output shape: (6/2, 14/2, 14/2, 4) = (3, 7, 7, 4)
    layers.MaxPooling3D(pool_size=(2, 2, 2)),
    
    # Flatten: 3 * 7 * 7 * 4 = 588 features
    layers.Flatten(),
    
    # Dense Classification Head: 2 output classes (e.g., Action A vs Action B)
    layers.Dense(units=2)
])

# 2. Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Display architecture and parameter breakdown
model.summary()

# 3. Create Synthetic 3D Volumetric Data
# 4 video samples of shape (8 frames, 16 height, 16 width, 1 channel)
X_train = tf.random.normal(shape=(4, 8, 16, 16, 1))
y_train = tf.constant([0, 1, 1, 0])

# 4. Train the Model (5 Epochs)
print("\nStarting Training:")
history = model.fit(X_train, y_train, epochs=5, batch_size=2, verbose=1)