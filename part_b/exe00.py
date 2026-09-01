import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Build the CNN Model Architecture
model = models.Sequential([
    # Input layer: 8x8 grayscale image (1 channel)
    layers.Input(shape=(8, 8, 1)),
    
    # Conv Layer: 4 filters, 3x3 kernel (valid padding -> output: 6x6x4)
    layers.Conv2D(filters=4, kernel_size=(3, 3), activation='relu'),
    
    # MaxPool Layer: 2x2 pool window (output: 3x3x4)
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten: 3 * 3 * 4 = 36 values
    layers.Flatten(),
    
    # Fully Connected Output Layer: 2 classes (logits)
    layers.Dense(units=2)
])

# 2. Compile Model with Loss and Optimizer
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Display architecture and parameter counts
model.summary()

# 3. Create Synthetic Dummy Data
# 5 images of shape (8, 8, 1) and 5 integer class labels (0 or 1)
X_train = tf.random.normal(shape=(5, 8, 8, 1))
y_train = tf.constant([0, 1, 1, 0, 1])

# 4. Train the Model (5 Epochs)
print("\nStarting Training:")
history = model.fit(X_train, y_train, epochs=5, batch_size=5, verbose=1)