import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Build the Conv1D Architecture
model = models.Sequential([
    # Input shape: (Time_Steps / Sequence_Length, Features_Per_Step)
    # Example: A sequence of 10 time steps, each with 1 sensor feature
    layers.Input(shape=(10, 1)),
    
    # Conv1D: 4 filters, kernel_size 3 (sliding window of 3 time steps)
    # Output shape: (10 - 3 + 1) = (8, 4)
    layers.Conv1D(filters=4, kernel_size=3, activation='relu'),
    
    # MaxPool1D: pooling window of 2
    # Output shape: 8 / 2 = (4, 4)
    layers.MaxPooling1D(pool_size=2),
    
    # Flatten: 4 time steps * 4 filters = 16 features
    layers.Flatten(),
    
    # Dense Classification Head: 2 output classes (logits)
    layers.Dense(units=2)
])

# 2. Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Display architecture and dimensions
model.summary()

# 3. Create Synthetic Sequential Data
# 6 sequence samples of length 10, with 1 feature per step
X_train = tf.random.normal(shape=(6, 10, 1))
y_train = tf.constant([0, 1, 0, 1, 1, 0])

# 4. Train the Model (5 Epochs)
print("\nStarting Training:")
history = model.fit(X_train, y_train, epochs=5, batch_size=2, verbose=1)