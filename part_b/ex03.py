import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load and Preprocess a Real Image
# (Using a sample image built directly into Keras/TensorFlow for instant execution)
image_path = tf.keras.utils.get_file(
    'cat.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg'
)

# Load image in grayscale and resize to 200x200
img = tf.keras.utils.load_img(image_path, color_mode='grayscale', target_size=(200, 200))
img_array = tf.keras.utils.img_to_array(img)  # Shape: (200, 200, 1)

# Normalize pixel values to [0, 1] and add batch dimension -> (1, 200, 200, 1)
input_tensor = np.expand_dims(img_array / 255.0, axis=0)

# 2. Define Two Classic Handcrafted 3x3 Kernels
# Kernel 1: Sobel Horizontal Edge Detector
sobel_horizontal = np.array([
    [-1.0, -2.0, -1.0],
    [ 0.0,  0.0,  0.0],
    [ 1.0,  2.0,  1.0]
])

# Kernel 2: Sharpen Filter
sharpen_filter = np.array([
    [ 0.0, -1.0,  0.0],
    [-1.0,  5.0, -1.0],
    [ 0.0, -1.0,  0.0]
])

# Reshape kernels to TensorFlow format: (kernel_H, kernel_W, in_channels, out_channels)
# We stack both into a single tensor -> (3, 3, 1, 2)
custom_weights = np.stack([sobel_horizontal, sharpen_filter], axis=-1)
custom_weights = np.expand_dims(custom_weights, axis=2)
custom_bias = np.array([0.0, 0.0])  # Zero bias

# 3. Create Conv2D Layer and Set Custom Filter Weights
conv_layer = layers.Conv2D(
    filters=2,
    kernel_size=(3, 3),
    padding='same',
    use_bias=True
)

# Build layer and manually assign handcrafted weights
conv_layer.build(input_shape=(None, 200, 200, 1))
conv_layer.set_weights([custom_weights, custom_bias])

# 4. Forward Pass: Apply Convolution and ReLU
feature_maps = conv_layer(input_tensor)
activated_maps = layers.ReLU()(feature_maps)

# 5. Plot the Original vs. Extracted Feature Maps
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original Grayscale Image
axes[0].imshow(input_tensor[0, :, :, 0], cmap='gray')
axes[0].set_title("Original Input (200x200)")
axes[0].axis('off')

# Filter 1: Edge Detection Feature Map
axes[1].imshow(activated_maps[0, :, :, 0], cmap='gray')
axes[1].set_title("Filter 1: Horizontal Edges")
axes[1].axis('off')

# Filter 2: Sharpened Feature Map
axes[2].imshow(activated_maps[0, :, :, 1], cmap='gray')
axes[2].set_title("Filter 2: Sharpened Features")
axes[2].axis('off')

plt.tight_layout()
plt.show()