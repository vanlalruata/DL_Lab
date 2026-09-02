import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# 1. Load and Preprocess a Real Image
url = "https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as r:
    from io import BytesIO
    img = Image.open(BytesIO(r.read())).convert('L').resize((200, 200))

img_array = np.array(img, dtype=np.float32) / 255.0  # Shape: (200, 200)
input_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # (1, 1, 200, 200)

# 2. Define Two Classic Handcrafted 3x3 Kernels
sobel_horizontal = np.array([
    [-1.0, -2.0, -1.0],
    [ 0.0,  0.0,  0.0],
    [ 1.0,  2.0,  1.0]
])

sharpen_filter = np.array([
    [ 0.0, -1.0,  0.0],
    [-1.0,  5.0, -1.0],
    [ 0.0, -1.0,  0.0]
])

# Reshape kernels to PyTorch Conv2d format: (out_channels, in_channels, kH, kW)
custom_weights = np.stack([sobel_horizontal, sharpen_filter], axis=0)  # (2, 3, 3)
custom_weights = custom_weights[:, None, :, :]  # (2, 1, 3, 3)
custom_bias = np.array([0.0, 0.0])

custom_weights_t = torch.from_numpy(custom_weights).float()
custom_bias_t = torch.from_numpy(custom_bias).float()

# 3. Create Conv2D Layer and Set Custom Filter Weights
conv_layer = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1, bias=True)
with torch.no_grad():
    conv_layer.weight.copy_(custom_weights_t)
    conv_layer.bias.copy_(custom_bias_t)

# 4. Forward Pass: Apply Convolution and ReLU
feature_maps = conv_layer(input_tensor)
activated_maps = F.relu(feature_maps)

# 5. Plot the Original vs. Extracted Feature Maps
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(input_tensor[0, 0].numpy(), cmap='gray')
axes[0].set_title("Original Input (200x200)")
axes[0].axis('off')

axes[1].imshow(activated_maps[0, 0].detach().numpy(), cmap='gray')
axes[1].set_title("Filter 1: Horizontal Edges")
axes[1].axis('off')

axes[2].imshow(activated_maps[0, 1].detach().numpy(), cmap='gray')
axes[2].set_title("Filter 2: Sharpened Features")
axes[2].axis('off')

plt.tight_layout()
plt.show()