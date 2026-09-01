import numpy as np

# 1. Forward Pass Operations

def conv2d(image, kernel, bias):
    """
    Applies a 2D convolution.
    image shape:  (H, W)
    kernel shape: (kH, kW)
    """
    H, W = image.shape
    kH, kW = kernel.shape
    out_h = H - kH + 1
    out_w = W - kW + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            # Extract image patch matching the kernel size
            patch = image[i:i+kH, j:j+kW]
            # Element-wise multiply, sum, and add bias
            output[i, j] = np.sum(patch * kernel) + bias

    return output

def relu(x):
    """ReLU activation: max(0, x)"""
    return np.maximum(0, x)

def max_pool2d(feature_map, size=2, stride=2):
    """
    Applies 2D Max Pooling.
    feature_map shape: (H, W)
    """
    H, W = feature_map.shape
    out_h = (H - size) // stride + 1
    out_w = (W - size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride
            patch = feature_map[h_start:h_start+size, w_start:w_start+size]
            output[i, j] = np.max(patch)

    return output

def softmax(z):
    """Converts raw logits to probabilities."""
    exp_z = np.exp(z - np.max(z))  # Stable softmax
    return exp_z / np.sum(exp_z)

# 2. Setup Dummy Input and Weights

# Single grayscale image (4x4)
image = np.array([
    [1.0, 2.0, 0.0, 1.0],
    [0.0, 1.0, 3.0, 2.0],
    [2.0, 0.0, 1.0, 1.0],
    [1.0, 2.0, 0.0, 0.0]
])

# 2x2 Convolution Filter / Kernel and Bias
kernel = np.array([
    [1.0, 0.0],
    [0.0, -1.0]
])
conv_bias = 0.5

# Dense layer weights (maps 4 pooled features -> 2 classes)
dense_weights = np.random.randn(4, 2)
dense_bias = np.zeros(2)

# 3. Step-by-Step Forward Pass

# Step A: Convolution -> (4x4) with (2x2) kernel = (3x3) feature map
conv_out = conv2d(image, kernel, conv_bias)

# Step B: ReLU Activation
relu_out = relu(conv_out)

# Step C: Max Pooling (2x2, stride 1) -> (3x3) becomes (2x2)
pooled_out = max_pool2d(relu_out, size=2, stride=1)

# Step D: Flatten (2x2) matrix to 1D vector (4 elements)
flattened = pooled_out.flatten()

# Step E: Fully Connected / Dense Layer
logits = np.dot(flattened, dense_weights) + dense_bias
probabilities = softmax(logits)

# 4. Display intermediate outputs
print("Input Image (4x4):\n", image)
print("\n1. Conv Output (3x3):\n", np.round(conv_out, 2))
print("\n2. ReLU Output (3x3):\n", np.round(relu_out, 2))
print("\n3. MaxPool Output (2x2):\n", np.round(pooled_out, 2))
print("\n4. Flattened Vector (4,):\n", np.round(flattened, 2))
print("\n5. Class Probabilities (2 classes):\n", np.round(probabilities, 4))