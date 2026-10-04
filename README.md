# Deep Learning Lab — Practical & Question Bank

**Programme:** Master of Computer Applications (MCA)

**Institution:** Mizoram University, Aizawl, Mizoram, India

**Course:** Deep Learning Lab (Practical Work / Lab Records)

This repository contains the lab exercises and practical records for the MCA Deep
Learning Lab at Mizoram University. It includes the syllabus-mapped practicals
(`practical_01_mp_neuron.py` ... `practical_30_early_stopping_lr_schedule.py`)
covering foundational computational units, perceptrons, activations, losses,
MLPs, optimizers, and deep neural networks in NumPy and PyTorch; an extended question
bank of 50 items covering CNNs (AlexNet, VGG, GoogLeNet, ResNet, DenseNet), GNNs,
RNN/LSTM/GRU, and GANs; plus additional hands-on exercises in dataset analysis,
network security, and NLP / Transformer / LLM topics (including Mizo-language
translation and POS tagging).

---

## Part A — Module Practicals (from Lab Unit 1 & Advanced DNN)

### Module 1: Foundational Computational Units & Perceptrons (NumPy)
- **Practical 1 — McCulloch-Pitts Neuron from Scratch** (`part_a/practical_01_mp_neuron.py`)  
  Implement an M-P neuron class in NumPy to simulate AND, OR, NOT, and NOR. Show via
  output truth tables why a single-layer M-P thresholding fails on the non-linearly
  separable XOR gate.
- **Practical 2 — Single-Layer Perceptron Learning Algorithm (PLA)** (`part_a/practical_02_perceptron_pla.py`)  
  Code the Rosenblatt Perceptron with step activation. Generate a linearly separable
  2D dataset with `sklearn.datasets.make_blobs`, train the perceptron, and plot the
  evolving decision boundary at each epoch until convergence.
- **Practical 3 — Demonstrating the Linear Separability Constraint** (`part_a/practical_03_linear_separability.py`)  
  Test the Perceptron on a non-linearly separable dataset (XOR / concentric circles).
  Visualize how it fails to converge, plotting perpetual oscillations in classification
  loss and boundary updates.

### Module 2: Activation Functions & Visualizations (NumPy & Matplotlib)
- **Practical 4 — Activation Function Zoo and Gradient Visualizer** (`part_a/practical_04_activation_functions.py`)  
  Implement forward and derivative of Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, and SELU.
  Plot a 2×3 grid comparing each function with its first derivative over x ∈ [-5, 5].
- **Practical 5 — The Vanishing Gradient Simulation in Deep Feedforward Networks** (`part_a/practical_05_vanishing_gradient.py`)  
  Build an N-layer forward pass in pure NumPy. Back-propagate an initial gradient
  through 10 hidden layers using Sigmoid vs ReLU; plot gradient magnitude per layer.
- **Practical 6 — Softmax and Stable Numerical Implementations** (`part_a/practical_06_stable_softmax.py`)  
  Implement standard vs numerically stable Softmax (subtract max(z)). Pass extreme
  logits z = [1000, 1001, 1002] to show vanilla Softmax NaN overflow vs stable handling.

### Module 3: Loss Functions (NumPy & PyTorch)
- **Practical 7 — Regression Loss Functions (MSE, MAE, Huber)** (`part_a/practical_07_regression_losses.py`)  
  Implement MSE, MAE, and Huber (δ=1.0) in NumPy on a synthetic dataset with high
  magnitude outliers; plot and compare outlier sensitivity.
- **Practical 8 — Binary Cross-Entropy vs MSE for Binary Classification** (`part_a/practical_08_bce_vs_mse.py`)  
  Build a single Sigmoid neuron; compare loss-surface convexities of MSE vs BCE over
  ranges of weight and bias values.
- **Practical 9 — Multi-Class CCE & Softmax Coupling** (`part_a/practical_09_cce_softmax.py`)  
  Implement CCE from scratch in NumPy; compare performance and memory of One-Hot
  (Categorical CE) vs Integer (Sparse Categorical CE) targets.
- **Practical 10 — Advanced Loss Functions in PyTorch (Focal Loss)** (`part_a/practical_10_focal_loss.py`)  
  Implement Focal Loss as a custom `torch.nn.Module`; train on a 95/5 imbalanced
  dataset and compare accuracy/recall with `nn.BCEWithLogitsLoss`.

### Module 4: Multi-Layer Perceptrons & Backpropagation from Scratch (NumPy)
- **Practical 11 — 2-Layer MLP Backpropagation from Scratch (Solving XOR)** (`part_a/practical_11_mlp_backprop_xor.py`)  
  Build a 2-2-1 MLP with Sigmoid activations; derive and code manual backprop
  (∂L/∂W1, ∂L/∂b1, ∂L/∂W2, ∂L/∂b2) to learn XOR.
- **Practical 12 — General N-Layer MLP Engine in NumPy** (`part_a/practical_12_mlp_engine_iris.py`)  
  Build a flexible class accepting arbitrary topologies (e.g. [4,16,8,3]); implement
  automated forward/backward and mini-batch updates to classify the Iris dataset.

### Module 5: Optimizers from Scratch (NumPy)
- **Practical 13 — Gradient Descent Variants Comparison** (`part_a/practical_13_gradient_descent_variants.py`)  
  Implement Batch GD, SGD, and Mini-Batch GD on linear regression; plot loss vs CPU
  time and weight trajectories on a 2D contour map.
- **Practical 14 — Momentum & Nesterov Accelerated Gradient (NAG)** (`part_a/practical_14_momentum_nag.py`)  
  Implement Polyak Momentum and NAG; optimize a 2D pathological ravine (Rosenbrock)
  and visualize damped transverse oscillations.
- **Practical 15 — Adaptive Learning Rates (AdaGrad vs RMSProp)** (`part_a/practical_15_adagrad_rmsprop.py`)  
  Implement AdaGrad and RMSProp; show AdaGrad stalls on sparse gradients while RMSProp
  continues to learn.
- **Practical 16 — Complete Implementation of the Adam Optimizer** (`part_a/practical_16_adam_optimizer.py`)  
  Code Adam (m_t, v_t, bias corrections m̂_t, v̂_t); plot weight trajectories and the
  smoothing impact of bias correction over the first 10 steps.
- **Practical 17 — Modern Optimizer Variants (AdamW & AdaDelta)** (`part_a/practical_17_adamw_adadelta.py`)  
  Implement AdamW (decoupled weight decay) and AdaDelta (learning-rate-free); compare
  weight-norm decay vs standard Adam + L2 over 100 epochs.

### Module 6: End-to-End Neural Networks in PyTorch
- **Practical 18 — End-to-End Classification Pipeline using torch.nn** (`part_a/practical_18_pytorch_mlp_mnist.py`)  
  Build a 3-hidden-layer MLP with `nn.Sequential`/`nn.Linear`/`nn.ReLU`/
  `nn.CrossEntropyLoss`; train on MNIST with a DataLoader.
- **Practical 19 — PyTorch Custom Optimizer & Loss Benchmarking** (`part_a/practical_19_optimizer_benchmark.py`)  
  Train identical MLPs on Fashion-MNIST with SGD, SGD(momentum=0.9), Adagrad, RMSprop,
  and AdamW; plot combined training-loss and validation-accuracy curves.
- **Practical 20 — Self-Normalizing Networks with SELU and AlphaDropout** (`part_a/practical_20_selu_alphadropout.py`)  
  Build a 10-layer feedforward net with `nn.SELU`/`nn.AlphaDropout`; record mean and
  variance of hidden activations to verify the self-normalizing property.

### Module 7: Simple DNN (Deep Neural Networks) — Hidden Layers

Introduces fully-connected networks with one or more hidden layers. The first
three practicals are written **entirely in NumPy** (no torch, no tensorflow) so
the forward and backward passes are explicit. The remaining practicals use
**PyTorch** to study how depth, activations, regularisation, normalisation and
initialisation affect a DNN.

#### From-scratch DNN (pure NumPy)
- **Practical 21 — 2-Layer DNN Forward Pass from Scratch (NumPy)** (`part_a/practical_21_dnn_forward_numpy.py`)  
  Build a 2-layer MLP (input → hidden → output) using only NumPy: weight init,
  ReLU + sigmoid activations, and visualise the *untrained* decision boundary.
- **Practical 22 — 2-Layer DNN with Manual Backpropagation (NumPy)** (`part_a/practical_22_dnn_backprop_numpy.py`)  
  Hand-code the full forward + backward pass (no autograd) for a 2-8-1 network
  with sigmoid + BCE; train on `make_moons` and visualise the learned boundary.
- **Practical 23 — 3-Layer DNN & Vanishing Gradients (NumPy)** (`part_a/practical_23_vanishing_grad_3layer.py`)  
  2-8-4-1 network; compare the gradient magnitudes at each layer for
  Sigmoid vs ReLU to show why ReLU mitigates vanishing gradients.

#### DNN in PyTorch
- **Practical 24 — 2-Layer DNN in PyTorch (autograd)** (`part_a/practical_24_dnn_pytorch_autograd.py`)  
  Re-implement Practical 22 using `nn.Module`, `nn.Linear`, `BCELoss` and
  `torch.optim.SGD`; confirm the same decision boundary is learned.
- **Practical 25 — Deeper DNN (4 Hidden Layers)** (`part_a/practical_25_deep_dnn_4hidden.py`)  
  A 2→32→16→8→4→1 MLP on `make_circles`; plot train/val loss & accuracy curves.
- **Practical 26 — Activation Function Comparison in a DNN** (`part_a/practical_26_activation_comparison_dnn.py`)  
  Train the same 3-hidden-layer MLP with Sigmoid, Tanh, ReLU, and LeakyReLU;
  compare final validation accuracy.
- **Practical 27 — DNN with Dropout Regularisation** (`part_a/practical_27_dropout_regularisation.py`)  
  Same MLP trained with p=0 vs p=0.5 dropout; visualise the train/val
  loss gap and show that dropout reduces overfitting.
- **Practical 28 — DNN with Batch Normalisation** (`part_a/practical_28_batch_normalisation.py`)  
  Compare training curves with and without `nn.BatchNorm1d`; BN converges
  faster and to a lower loss at a higher learning rate.
- **Practical 29 — DNN Weight Initialisation (Random vs Xavier vs He)** (`part_a/practical_29_weight_initialisation.py`)  
  4-hidden-layer ReLU MLP initialised three ways; show that Kaiming/He init
  gives the lowest training loss.
- **Practical 30 — DNN with Early Stopping & Learning-Rate Scheduling** (`part_a/practical_30_early_stopping_lr_schedule.py`)  
  `ReduceLROnPlateau` LR schedule plus early stopping that saves the best
  model state on validation loss; shows the saved best epoch and the LR decay
  curve.

```bash
python part_a/practical_21_dnn_forward_numpy.py      # numpy forward pass
python part_a/practical_22_dnn_backprop_numpy.py     # numpy backprop
python part_a/practical_24_dnn_pytorch_autograd.py   # same model in PyTorch
python part_a/practical_28_batch_normalisation.py    # BatchNorm comparison
```

---

## Part B — Extended Question Bank (50 questions)

### CNNs & Classic Architectures (1–20)
1. Implement a 2D convolution operation from scratch in NumPy (no `nn.Conv2d`). (`part_b/cnn_q01_conv2d_scratch.py`)
2. Explain and code the difference between "same", "valid", and "causal" padding. (`part_b/cnn_q02_padding_modes.py`)
3. Derive the output spatial size of a conv layer given kernel, stride, and padding. (`part_b/cnn_q03_output_size_formula.py`)
4. Implement max-pooling and average-pooling from scratch; discuss translation invariance. (`part_b/cnn_q04_pooling_scratch.py`)
5. Build AlexNet in PyTorch and discuss why ReLU + dropout were key to its success. (`part_b/cnn_q05_alexnet.py`)
6. Reproduce VGG-16 block design; analyze parameter count growth from stacked 3×3 kernels. (`part_b/cnn_q06_vgg16.py`)
7. Compare 7×7, 5×5, and 3×3 convolutions: receptive field vs parameter cost. (`part_b/cnn_q07_conv_size_comparison.py`)
8. Implement GoogLeNet/Inception module with parallel branches (1×1, 3×3, 5×5, pool). (`part_b/cnn_q08_inception_module.py`)
9. Explain the role of 1×1 convolutions (dimensionality reduction, channel mixing). (`part_b/cnn_q09_conv1x1.py`)
10. Implement batch normalization from scratch and integrate it into a CNN training loop. (`part_b/cnn_q10_batchnorm_scratch.py`)
11. Build ResNet basic-block and bottleneck-block; implement residual (skip) connections. (`part_b/cnn_q11_resnet_blocks.py`)
12. Explain why residual connections mitigate vanishing gradients in very deep networks. (`part_b/cnn_q12_skip_connections.py`)
13. Implement ResNeXt grouped convolutions and compare with standard ResNet. (`part_b/cnn_q13_resnext_grouped_conv.py`)
14. Build DenseNet dense blocks with concatenation; discuss feature reuse and parameters. (`part_b/cnn_q14_densenet_blocks.py`)
15. Implement a depthwise-separable convolution (MobileNet style) and count its savings. (`part_b/cnn_q15_depthwise_separable.py`)
16. Compare parameter counts and FLOPs of AlexNet, VGG-16, and ResNet-50. (`part_b/cnn_q16_param_flop_comparison.py`)
17. Implement global average pooling and explain its use as a regularizer in GoogLeNet. (`part_b/cnn_q17_global_avg_pooling.py`)
18. Build a U-Net style encoder-decoder with skip connections for image segmentation. (`part_b/cnn_q18_unet_segmentation.py`)
19. Implement transpose convolution (fractional strided conv) for upsampling. (`part_b/cnn_q19_transpose_conv.py`)
20. Visualize CNN filters and feature maps of a pretrained model (e.g. via hooks). (`part_b/cnn_q20_filter_visualisation.py`)

### RNN / LSTM / GRU (21–34)
21. Implement a simple RNN cell from scratch and train it on a sine-wave prediction task. (`part_b/rnn_q21_rnn_cell_sine.py`)
22. Derive the BPTT equations for a vanilla RNN and discuss the exploding gradient problem. (`part_b/rnn_q22_bptt_exploding_grad.py`)
23. Implement gradient clipping and demonstrate its effect on RNN training stability. (`part_b/rnn_q23_gradient_clipping.py`)
24. Build an LSTM cell from scratch; explain the roles of forget, input, and output gates. (`part_b/rnn_q24_lstm_cell_scratch.py`)
25. Implement a GRU cell and compare its gate structure to LSTM. (`part_b/rnn_q25_gru_cell_scratch.py`)
26. Compare LSTM vs GRU on a language-modeling or sequence task (perplexity/accuracy). (`part_b/rnn_q26_lstm_vs_gru.py`)
27. Train an LSTM for sentiment classification on a text dataset (e.g. IMDB). (`part_b/rnn_q27_lstm_sentiment_imdb.py`)
28. Implement sequence-to-sequence for character-level text generation. (`part_b/rnn_q28_seq2seq_char_gen.py`)
29. Build a bidirectional RNN/LSTM and explain when future context helps. (`part_b/rnn_q29_bidirectional_rnn.py`)
30. Implement an attention mechanism over LSTM hidden states (Bahdanau-style). (`part_b/rnn_q30_bahdanau_attention.py`)
31. Apply masking in padded sequence batches (`pack_padded_sequence`). (`part_b/rnn_q31_pack_padded_sequence.py`)
32. Visualize what an LSTM gate learns on a long-range dependency toy problem. (`part_b/rnn_q32_lstm_gate_visualisation.py`)
33. Compare teacher forcing vs scheduled sampling in sequence generation. (`part_b/rnn_q33_teacher_forcing.py`)
34. Implement a stacked (multi-layer) LSTM and analyze representational depth. (`part_b/rnn_q34_stacked_lstm.py`)

### RNN Architectures — Practical Demonstrations (51–59)
51. Demonstrate Many-to-Many RNN architecture (POS tagging): implement both from-scratch NumPy vanilla RNN and PyTorch `nn.RNN` version on a POS tagging task where each time step produces an output tag. (`part_b/rnn_ex51_many_to_many_pos.py`)
52. Demonstrate One-to-Many RNN architecture (music/text generation): implement both vanilla RNN and PyTorch `nn.RNN` for sequence generation from a single input (music melody generation, image captioning). (`part_b/rnn_ex52_one_to_many_music_gen.py`)
53. Demonstrate Many-to-One RNN architecture (sentiment classification): implement both vanilla RNN and PyTorch `nn.RNN` where a full input sequence maps to a single classification label. (`part_b/rnn_ex53_many_to_one_sentiment.py`)
54. Vanilla RNN from scratch (NumPy): build RNN cell, forward pass, BPTT with gradient clipping, momentum optimizer, and train on sine-wave prediction — no frameworks at all. (`part_b/rnn_ex54_vanilla_rnn_numpy.py`)
55. PyTorch RNN covering all three architectures: `nn.RNN` for Many-to-Many (POS tagging), One-to-Many (sequence generation), and Many-to-One (sentiment classification); includes RNN/LSTM/GRU comparison and teacher forcing discussion. (`part_b/rnn_ex55_pytorch_rnn_all_arch.py`)
56. English-Mizo Text-to-Text translation using Seq2Seq RNN with Bahdanau attention and POS embeddings: load parallel corpus (engmiz.txt), build vocabularies, train encoder-decoder, demonstrate translation with attention visualization and POS analysis. (`part_b/rnn_ex56_eng_mizo_seq2seq.py`)
57. Vanilla LSTM from scratch (NumPy): scalar and vector LSTM cells matching classroom calculations, Bahdanau attention, all three architecture patterns (Many-to-Many, One-to-Many, Many-to-One), training loop, parameter count calculation, and RNN vs LSTM comparison. (`part_b/rnn_ex57_lstm_numpy_classroom.py`)
58. Vanilla GRU from scratch (NumPy): scalar and vector GRU cells, exact classroom blackboard arithmetic trace, 3 recurrent architectures, manual BPTT training loop, parameter count derivation, and vanishing gradient comparison. (`part_b/rnn_ex58_gru_numpy_classroom.py`)
59. GRU sequence forecasting & comparative lab exercise (PyTorch): Custom GRU cell vs PyTorch `nn.GRU` vs LSTM vs Vanilla RNN on composite multi-harmonic sequence modeling, training convergence curves, test MSE/MAE, and student lab exercises. (`part_b/rnn_ex59_gru_exercise.py`)

#### Files
| File | Architecture | What it shows |
|------|-------------|----------------|
| `part_b/rnn_ex51_many_to_many_pos.py` | Many-to-Many | Vanilla RNN (NumPy) + PyTorch RNN for POS tagging; output at every time step |
| `part_b/rnn_ex52_one_to_many_music_gen.py` | One-to-Many | Vanilla RNN (NumPy) + PyTorch RNN for music generation + image captioning demo |
| `part_b/rnn_ex53_many_to_one_sentiment.py` | Many-to-One | Vanilla RNN (NumPy) + PyTorch RNN for sentiment analysis + speech rec + anomaly detection demos |
| `part_b/rnn_ex54_vanilla_rnn_numpy.py` | Vanilla RNN | Pure NumPy RNN cell, BPTT, gradient flow analysis, sine-wave training, architecture comparison |
| `part_b/rnn_ex55_pytorch_rnn_all_arch.py` | PyTorch RNN | `nn.RNN` for all three architectures + RNN/LSTM/GRU comparison + teacher forcing |
| `part_b/rnn_ex56_eng_mizo_seq2seq.py` | Many-to-Many Seq2Seq | English-Mizo T2T translation using `nn.RNN` + Bahdanau attention + POS embeddings; real parallel corpus (engmiz.txt) |
| `part_b/rnn_ex57_lstm_numpy_classroom.py` | Vanilla LSTM | Pure NumPy LSTM: scalar & vector cells, classroom calculations, all 3 architectures, training, RNN vs LSTM |
| `part_b/rnn_ex58_gru_numpy_classroom.py` | Vanilla GRU | Pure NumPy GRU: scalar arithmetic trace, vector cell, manual BPTT, 3 architectures, parameter derivation |
| `part_b/rnn_ex59_gru_exercise.py` | PyTorch GRU Exercise | Custom GRU vs Native PyTorch GRU vs LSTM vs RNN benchmark, convergence plots, student lab exercises |

### GANs (35–43)
35. Implement a basic GAN (generator + discriminator MLP) on a 2D Gaussian mixture. (`part_b/gan_q35_basic_gan_2d.py`)
36. Explain the minimax objective and the Nash equilibrium of a GAN. (`part_b/gan_q36_minimax_nash.py`)
37. Implement the Wasserstein GAN (WGAN) with weight clipping and critic training. (`part_b/gan_q37_wgan_weight_clip.py`)
38. Implement WGAN-GP (gradient penalty) and compare stability to vanilla GAN. (`part_b/gan_q38_wgan_gp.py`)
39. Build a DCGAN for generating MNIST/Fashion-MNIST images. (`part_b/gan_q39_dcgan_mnist.py`)
40. Implement a Conditional GAN (cGAN) that generates class-conditioned samples. (`part_b/gan_q40_conditional_gan.py`)
41. Train a CycleGAN (two generators, two discriminators) for unpaired image-to-image translation. (`part_b/gan_q41_cyclegan.py`)
42. Discuss and mitigate mode collapse; visualize generator samples over training. (`part_b/gan_q42_mode_collapse.py`)
43. Implement a Pix2Pix (cGAN with U-Net generator + patch discriminator) for paired translation. (`part_b/gan_q43_pix2pix.py`)

### GNNs (44–50)
44. Implement a basic message-passing layer (GCN) from scratch using adjacency + degree normalization. (`part_b/gnn_q44_gcn_scratch.py`)
45. Apply a GCN to the Cora citation dataset for node classification. (`part_b/gnn_q45_gcn_cora.py`)
46. Build a GraphSAGE layer with neighbor sampling and explain inductive learning. (`part_b/gnn_q46_graphsage.py`)
47. Implement Graph Attention Networks (GAT) with learned attention coefficients. (`part_b/gnn_q47_gat_attention.py`)
48. Compare spectral (GCN) vs spatial (GraphSAGE/GAT) convolution approaches. (`part_b/gnn_q48_spectral_vs_spatial.py`)
49. Use a GNN for link prediction (edge existence) with negative sampling. (`part_b/gnn_q49_link_prediction.py`)
50. Implement a readout/pooling layer for graph-level classification (e.g., molecular property prediction). (`part_b/gnn_q50_graph_pooling.py`)

### Part B — PyTorch CNN Exercises
They sit alongside the 50-question bank and demonstrate CNN ideas end-to-end with NumPy and PyTorch.

| File | Topic | What it shows |
|------|-------|----------------|
| `part_b/cnn_ex00_synthetic_2d_cnn.py` | 2D CNN on synthetic images | Build a 3x3-Conv -> ReLU -> MaxPool -> Dense CNN in `torch.nn`, train on random 8x8 data, print loss/acc per epoch. |
| `part_b/cnn_ex01_sequence_1d_cnn.py` | 1D CNN over sequences | Build a `nn.Conv1d` -> ReLU -> MaxPool1d -> Dense model for sequence classification, mini-batch SGD training loop. |
| `part_b/cnn_ex02_volumetric_3d_cnn.py` | 3D CNN on volumetric data | `nn.Conv3d` + `nn.MaxPool3d` over (frames, H, W) inputs; binary classification with a 2-class dense head. |
| `part_b/cnn_ex03_custom_kernels_sobel.py` | Custom convolution kernels (Sobel, Sharpen) | Handcrafted 3x3 filters loaded into `nn.Conv2d` weights; applies them to a real image fetched from a public URL. |
| `part_b/cnn_ex04_filter_visualisation.py` | Visualizing learned CNN filters | Trains one epoch of a small CNN on CIFAR-10, then plots the 16 learned 3x3 kernels from the first conv layer. |
| `part_b/cnn_ex05_cats_dogs_classifier.py` | Real-image binary classification (cats vs dogs) | Downloads the cats_and_dogs_filtered dataset, builds a 3-conv-layer CNN, trains 5 epochs, and shows sample predictions. |
| `part_b/cnn_ex06_cifar10_cat_dog.py` | Cat vs Dog on CIFAR-10 (binary) | Filters CIFAR-10 to classes {cat, dog}, trains a CNN with `BCEWithLogitsLoss` for 5 epochs, visualizes predictions. |
| `part_b/cnn_ex07_numpy_forward_step_by_step.py` | Step-by-step CNN in pure NumPy | Forward pass with manual Conv2D, ReLU, MaxPool2D, Flatten, Dense layer, and Softmax on a 4x4 matrix. |
| `part_b/cnn_ex08_minimal_cnn_pytorch.py` | Minimal PyTorch ConvNet | Minimal 1-Conv + 1-Linear architecture trained on synthetic 8x8 data with SGD optimization. |
| `part_b/cnn_ex09_mnist_cnn_pytorch.py` | Full MNIST CNN Classification | 2-block ConvNet + 2-layer classifier on MNIST with DataLoader, cross-entropy loss, Adam, and test accuracy. |

```bash
python part_b/cnn_ex00_synthetic_2d_cnn.py          # tiny synthetic CNN
python part_b/cnn_ex03_custom_kernels_sobel.py     # handcrafted Sobel/sharpen filter visualization
python part_b/cnn_ex05_cats_dogs_classifier.py     # cats vs dogs (auto-downloads dataset)
python part_b/cnn_ex06_cifar10_cat_dog.py          # CIFAR-10 cat vs dog (auto-downloads)
python part_b/cnn_ex07_numpy_forward_step_by_step.py # NumPy manual CNN forward pass
python part_b/cnn_ex09_mnist_cnn_pytorch.py        # complete MNIST training pipeline
```

### Part B — PyTorch RNN & GRU Exercises

| File | Topic | What it shows |
|------|-------|----------------|
| `part_b/rnn_ex51_many_to_many_pos.py` | Many-to-Many RNN | POS tagging with vanilla NumPy RNN + PyTorch `nn.RNN`; output at every time step |
| `part_b/rnn_ex52_one_to_many_music_gen.py` | One-to-Many RNN | Music generation + image captioning with vanilla NumPy RNN + PyTorch `nn.RNN` |
| `part_b/rnn_ex53_many_to_one_sentiment.py` | Many-to-One RNN | Sentiment analysis with vanilla NumPy RNN + PyTorch `nn.RNN`; sequence → single label |
| `part_b/rnn_ex54_vanilla_rnn_numpy.py` | Vanilla RNN from scratch | Pure NumPy RNN cell, BPTT, gradient clipping, sine-wave prediction |
| `part_b/rnn_ex55_pytorch_rnn_all_arch.py` | PyTorch RNN all architectures | `nn.RNN` for Many-to-Many, One-to-Many, Many-to-One; RNN/LSTM/GRU comparison |
| `part_b/rnn_ex56_eng_mizo_seq2seq.py` | English-Mizo T2T Translation | Seq2Seq RNN with attention + POS on engmiz parallel corpus |
| `part_b/rnn_ex57_lstm_numpy_classroom.py` | Vanilla LSTM (NumPy) | Scalar & vector LSTM cells, classroom calculations, 3 architectures, BPTT |
| `part_b/rnn_ex58_gru_numpy_classroom.py` | Vanilla GRU (NumPy) | Scalar arithmetic trace, vectorized cell, manual BPTT, 3 architectures |
| `part_b/rnn_ex59_gru_exercise.py` | GRU Sequence Modeling Exercise | Custom GRU vs Native PyTorch GRU/LSTM/RNN, loss curves, test forecasting |

```bash
python part_b/rnn_ex58_gru_numpy_classroom.py      # NumPy GRU derivation and classroom trace
python part_b/rnn_ex59_gru_exercise.py             # PyTorch GRU vs LSTM vs RNN benchmark & exercise
```

### Part C — Dataset Exercises (EDA, Train/Validate/Test + Plots)

Each exercise performs dataset analysis, trains a model with a validation split,
reports test accuracy/loss, plots accuracy & loss curves, plots ROC (one-vs-rest /
macro), and measures inference time and parameter count (complexity proxy).
Figures are written to `part_c/figures/`.

| File | Dataset | Model | Highlights |
|------|---------|-------|------------|
| `part_c/pc01_iris_logreg.py` | Iris | LogisticRegression | EDA, learning curve, OvR ROC, μs/sample inference |
| `part_c/pc02_breast_cancer_lr_mlp.py` | Breast Cancer | LR vs MLP | ROC-AUC comparison, loss curve, params vs latency |
| `part_c/pc03_mnist_cnn.py` | MNIST | CNN (PyTorch) | per-epoch acc/loss, 10-class ROC, throughput, params |
| `part_c/pc04_fashion_mnist_shallow_deep.py` | Fashion-MNIST | Shallow vs Deep CNN | architecture comparison, acc/loss, complexity bar charts |
| `part_c/pc05_mlp_time_complexity.py` | Synthetic tabular | MLP (PyTorch) | inference scaling vs samples and model width, ROC |

Run example:
```bash
python part_c/pc01_iris_logreg.py
python part_c/pc03_mnist_cnn.py   # downloads MNIST on first run
```

---

## Part D — Network Security Exercises (Wireless, SDN, Cloud, Edge, IoT, Adversarial)

Each exercise applies ML/DL to a security domain, with EDA, train/validate/test,
accuracy/loss/ROC plots, and inference-time / footprint analysis. Figures go to
`part_d/figures/`.

### Classical ML security (pd01–pd05)
| File | Domain | Technique | Highlights |
|------|--------|-----------|------------|
| `part_d/pd01_wireless_ids_mlp.py` | Wireless (802.11) | MLP | Synthetic RSSI/SNR/rate IDS, ROC, μs/flow latency |
| `part_d/pd02_sdn_ddos_lr_mlp_rf.py` | SDN | LR vs MLP vs RF | DDoS detection, ROC-AUC comparison, complexity bars |
| `part_d/pd03_cloud_anomaly_autoencoder.py` | Cloud | Autoencoder | Reconstruction-error anomaly detection, ROC, error dist |
| `part_d/pd04_edge_fedavg.py` | Edge / SDN | Federated (FedAvg) | Edge-node FL vs centralized, ROC, inference latency |
| `part_d/pd05_iot_edge_dt_mlp.py` | IoT/Edge | DT vs MLP | Footprint/accuracy/latency trade-off for edge gateways |

### Deep-learning security (pd06–pd10)
| File | Domain | Technique | Highlights |
|------|--------|-----------|------------|
| `part_d/pd06_cnn_traffic.py` | Traffic | 1D-CNN | Sequence-of-flows ConvNet, acc/loss/ROC, latency |
| `part_d/pd07_lstm_gru_ids.py` | IDS | LSTM / GRU | Temporal intrusion detection, ROC, latency |
| `part_d/pd08_adversarial_fgsm.py` | Adversarial ML | FGSM attack | IDS robustness vs perturbation eps, accuracy drop |
| `part_d/pd09_transformer_anomaly.py` | Cloud/IoT | Transformer encoder | Self-attention anomaly detection, ROC, latency |
| `part_d/pd10_gan_augmentation.py` | Data scarcity | GAN | Synthetic attack generation boosts classifier AUC |

Run example:
```bash
python part_d/pd01_wireless_ids_mlp.py
python part_d/pd08_adversarial_fgsm.py     # FGSM robustness demo
python part_d/pd10_gan_augmentation.py    # GAN data augmentation
```

### Real-dataset pipelines with full evaluation (pd11–pd13)
These download/extract **KDD'99**, **NSL-KDD**, and **INSDN**, parse the CSV/raw
files, split into train/validation/test, and train **CNN, LSTM, GRU, Hybrid
(CNN+LSTM), and GAN-augmented** models. For each they report accuracy, loss &
ROC curves, confusion matrices, training/inference time and parameter counts, plus
interpretability via **ANOVA**, **SHAP**, and **LIME**. A shared module
`part_d/security_utils.py` handles downloading (with a synthetic fallback if
offline), preprocessing, model definitions, metrics, and plotting.

| File | Dataset | What it does |
|------|---------|--------------|
| `part_d/security_utils.py` | — | Shared loader/preprocess/models/metrics/interpretability |
| `part_d/pd11_kdd99_pipeline.py` | KDD'99 | Full CNN/LSTM/GRU/Hybrid/GAN pipeline + SHAP/LIME/ANOVA |
| `part_d/pd12_nslkdd_pipeline.py` | NSL-KDD | Same pipeline on NSL-KDD |
| `part_d/pd13_insdn_pipeline.py` | INSDN (SDN) | Same pipeline on INSDN flow CSV |

```bash
python part_d/pd11_kdd99_pipeline.py    # downloads KDD'99 if reachable, else synthetic
python part_d/pd13_insdn_pipeline.py    # downloads INSDN if reachable, else synthetic
```

---

## Part E — NLP / Transformers / LLMs (with datasets)

Practical exercises spanning tokenization, from-scratch Transformers, LLM prompting,
embeddings/search, parameter-efficient fine-tuning, and a **Mizo-language**
case study (Mizo↔English, Mizo↔Hindi, both directions) with corpus creation,
POS tagging, and tone/diacritic-aware translation.

| File | Topic | Technique |
|------|-------|-----------|
| `part_e/pe01_sentiment_distilbert.py` | Sentiment classification | Transformer (DistilBERT) / BiLSTM fallback + ROC |
| `part_e/pe02_transformer_encoder_scratch.py` | Transformer encoder from scratch | PosEnc + multi-head attention, attention viz |
| `part_e/pe03_multihead_attention_numpy.py` | Multi-head self-attention | NumPy implementation + heatmaps |
| `part_e/pe04_gpt_decoder_scratch.py` | Decoder-only GPT (char-level) | Causal LM, text generation |
| `part_e/pe05_tokenization_bpe.py` | Word vs subword (BPE) | HF tokenizers / from-scratch BPE |
| `part_e/pe06_ner_bilstm.py` | Named Entity Recognition | BiLSTM tagger + span-F1 |
| `part_e/pe07_extractive_qa_squad.py` | Extractive QA (SQuAD) | HF QA pipeline / TF-IDF fallback |
| `part_e/pe08_llm_prompting.py` | LLM prompting + perplexity | zero/few-shot/CoT, latency, PPL |
| `part_e/pe09_semantic_search_embeddings.py` | Semantic search | sentence-transformers / TF-IDF + Recall@k |
| `part_e/pe10_lora_peft.py` | Efficient fine-tuning | LoRA vs full FT (params/latency) |
| `part_e/pe11_mizo_corpus_pos.py` | Mizo corpus + POS tagging | Diacritic-aware tokenizer/POS (lêi/léi/lèi) |
| `part_e/pe12_mizo_translation_transformer.py` | Mizo translation (4 directions) | Char-level encoder-decoder Transformer |

The Mizo exercises demonstrate why **diacritics matter**: `Lêi` (purchase, VERB),
`Léi` (tongue/bent, NOUN/ADJ) and `Lèi` (bridge/ladder, NOUN) are distinct; a
diacritic-blind model collapses them and cannot disambiguate meaning.

```bash
python part_e/pe11_mizo_corpus_pos.py     # builds part_e/data/mizo_corpus.csv + POS
python part_e/pe12_mizo_translation_transformer.py    # trains 4 translation directions
# set PE07_HF=1 / PE08_HF=1 to use real HuggingFace models (downloads weights)
```

---

## Part F — Music / Audio: MIDI, WAV & Composition

Practical exercises for music information processing: build (or download) a MIDI
dataset, extract features, classify genre, and **compose new music**. A shared
`part_f/music_utils.py` synthesizes a multi-genre MIDI-style dataset when no real
`.mid` files are present, and can load real datasets (MAESTRO / Lakh / Wikifonia)
via `pretty_midi` if you point `download_or_load()` at a folder of
`<genre>/*.mid` files. WAV synthesis/spectrograms use only the stdlib `wave`
module + numpy (no `librosa` required); MIDI export uses `pretty_midi` when present
and falls back to `.npz` otherwise.

| File | Task | Model / Technique |
|------|------|-------------------|
| `part_f/pf01_midi_eda_features.py` | MIDI/WAV loading + EDA | feature extraction, boxplots, WAV spectrograms |
| `part_f/pf02_genre_classifier_rf_mlp.py` | Genre classification | RandomForest + MLP, ROC/confusion, timing |
| `part_f/pf03_lstm_composer.py` | Music composition | LSTM language model over note tokens |
| `part_f/pf04_transformer_composer.py` | Music composition | Decoder-only Transformer LM (genre-tokenized) |
| `part_f/pf05_vae_style_composer.py` | Style representation | VAE latent space + genre interpolation (style transfer) |

```bash
python part_f/pf01_midi_eda_features.py        # dataset, features, WAV + spectrograms
python part_f/pf03_lstm_composer.py            # generates part_f/generated/lstm_composition.mid
python part_f/pf05_vae_style_composer.py       # interpolates classical<->rock in latent space
```

---

## Repository Layout
```
DL_Lab/
├── README.md                # this file (syllabus + 50-question bank + exercises)
├── part_a/                  # practical_01..practical_30 (Foundations, Activations, Losses, MLPs, Optimizers, PyTorch DNNs) + figures/
├── part_b/                  # cnn_q01..q20, rnn_q21..q34, gan_q35..q43, gnn_q44..q50 (Question bank) + cnn_ex00..ex09 (CNN exercises) + rnn_ex51..ex59 (RNN & GRU demos, translation) + engmiz.txt
├── part_c/                  # pc01..pc05 dataset exercises (Iris, Cancer, MNIST, Fashion-MNIST, Time Complexity) + figures/
├── part_d/                  # pd01..pd13 network-security exercises, security_utils.py, data/ + figures/
├── part_e/                  # pe01..pe12 NLP/Transformer/LLM + Mizo translation, data/ + figures/
└── part_f/                  # pf01..pf05 music/MIDI/WAV composition, music_utils.py, data/ + generated/ + figures/
```

## Requirements & Installation

All practicals share a common core; some need deep-learning or graph libraries.

### Packages required
| Package | Used by | Purpose |
|---------|---------|---------|
| `numpy` | Part A (01–17, 21–23), Part B (cnn_q01–04, rnn_q21–25, gnn_q44, cnn_ex07, rnn_ex51–54, 57), Part C (pc01/02/05), Part D | Array math, from-scratch models |
| `matplotlib` | Part A (04, 05, 06, 13–14, 16…), Part B (visualizations), Part C/D/E/F | Plotting accuracy/loss/ROC |
| `scikit-learn` | Part A (02, 03, 09, 12), Part C (pc01/02/05), Part D (metrics, `f_classif`, preprocessing) | Datasets, metrics, feature selection |
| `scipy` | Part D (stats, `f_classif` backend) | ANOVA / hypothesis testing |
| `pandas` | Part D (INSDN CSV parsing in `security_utils.py`) | Tabular data loading |
| `torch` | Part A (10, 18–20, 24–30), Part B (cnn_q05–20, rnn_q23–34, gan_q35–43, cnn_ex*, rnn_ex*), Part C (pc03/04/05), Part D (all `pd`) | Neural nets, autograd, training loops |
| `torchvision` | Part A (18, 19), Part B (cnn_q20, cnn_ex04–06, cnn_ex09), Part C (pc03/04) | MNIST / Fashion-MNIST / CIFAR-10 / cats-vs-dogs datasets, pretrained models |
| `torch_geometric` | Part B (gnn_q45) | GCN/GAT/GraphSAGE on Cora (optional) |
| `nltk` / `transformers` | Part B (rnn_q27–28 tokenization, optional), Part E | Text datasets / tokenizers (optional) |
| `shap` | Part D (`security_utils.shap_summary`) | Model interpretability (feature attributions) |
| `lime` | Part D (`security_utils.lime_explain`) | Local instance explanations |

### Install
```bash
# Core (covers Part A 01–17, Part B scratch models, Part C pc01/02/05)
pip install numpy matplotlib scikit-learn

# Deep learning (Part A 10/18–30, Part B PyTorch models, Part C pc03/04/05, Part D all pd)
pip install torch torchvision

# Part D real-dataset pipelines (pd11-pd13): parsing + interpretability
pip install pandas scipy shap lime

# Optional: graph neural networks (Part B gnn_q45) and NLP helpers (Part B & E)
pip install torch_geometric nltk transformers
```

### Dataset downloads (automatic on first run)
- `torchvision` datasets: **MNIST**, **Fashion-MNIST** (pc03, pc04, Part A 18/19, Part B gan_q39/40, cnn_ex09)
- `torch_geometric` dataset: **Cora** (gnn_q45)
- `scikit-learn` datasets: **Iris**, **Breast Cancer**, **Blobs**, **Circles** (local, no download)

> Note: `torch`/`torchvision` builds are CUDA-optional; CPU-only installs work for every
> script here. If GPU is unavailable the code automatically falls back to CPU.

> Note: `torch`/`torchvision` builds are CUDA-optional; CPU-only installs work for every
> script here. If GPU is unavailable the code automatically falls back to CPU.
