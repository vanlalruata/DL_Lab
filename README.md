# Deep Learning Lab — Practical & Question Bank

**Programme:** Master of Computer Applications (MCA)
**Institution:** Mizoram University, Aizawl, Mizoram, India
**Course:** Deep Learning Lab (Practical Work / Lab Records)

This repository contains the lab exercises and practical records for the MCA Deep
Learning Lab at Mizoram University. It includes the syllabus-mapped practicals
(`practical_1.py` ... `practical_20.py`) from Lab Unit 1, an extended question
bank of 50 items covering CNNs (AlexNet, VGG, GoogLeNet, ResNet, DenseNet), GNNs,
RNN/LSTM/GRU, and GANs, plus additional hands-on exercises in dataset analysis,
network security, and NLP / Transformer / LLM topics (including Mizo-language
translation and POS tagging).

---

## Part A — Module Practicals (from Lab Unit 1)

### Module 1: Foundational Computational Units & Perceptrons (NumPy)
- **Practical 1 — McCulloch-Pitts Neuron from Scratch**
  Implement an M-P neuron class in NumPy to simulate AND, OR, NOT, and NOR. Show via
  output truth tables why a single-layer M-P thresholding fails on the non-linearly
  separable XOR gate.
- **Practical 2 — Single-Layer Perceptron Learning Algorithm (PLA)**
  Code the Rosenblatt Perceptron with step activation. Generate a linearly separable
  2D dataset with `sklearn.datasets.make_blobs`, train the perceptron, and plot the
  evolving decision boundary at each epoch until convergence.
- **Practical 3 — Demonstrating the Linear Separability Constraint**
  Test the Perceptron on a non-linearly separable dataset (XOR / concentric circles).
  Visualize how it fails to converge, plotting perpetual oscillations in classification
  loss and boundary updates.

### Module 2: Activation Functions & Visualizations (NumPy & Matplotlib)
- **Practical 4 — Activation Function Zoo and Gradient Visualizer**
  Implement forward and derivative of Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, and SELU.
  Plot a 2×3 grid comparing each function with its first derivative over x ∈ [-5, 5].
- **Practical 5 — The Vanishing Gradient Simulation in Deep Feedforward Networks**
  Build an N-layer forward pass in pure NumPy. Back-propagate an initial gradient
  through 10 hidden layers using Sigmoid vs ReLU; plot gradient magnitude per layer.
- **Practical 6 — Softmax and Stable Numerical Implementations**
  Implement standard vs numerically stable Softmax (subtract max(z)). Pass extreme
  logits z = [1000, 1001, 1002] to show vanilla Softmax NaN overflow vs stable handling.

### Module 3: Loss Functions (NumPy & PyTorch)
- **Practical 7 — Regression Loss Functions (MSE, MAE, Huber)**
  Implement MSE, MAE, and Huber (δ=1.0) in NumPy on a synthetic dataset with high
  magnitude outliers; plot and compare outlier sensitivity.
- **Practical 8 — Binary Cross-Entropy vs MSE for Binary Classification**
  Build a single Sigmoid neuron; compare loss-surface convexities of MSE vs BCE over
  ranges of weight and bias values.
- **Practical 9 — Multi-Class CCE & Softmax Coupling**
  Implement CCE from scratch in NumPy; compare performance and memory of One-Hot
  (Categorical CE) vs Integer (Sparse Categorical CE) targets.
- **Practical 10 — Advanced Loss Functions in PyTorch (Focal Loss)**
  Implement Focal Loss as a custom `torch.nn.Module`; train on a 95/5 imbalanced
  dataset and compare accuracy/recall with `nn.BCEWithLogitsLoss`.

### Module 4: Multi-Layer Perceptrons & Backpropagation from Scratch (NumPy)
- **Practical 11 — 2-Layer MLP Backpropagation from Scratch (Solving XOR)**
  Build a 2-2-1 MLP with Sigmoid activations; derive and code manual backprop
  (∂L/∂W1, ∂L/∂b1, ∂L/∂W2, ∂L/∂b2) to learn XOR.
- **Practical 12 — General N-Layer MLP Engine in NumPy**
  Build a flexible class accepting arbitrary topologies (e.g. [4,16,8,3]); implement
  automated forward/backward and mini-batch updates to classify the Iris dataset.

### Module 5: Optimizers from Scratch (NumPy)
- **Practical 13 — Gradient Descent Variants Comparison**
  Implement Batch GD, SGD, and Mini-Batch GD on linear regression; plot loss vs CPU
  time and weight trajectories on a 2D contour map.
- **Practical 14 — Momentum & Nesterov Accelerated Gradient (NAG)**
  Implement Polyak Momentum and NAG; optimize a 2D pathological ravine (Rosenbrock)
  and visualize damped transverse oscillations.
- **Practical 15 — Adaptive Learning Rates (AdaGrad vs RMSProp)**
  Implement AdaGrad and RMSProp; show AdaGrad stalls on sparse gradients while RMSProp
  continues to learn.
- **Practical 16 — Complete Implementation of the Adam Optimizer**
  Code Adam (m_t, v_t, bias corrections m̂_t, v̂_t); plot weight trajectories and the
  smoothing impact of bias correction over the first 10 steps.
- **Practical 17 — Modern Optimizer Variants (AdamW & AdaDelta)**
  Implement AdamW (decoupled weight decay) and AdaDelta (learning-rate-free); compare
  weight-norm decay vs standard Adam + L2 over 100 epochs.

### Module 6: End-to-End Neural Networks in PyTorch
- **Practical 18 — End-to-End Classification Pipeline using torch.nn**
  Build a 3-hidden-layer MLP with `nn.Sequential`/`nn.Linear`/`nn.ReLU`/
  `nn.CrossEntropyLoss`; train on MNIST with a DataLoader.
- **Practical 19 — PyTorch Custom Optimizer & Loss Benchmarking**
  Train identical MLPs on Fashion-MNIST with SGD, SGD(momentum=0.9), Adagrad, RMSprop,
  and AdamW; plot combined training-loss and validation-accuracy curves.
- **Practical 20 — Self-Normalizing Networks with SELU and AlphaDropout**
  Build a 10-layer feedforward net with `nn.SELU`/`nn.AlphaDropout`; record mean and
  variance of hidden activations to verify the self-normalizing property.

---

## Part B — Extended Question Bank (50 questions)

### CNNs & Classic Architectures (1–20)
1. Implement a 2D convolution operation from scratch in NumPy (no `nn.Conv2d`).
2. Explain and code the difference between "same", "valid", and "causal" padding.
3. Derive the output spatial size of a conv layer given kernel, stride, and padding.
4. Implement max-pooling and average-pooling from scratch; discuss translation invariance.
5. Build AlexNet in PyTorch and discuss why ReLU + dropout were key to its success.
6. Reproduce VGG-16 block design; analyze parameter count growth from stacked 3×3 kernels.
7. Compare 7×7, 5×5, and 3×3 convolutions: receptive field vs parameter cost.
8. Implement GoogLeNet/Inception module with parallel branches (1×1, 3×3, 5×5, pool).
9. Explain the role of 1×1 convolutions (dimensionality reduction, channel mixing).
10. Implement batch normalization from scratch and integrate it into a CNN training loop.
11. Build ResNet basic-block and bottleneck-block; implement residual (skip) connections.
12. Explain why residual connections mitigate vanishing gradients in very deep networks.
13. Implement ResNeXt grouped convolutions and compare with standard ResNet.
14. Build DenseNet dense blocks with concatenation; discuss feature reuse and parameters.
15. Implement a depthwise-separable convolution (MobileNet style) and count its savings.
16. Compare parameter counts and FLOPs of AlexNet, VGG-16, and ResNet-50.
17. Implement global average pooling and explain its use as a regularizer in GoogLeNet.
18. Build a U-Net style encoder-decoder with skip connections for image segmentation.
19. Implement transpose convolution (fractional strided conv) for upsampling.
20. Visualize CNN filters and feature maps of a pretrained model (e.g. via hooks).

### RNN / LSTM / GRU (21–34)
21. Implement a simple RNN cell from scratch and train it on a sine-wave prediction task.
22. Derive the BPTT equations for a vanilla RNN and discuss the exploding gradient problem.
23. Implement gradient clipping and demonstrate its effect on RNN training stability.
24. Build an LSTM cell from scratch; explain the roles of forget, input, and output gates.
25. Implement a GRU cell and compare its gate structure to LSTM.
26. Compare LSTM vs GRU on a language-modeling or sequence task (perplexity/accuracy).
27. Train an LSTM for sentiment classification on a text dataset (e.g. IMDB).
28. Implement sequence-to-sequence for character-level text generation.
29. Build a bidirectional RNN/LSTM and explain when future context helps.
30. Implement an attention mechanism over LSTM hidden states (Bahdanau-style).
31. Apply masking in padded sequence batches (`pack_padded_sequence`).
32. Visualize what an LSTM gate learns on a long-range dependency toy problem.
33. Compare teacher forcing vs scheduled sampling in sequence generation.
34. Implement a stacked (multi-layer) LSTM and analyze representational depth.

### GANs (35–43)
35. Implement a basic GAN (generator + discriminator MLP) on a 2D Gaussian mixture.
36. Explain the minimax objective and the Nash equilibrium of a GAN.
37. Implement the Wasserstein GAN (WGAN) with weight clipping and critic training.
38. Implement WGAN-GP (gradient penalty) and compare stability to vanilla GAN.
39. Build a DCGAN for generating MNIST/Fashion-MNIST images.
40. Implement a Conditional GAN (cGAN) that generates class-conditioned samples.
41. Train a CycleGAN (two generators, two discriminators) for unpaired image-to-image translation.
42. Discuss and mitigate mode collapse; visualize generator samples over training.
43. Implement a Pix2Pix (cGAN with U-Net generator + patch discriminator) for paired translation.

### GNNs (44–50)
44. Implement a basic message-passing layer (GCN) from scratch using adjacency + degree normalization.
45. Apply a GCN to the Cora citation dataset for node classification.
46. Build a GraphSAGE layer with neighbor sampling and explain inductive learning.
47. Implement Graph Attention Networks (GAT) with learned attention coefficients.
48. Compare spectral (GCN) vs spatial (GraphSAGE/GAT) convolution approaches.
49. Use a GNN for link prediction (edge existence) with negative sampling.
50. Implement a readout/pooling layer for graph-level classification (e.g., molecular property prediction).

---

---

## Part C — Dataset Exercises (EDA, Train/Validate/Test + Plots)

Each exercise performs dataset analysis, trains a model with a validation split,
reports test accuracy/loss, plots accuracy & loss curves, plots ROC (one-vs-rest /
macro), and measures inference time and parameter count (complexity proxy).
Figures are written to `part_c/figures/`.

| File | Dataset | Model | Highlights |
|------|---------|-------|------------|
| `part_c/pc01_iris.py` | Iris | LogisticRegression | EDA, learning curve, OvR ROC, μs/sample inference |
| `part_c/pc02_breast_cancer.py` | Breast Cancer | LR vs MLP | ROC-AUC comparison, loss curve, params vs latency |
| `part_c/pc03_mnist_cnn.py` | MNIST | CNN (PyTorch) | per-epoch acc/loss, 10-class ROC, throughput, params |
| `part_c/pc04_fashion_mnist.py` | Fashion-MNIST | Shallow vs Deep CNN | architecture comparison, acc/loss, complexity bar charts |
| `part_c/pc05_time_complexity.py` | Synthetic tabular | MLP (PyTorch) | inference scaling vs samples and model width, ROC |

Run example:
```bash
python part_c/pc01_iris.py
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
| `part_d/pd01_wireless_ids.py` | Wireless (802.11) | MLP | Synthetic RSSI/SNR/rate IDS, ROC, μs/flow latency |
| `part_d/pd02_sdn_ddos.py` | SDN | LR vs MLP vs RF | DDoS detection, ROC-AUC comparison, complexity bars |
| `part_d/pd03_cloud_anomaly.py` | Cloud | Autoencoder | Reconstruction-error anomaly detection, ROC, error dist |
| `part_d/pd04_edge_federated.py` | Edge / SDN | Federated (FedAvg) | Edge-node FL vs centralized, ROC, inference latency |
| `part_d/pd05_iot_edge.py` | IoT/Edge | DT vs MLP | Footprint/accuracy/latency trade-off for edge gateways |

### Deep-learning security (pd06–pd10)
| File | Domain | Technique | Highlights |
|------|--------|-----------|------------|
| `part_d/pd06_cnn_traffic.py` | Traffic | 1D-CNN | Sequence-of-flows ConvNet, acc/loss/ROC, latency |
| `part_d/pd07_lstm_ids.py` | IDS | LSTM / GRU | Temporal intrusion detection, ROC, latency |
| `part_d/pd08_adversarial_ids.py` | Adversarial ML | FGSM attack | IDS robustness vs perturbation eps, accuracy drop |
| `part_d/pd09_transformer_anomaly.py` | Cloud/IoT | Transformer encoder | Self-attention anomaly detection, ROC, latency |
| `part_d/pd10_gan_augmentation.py` | Data scarcity | GAN | Synthetic attack generation boosts classifier AUC |

Run example:
```bash
python part_d/pd01_wireless_ids.py
python part_d/pd08_adversarial_ids.py     # FGSM robustness demo
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
| `part_e/pe01_sentiment_transformer.py` | Sentiment classification | Transformer (DistilBERT) / BiLSTM fallback + ROC |
| `part_e/pe02_transformer_scratch.py` | Transformer encoder from scratch | PosEnc + multi-head attention, attention viz |
| `part_e/pe03_attention_numpy.py` | Multi-head self-attention | NumPy implementation + heatmaps |
| `part_e/pe04_gpt_scratch.py` | Decoder-only GPT (char-level) | Causal LM, text generation |
| `part_e/pe05_tokenization_bpe.py` | Word vs subword (BPE) | HF tokenizers / from-scratch BPE |
| `part_e/pe06_ner.py` | Named Entity Recognition | BiLSTM tagger + span-F1 |
| `part_e/pe07_qa_squad.py` | Extractive QA (SQuAD) | HF QA pipeline / TF-IDF fallback |
| `part_e/pe08_llm_prompting.py` | LLM prompting + perplexity | zero/few-shot/CoT, latency, PPL |
| `part_e/pe09_embeddings_search.py` | Semantic search | sentence-transformers / TF-IDF + Recall@k |
| `part_e/pe10_lora_peft.py` | Efficient fine-tuning | LoRA vs full FT (params/latency) |
| `part_e/pe11_mizo_corpus_pos.py` | Mizo corpus + POS tagging | Diacritic-aware tokenizer/POS (lêi/léi/lèi) |
| `part_e/pe12_mizo_transformer.py` | Mizo translation (4 directions) | Char-level encoder-decoder Transformer |

The Mizo exercises demonstrate why **diacritics matter**: `Lêi` (purchase, VERB),
`Léi` (tongue/bent, NOUN/ADJ) and `Lèi` (bridge/ladder, NOUN) are distinct; a
diacritic-blind model collapses them and cannot disambiguate meaning.

```bash
python part_e/pe11_mizo_corpus_pos.py     # builds part_e/data/mizo_corpus.csv + POS
python part_e/pe12_mizo_transformer.py    # trains 4 translation directions
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
| `part_f/pf01_midi_features.py` | MIDI/WAV loading + EDA | feature extraction, boxplots, WAV spectrograms |
| `part_f/pf02_genre_classifier.py` | Genre classification | RandomForest + MLP, ROC/confusion, timing |
| `part_f/pf03_lstm_composer.py` | Music composition | LSTM language model over note tokens |
| `part_f/pf04_transformer_composer.py` | Music composition | Decoder-only Transformer LM (genre-tokenized) |
| `part_f/pf05_vae_composer.py` | Style representation | VAE latent space + genre interpolation (style transfer) |

```bash
python part_f/pf01_midi_features.py        # dataset, features, WAV + spectrograms
python part_f/pf03_lstm_composer.py        # generates part_f/generated/lstm_composition.mid
python part_f/pf05_vae_composer.py         # interpolates classical<->rock in latent space
```

---

## Repository Layout
```
DL_Lab/
├── README.md                # this file (syllabus + 50-question bank + exercises)
├── part_a/                  # practical_1.py ... practical_20.py  (Module labs)
├── part_b/                  # ext01.py ... ext50.py                (CNN/RNN/GAN/GNN)
├── part_c/                  # pc01..pc05 dataset exercises + figures/
├── part_d/                  # pd01..pd13 network-security exercises, security_utils.py, data/ + figures/
├── part_e/                  # pe01..pe12 NLP/Transformer/LLM + Mizo translation, data/ + figures/
└── part_f/                  # pf01..pf05 music/MIDI/WAV composition, music_utils.py, data/ + generated/ + figures/
```

## Requirements & Installation

All practicals share a common core; some need deep-learning or graph libraries.

### Packages required
| Package | Used by | Purpose |
|---------|---------|---------|
| `numpy` | Part A (1–17), Part B (all `ext`), Part C (pc01/02/05), Part D | Array math, from-scratch models |
| `matplotlib` | Part A (4,5,6,13–14,16…), Part B (all `ext`), Part C/D (all `pc`/`pd`) | Plotting accuracy/loss/ROC |
| `scikit-learn` | Part A (2,3,9,12), Part C (pc01/02/05), Part D (metrics, `f_classif`, preprocessing) | Datasets, metrics, feature selection |
| `scipy` | Part D (stats, `f_classif` backend) | ANOVA / hypothesis testing |
| `pandas` | Part D (INSDN CSV parsing in `security_utils.py`) | Tabular data loading |
| `torch` | Part A (10,18–20), Part B (ext05–20,23–41), Part C (pc03/04/05), Part D (all `pd`) | Neural nets, autograd, training loops |
| `torchvision` | Part A (18,19), Part B (20), Part C (pc03/04) | MNIST / Fashion-MNIST datasets, pretrained models |
| `torch_geometric` | Part B (ext45) | GCN/GAT/GraphSAGE on Cora (optional) |
| `nltk` / `transformers` | Part B (ext27–28 tokenization, optional) | Text datasets / tokenizers (optional) |
| `shap` | Part D (`security_utils.shap_summary`) | Model interpretability (feature attributions) |
| `lime` | Part D (`security_utils.lime_explain`) | Local instance explanations |

### Install
```bash
# Core (covers Part A 1–17, Part B ext*, Part C pc01/02/05)
pip install numpy matplotlib scikit-learn

# Deep learning (Part A 10/18–20, Part B ext05–41, Part C pc03/04/05, Part D all pd)
pip install torch torchvision

# Part D real-dataset pipelines (pd11-pd13): parsing + interpretability
pip install pandas scipy shap lime

# Optional: graph neural networks (Part B ext45) and NLP helpers (ext27/28)
pip install torch_geometric nltk transformers
```

### Dataset downloads (automatic on first run)
- `torchvision` datasets: **MNIST**, **Fashion-MNIST** (pc03, pc04, Part A 18/19, Part B ext39/40)
- `torch_geometric` dataset: **Cora** (ext45)
- `scikit-learn` datasets: **Iris**, **Breast Cancer**, **Blobs**, **Circles** (local, no download)

> Note: `torch`/`torchvision` builds are CUDA-optional; CPU-only installs work for every
> script here. If GPU is unavailable the code automatically falls back to CPU.
