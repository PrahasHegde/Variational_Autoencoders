# CelebA Variational Autoencoder (VAE) & $\beta$-VAE Project

This project implements a Variational Autoencoder (VAE) and its disentangled variant ($\beta$-VAE) using PyTorch. Trained on the CelebA dataset, the model learns a continuous latent representation of human faces, allowing for image reconstruction, latent space interpolation (morphing), and feature disentanglement.

## 📂 Project Structure

```text
├── img_align_celeba/          # Dataset folder (contains .jpg images)
├── vae_celeba.pth             # Trained Baseline VAE weights (Task 1)
├── vae_beta_4.pth             # Trained Beta-VAE (Beta=4) weights (Task 3)
├── vae_beta_10.pth            # Trained Beta-VAE (Beta=10) weights (Task 3)
├── index.py                   # Task 1: Baseline VAE Training Script
├── visualize.py               # Task 2: Latent Space Visualization Script
├── beta_vae.py                # Task 3: Beta-VAE Training & Comparison Script
├── training_loss.png          # Plot: Training loss curve
├── reconstruction_results.png # Plot: Reconstruction samples
└── README.md                  # Project documentation
```

## 🚀 Tasks Overview

### Task 1: Baseline VAE Implementation
**Goal:** Build and train a standard VAE to compress and reconstruct 64x64 aligned face images.

* **Architecture:**
    * **Encoder:** 4-layer Convolutional Neural Network (Channels: 32 → 64 → 128 → 256). Uses `BatchNorm` and `LeakyReLU`.
    * **Latent Space:** 128-dimensional continuous vector.
    * **Decoder:** 4-layer Transposed Convolutional Network (Mirror of encoder) ending with `Sigmoid` activation.
* **Loss Function:** $L = \text{BCE (Reconstruction)} + \text{KLD (Regularization)}$.
* **Training:** Adam Optimizer ($lr=1e^{-3}$), Batch Size 64, 10 Epochs.

### Task 2: Latent Space Exploration
**Goal:** Analyze the learned manifold of faces.

* **Reconstruction:** Visualizing original vs. generated faces to verify model fidelity.
* **Interpolation:** Morphing between two distinct faces by linearly interpolating their latent vectors ($z$).
* **Traversal:** Varying single dimensions of the latent vector to observe "disentangled" features (e.g., rotating head, changing skin tone).

### Task 3: $\beta$-VAE (Disentanglement)
**Goal:** Train models with higher $\beta$ values to enforce stricter independence in the latent space.

* **Method:** Trained two variants with $\beta=4$ and $\beta=10$.
* **Loss Function:** $L = \text{BCE} + \beta \cdot \text{KLD}$.
* **Trade-off Observed:**
    * **Low $\beta$ (Standard):** Sharper images, but entangled features (changing hair might change gender).
    * **High $\beta$ (e.g., 10):** Blurrier images, but highly disentangled features (single neurons control specific attributes).

## 💻 How to Run

### Prerequisites
* Python 3.8+
* PyTorch (with CUDA support recommended)
* Torchvision, Matplotlib, Pillow

```bash
pip install torch torchvision matplotlib pillow
```

##Run the script

```bash
python VAE.py
python visualize.py
python beta_vae.py
