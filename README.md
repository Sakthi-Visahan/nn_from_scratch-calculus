# Framework-Free Neural Network Architecture from Scratch

### Overview
This repository contains a multi-layer deep neural network built entirely from scratch using pure **NumPy** and fundamental matrix calculus. Designed to establish mathematically rigorous baselines for non-linear regression, this project completely bypasses high-level machine learning frameworks (PyTorch, TensorFlow, Keras) to demonstrate explicit forward propagation, manual backpropagation, and vectorized gradient descent.

### Tech Stack
* **Language:** Python
* **Mathematics & Computation:** NumPy
* **Visualization:** Matplotlib

### Key Technical Features
* **Zero-Framework Architecture:** Every layer, weight tensor, and bias vector is initialized and managed directly via raw NumPy arrays.
* **Vectorized Matrix Operations:** Streamlined computational efficiency by utilizing linear algebra dot products (`np.dot`) and matrix batching, eliminating Python loops during training passes.
* **Explicit Backpropagation (Chain Rule):** Explicitly derived and coded loss gradients for Mean Squared Error (MSE) and ReLU derivatives across multiple hidden layers.
* **He (Kaiming) Initialization:** Implemented Gaussian weight initialization scaled by activation layer dimensions ($np.sqrt(2 / n)$) to prevent vanishing/exploding gradients during training.

### Mathematical Formulation

1. **Forward Pass:**
   $$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$$
   $$A^{[l]} = \text{ReLU}(Z^{[l]})$$

2. **Loss Function (Mean Squared Error):**
   $$\mathcal{L} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

3. **Backpropagation (Chain Rule Updates):**
   $$\delta^{[3]} = \frac{\partial \mathcal{L}}{\partial Z^{[3]}} = \frac{2}{m} (\hat{y} - y)$$
   $$\frac{\partial \mathcal{L}}{\partial W^{[3]}} = (A^{[2]})^T \delta^{[3]}$$
   $$\delta^{[2]} = (\delta^{[3]} (W^{[3]})^T) \odot \text{ReLU}'(Z^{[2]})$$

### How to Run
Ensure you have the basic numerical packages installed:
```bash
pip install numpy matplotlib
