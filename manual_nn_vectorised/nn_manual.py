import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SYNTHETIC DATA GENERATION (STRUCTURAL RESPONSE)
# ==========================================
# Generating a non-linear dataset simulating a damped structural resonance curve
np.random.seed(42)
X = np.linspace(0, 10, 200).reshape(-1, 1)
# True physical signal with some injected noise
y_true = np.sin(X) * np.exp(-0.2 * X) 
y_noisy = y_true + np.random.normal(0, 0.05, X.shape)

# Standardize inputs for stable gradient descent
X_scaled = (X - np.mean(X)) / np.std(X)

# ==========================================
# 2. NEURAL NETWORK ARCHITECTURE FROM SCRATCH
# ==========================================
class NumPyNet:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        
        # He Initialization (Optimal for ReLU activation)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        
        # Output layer weights
        self.W3 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b3 = np.zeros((1, output_size))

    # --- ACTIVATION FUNCTIONS ---
    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    # --- FORWARD PROPAGATION ---
    def forward(self, X):
        # Layer 1
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        # Layer 2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)
        
        # Output Layer (Linear Activation for Regression)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.Z3 
        
        return self.A3

    # --- BACKPROPAGATION (PURE MATRIX CALCULUS) ---
    def backward(self, X, y):
        m = X.shape[0] # Number of samples
        
        # 1. Derivative of Mean Squared Error (MSE) Loss
        dZ3 = (self.A3 - y) / m
        
        # 2. Output Layer Gradients
        dW3 = np.dot(self.A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        
        # 3. Hidden Layer 2 Gradients (Applying Chain Rule)
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self.relu_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # 4. Hidden Layer 1 Gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # 5. Gradient Descent Weight Updates
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # --- TRAINING LOOP ---
    def train(self, X, y, epochs):
        loss_history = []
        for epoch in range(epochs):
            predictions = self.forward(X)
            self.backward(X, y)
            
            # Calculate MSE Loss
            loss = np.mean(np.square(predictions - y))
            loss_history.append(loss)
            
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1:05d} | MSE Loss: {loss:.6f}")
                
        return loss_history

# ==========================================
# 3. EXECUTION & VISUALIZATION
# ==========================================
print("Initializing Custom NumPy Neural Network...")
# Architecture: 1 Input -> 32 Hidden -> 32 Hidden -> 1 Output
nn_model = NumPyNet(input_size=1, hidden_size=32, output_size=1, learning_rate=0.05)

print("Starting Training Loop...")
losses = nn_model.train(X_scaled, y_noisy, epochs=10000)

# Generate final predictions for plotting
final_predictions = nn_model.forward(X_scaled)

# Plotting the results
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

# Plot 1: Model Fit
ax1.scatter(X, y_noisy, color='dodgerblue', alpha=0.4, label='Noisy Sensor Data')
ax1.plot(X, y_true, color='white', linestyle='--', label='True Ground Signal')
ax1.plot(X, final_predictions, color='crimson', linewidth=3, label='NumPy NN Prediction')
ax1.set_title("Custom NumPy Network vs. Non-Linear Data", fontweight='bold')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
ax1.legend(facecolor='#111111', edgecolor='#444444')
ax1.grid(color='#333333', linestyle='-', linewidth=0.5)

# Plot 2: Convergence Curve
ax2.plot(losses, color='limegreen', linewidth=2)
ax2.set_title("Gradient Descent Convergence", fontweight='bold')
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Mean Squared Error (MSE)")
ax2.grid(color='#333333', linestyle='-', linewidth=0.5)
ax2.set_yscale('log')

plt.tight_layout()
plt.show()