# Lévy-driven Ornstein-Uhlenbeck Process PINN

This package implements a Physics-Informed Neural Network (PINN) for solving the Lévy-driven Ornstein-Uhlenbeck process partial differential equation.

## Project Structure

```
levy_ou_pinn/
├── __init__.py              # Package initialization
├── config.py                # Configuration parameters and constants
├── model.py                 # Neural network model definition
├── integration.py           # Monte Carlo and trapezoidal integration methods
├── residual.py              # PDE residual computation
├── loss.py                  # Loss function computation
├── training.py              # Training loop and optimization
├── data_generation.py       # Training data generation
├── main.py                  # Main execution script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## File Descriptions

### `config.py`
Contains all configuration parameters including:
- Device settings (CPU/GPU)
- PDE parameters (k, theta, sigma, lambda_jump, jump_std)
- Domain boundaries and training point numbers
- Training hyperparameters

### `model.py`
Defines the `OU_PINN` neural network class:
- Multi-layer perceptron with tanh activation
- Sigmoid output activation for probability values
- Configurable architecture

### `integration.py`
Implements two methods for computing the Lévy integral:
- `compute_levy_integral_mc()`: Monte Carlo integration
- `compute_levy_integral_trapz()`: Trapezoidal rule integration

### `residual.py`
Contains PDE residual computation:
- `fun_r()`: Core residual function
- `get_r()`: Residual with automatic differentiation

### `loss.py`
Implements the total loss function combining:
- PDE residual loss
- Terminal condition loss
- Boundary condition loss

### `training.py`
Contains the training loop with:
- Adam optimizer
- Progress monitoring
- Loss history tracking

### `data_generation.py`
Generates training data including:
- Terminal condition points
- Boundary condition points
- Residual/collocation points

### `main.py`
Main execution script that orchestrates the entire training process.

## Usage

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Training

Execute the main script:
```bash
python main.py
```

### Using Individual Components

```python
from levy_ou_pinn import OU_PINN, generate_training_data, train
from levy_ou_pinn.config import device, k, theta, sigma, lambda_jump, jump_std

# Generate data
X_r, X_data, u_data = generate_training_data()

# Initialize model
model = OU_PINN(hidden_layers=6, neurons_per_layer=30).to(device)

# Train model
loss_history = train(model, X_r, X_data, u_data, k, theta, sigma, lambda_jump, jump_std)
```

## Key Features

- **Modular Design**: Each component is separated for maintainability
- **GPU Support**: Automatic GPU detection and usage
- **Flexible Integration**: Choice between Monte Carlo and trapezoidal integration
- **Comprehensive Loss**: Includes PDE, terminal, and boundary condition losses
- **Progress Monitoring**: Real-time training progress and loss visualization

## PDE Formulation

The code solves the Lévy-driven OU process PDE:

```
∂φ/∂t + k(θ-x)∂φ/∂x + (σ²/2)∂²φ/∂x² + λ∫[φ(t,x+z) - φ(t,x)]p(z)dz = 0
```

Where:
- φ(t,x) is the probability function
- k: mean reversion speed
- θ: mean reversion level  
- σ: volatility
- λ: jump intensity
- p(z): Normal distribution PDF for jump sizes

## Model Output

The trained model outputs the probability P(X_T ≤ K | X_t = x) for the Lévy-driven OU process. 