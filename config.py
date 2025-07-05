import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set data type
DTYPE = torch.float32

# PDE Parameters
PDE_CONFIG = {
    'k': 0.3,             # Mean reversion speed
    'theta': 0.0,         # Mean reversion level
    'sigma': 0.2,         # Volatility
    'lambda_jump': 1.0,   # Jump intensity
    'jump_std': 0.2       # Std dev of jump size
}

# Domain boundaries
DOMAIN_CONFIG = {
    'tmin': 0.0, 
    'tmax': 1.0,
    'xmin': -0.5, 
    'xmax': 2.0
}

# Number of training points
POINTS_CONFIG = {
    'N_T': 500,    # Number of terminal condition points
    'N_b': 500,    # Number of boundary condition points
    'N_r': 15000   # Number of residual points
}

# PD probability threshold
K_threshold = 0.0

# Training configuration parameters
TRAINING_CONFIG = {
    'epochs': 30000,
    'lr': 0.005,
    'print_interval': 100
}

# Scheduler parameters
SCHEDULER_CONFIG = {
    'step_size': 4000,
    'gamma': 0.5
}

# Neural network architecture
NN_CONFIG = {
    'hidden_layers': 6,
    'neurons_per_layer': 30
}
