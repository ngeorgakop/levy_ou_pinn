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
    'N_r': 15000,   # Number of residual points
    'N_mc': 1000    # Number of Monte Carlo anchor points
}

# PD probability threshold
K_threshold = 0.0

# Training configuration parameters
TRAINING_CONFIG = {
    'epochs': 40000,
    'lr': 0.0002,
    'print_interval': 100,
    'mc_recalc_interval': None,  # Disable MC target recalculation during training
    'early_stop_threshold': 0.000099  # Stop training when loss sum goes below this
}

# Data management configuration
DATA_CONFIG = {
    'use_saved_data': True,      # If True, try to load from saved files first
    'save_generated_data': False,  # If True, save newly generated data to files
    'data_dir': 'data'           # Directory for saving/loading training data
}

# Neural network architecture
NN_CONFIG = {
    'hidden_layers': 6,
    'neurons_per_layer': 30
}

# Integration parameters
INTEGRATION_CONFIG = {
    'num_steps': 100,
    'integration_width': 5.0  # In terms of std deviations
}

# Unpack dictionaries for direct import
k = PDE_CONFIG['k']
theta = PDE_CONFIG['theta']
sigma = PDE_CONFIG['sigma']
lambda_jump = PDE_CONFIG['lambda_jump']
jump_std = PDE_CONFIG['jump_std']

tmin = DOMAIN_CONFIG['tmin']
tmax = DOMAIN_CONFIG['tmax']
xmin = DOMAIN_CONFIG['xmin']
xmax = DOMAIN_CONFIG['xmax']

N_T = POINTS_CONFIG['N_T']
N_b = POINTS_CONFIG['N_b']
N_r = POINTS_CONFIG['N_r']
N_mc = POINTS_CONFIG['N_mc']

EPOCHS = TRAINING_CONFIG['epochs']
LEARNING_RATE = TRAINING_CONFIG['lr']
PRINT_INTERVAL = TRAINING_CONFIG['print_interval']
MC_RECALC_INTERVAL = TRAINING_CONFIG['mc_recalc_interval']
EARLY_STOP_THRESHOLD = TRAINING_CONFIG['early_stop_threshold']

USE_SAVED_DATA = DATA_CONFIG['use_saved_data']
SAVE_GENERATED_DATA = DATA_CONFIG['save_generated_data']
DATA_DIR = DATA_CONFIG['data_dir']

HIDDEN_LAYERS = NN_CONFIG['hidden_layers']
NEURONS_PER_LAYER = NN_CONFIG['neurons_per_layer']

INTEGRATION_NUM_STEPS = INTEGRATION_CONFIG['num_steps']
INTEGRATION_WIDTH = INTEGRATION_CONFIG['integration_width']

# Define lower and upper bounds for the domain
lb = torch.tensor([tmin, xmin], dtype=DTYPE, device=device)
ub = torch.tensor([tmax, xmax], dtype=DTYPE, device=device)

# Model save path
MODEL_PATH = "levy_ou_pinn_model.pth"
