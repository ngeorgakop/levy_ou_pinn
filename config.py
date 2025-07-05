import torch
import numpy as np
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set data type
DTYPE = torch.float32

# PDE Parameters
k = 0.3         # Mean reversion speed
theta = 0.0     # Mean reversion level
sigma = 0.2     # Volatility
lambda_jump = 1.0 # Jump intensity
jump_std = 0.2    # Std dev of jump size

# Domain boundaries
tmin, tmax = 0., 1.
xmin, xmax = -0.5, 2.

# Number of training points
N_T = 500   # Number of terminal condition points
N_b = 500   # Number of boundary condition points
N_r = 15000 # Number of residual points

# PD probability threshold
K_threshold = 0.0

# Training parameters
EPOCHS = 200
LEARNING_RATE = 0.001
HIDDEN_LAYERS = 6
NEURONS_PER_LAYER = 30

# Model saving path
MODEL_PATH = "ou_pinn_model_state_dict.pth"

# Derived constants
pi = torch.tensor(np.pi, dtype=DTYPE, device=device)
lb = torch.tensor([tmin, xmin], dtype=DTYPE, device=device)  # lower bounds
ub = torch.tensor([tmax, xmax], dtype=DTYPE, device=device)  # upper bounds

# Email Configuration - Load from environment variables or secrets file
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', '587'))
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER', '')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD', '')
EMAIL_USE_TLS = os.getenv('EMAIL_USE_TLS', 'True').lower() == 'true'

# Try to load from config_secrets.py if it exists
try:
    from config_secrets import *
    print("Loaded email configuration from config_secrets.py")
except ImportError:
    print("No config_secrets.py found. Using environment variables or defaults.")
    if not EMAIL_HOST_USER:
        print("Warning: Email credentials not configured. Set EMAIL_HOST_USER and EMAIL_HOST_PASSWORD environment variables or create config_secrets.py") 