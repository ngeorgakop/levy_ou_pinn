import torch

from config import (
    DTYPE, device, tmin, tmax, xmin, xmax, 
    N_T, N_b, N_r, K_threshold, lb, ub
)


def generate_training_data():
    """
    Generate training data for the PINN including terminal conditions,
    boundary conditions, and residual points.
    
    Returns:
        tuple: (X_r, X_data, u_data) where:
            - X_r: Residual/collocation points
            - X_data: List of [terminal_points, boundary_points] 
            - u_data: List of [terminal_targets, boundary_targets]
    """
    print(f"Domain: t in [{tmin}, {tmax}], x in [{xmin}, {xmax}]")
    print(f"Points: N_T={N_T}, N_b={N_b}, N_r={N_r}")
    print(f"Default Threshold K = {K_threshold}")
    
    # --- 1. Terminal Condition points (t=tmax) ---
    # Time coordinates are fixed at tmax
    t_T = torch.full((N_T, 1), tmax, dtype=DTYPE, device=device)
    # Spatial coordinates sampled uniformly within [xmin, xmax]
    x_T = torch.linspace(xmin, xmax, N_T, dtype=DTYPE, device=device).unsqueeze(1)
    # Combine t and x coordinates
    X_T = torch.cat([t_T, x_T], dim=1)
    # Calculate target values: 1.0 if x <= K, 0.0 otherwise
    u_T = (x_T <= K_threshold).to(DTYPE)  # Convert boolean to float
    
    print(f"Generated X_T shape: {X_T.shape}, u_T shape: {u_T.shape}")
    
    # --- 2. Boundary Condition points ---
    # Time coordinates sampled uniformly within [tmin, tmax]
    t_b = torch.rand((N_b, 1), dtype=DTYPE, device=device) * (ub[0] - lb[0]) + lb[0]
    
    # Spatial coordinates: N_b/2 points at xmin, N_b/2 points at xmax
    num_xmin = N_b // 2
    num_xmax = N_b - num_xmin
    x_b_min = torch.full((num_xmin, 1), xmin, dtype=DTYPE, device=device)
    x_b_max = torch.full((num_xmax, 1), xmax, dtype=DTYPE, device=device)
    x_b_combined = torch.cat([x_b_min, x_b_max], dim=0)
    
    # Target values: 1 at xmin, 0 at xmax
    u_b_min = torch.ones((num_xmin, 1), dtype=DTYPE, device=device)
    u_b_max = torch.zeros((num_xmax, 1), dtype=DTYPE, device=device)
    u_b_combined = torch.cat([u_b_min, u_b_max], dim=0)
    
    # Shuffle the points so xmin/xmax are mixed relative to time
    perm = torch.randperm(N_b)
    x_b = x_b_combined[perm]
    u_b = u_b_combined[perm]  # Apply same permutation to targets
    
    # Combine t and x coordinates
    X_b = torch.cat([t_b, x_b], dim=1)
    
    print(f"Generated X_b shape: {X_b.shape}, u_b shape: {u_b.shape}")
    
    # --- 3. Residual points (collocation points) ---
    # Time coordinates sampled uniformly
    t_r = torch.rand((N_r, 1), dtype=DTYPE, device=device) * (ub[0] - lb[0]) + lb[0]
    # Spatial coordinates sampled uniformly
    x_r = torch.rand((N_r, 1), dtype=DTYPE, device=device) * (ub[1] - lb[1]) + lb[1]
    # Combine t and x coordinates
    X_r = torch.cat([t_r, x_r], dim=1)
    
    # Enable gradient calculation for residual points
    X_r.requires_grad_(True)
    
    print(f"Generated X_r shape: {X_r.shape}")
    
    # --- Combine data lists for training ---
    X_data = [X_T, X_b]  # Terminal and Boundary coordinate tensors
    u_data = [u_T, u_b]  # Corresponding target value tensors
    
    print("\nData Generation Complete.")
    print(f"X_data contains {len(X_data)} tensors (Terminal, Boundary)")
    print(f"u_data contains {len(u_data)} tensors (Terminal Targets, Boundary Targets)")
    print(f"X_r tensor shape: {X_r.shape} (Residual Points)")
    
    return X_r, X_data, u_data 