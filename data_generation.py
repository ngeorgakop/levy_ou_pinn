import torch
import os
from pathlib import Path

from config import (
    DTYPE, device, tmin, tmax, xmin, xmax, 
    N_T, N_b, N_r, K_threshold, lb, ub, N_mc
)


def save_training_data(X_r, X_data, u_data, X_mc, data_dir="data"):
    """
    Save training data to files in the specified directory.
    
    Args:
        X_r: Residual/collocation points
        X_data: List of [terminal_points, boundary_points]
        u_data: List of [terminal_targets, boundary_targets] 
        X_mc: Monte Carlo anchor points
        data_dir: Directory to save data files
    """
    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving training data to '{data_dir}/' directory...")
    
    # Save residual points
    torch.save(X_r, os.path.join(data_dir, "X_r.pt"))
    print(f"  - Saved X_r.pt: {X_r.shape}")
    
    # Save terminal condition data
    torch.save(X_data[0], os.path.join(data_dir, "X_terminal.pt"))
    torch.save(u_data[0], os.path.join(data_dir, "u_terminal.pt"))
    print(f"  - Saved X_terminal.pt: {X_data[0].shape}")
    print(f"  - Saved u_terminal.pt: {u_data[0].shape}")
    
    # Save boundary condition data
    torch.save(X_data[1], os.path.join(data_dir, "X_boundary.pt"))
    torch.save(u_data[1], os.path.join(data_dir, "u_boundary.pt"))
    print(f"  - Saved X_boundary.pt: {X_data[1].shape}")
    print(f"  - Saved u_boundary.pt: {u_data[1].shape}")
    
    # Save Monte Carlo points
    torch.save(X_mc, os.path.join(data_dir, "X_mc.pt"))
    print(f"  - Saved X_mc.pt: {X_mc.shape}")
    
    # Save metadata
    metadata = {
        'N_T': N_T,
        'N_b': N_b, 
        'N_r': N_r,
        'N_mc': N_mc,
        'K_threshold': K_threshold,
        'domain': {'tmin': tmin, 'tmax': tmax, 'xmin': xmin, 'xmax': xmax},
        'bounds': {'lb': lb, 'ub': ub}
    }
    torch.save(metadata, os.path.join(data_dir, "metadata.pt"))
    print(f"  - Saved metadata.pt")
    
    print("Training data saved successfully!")


def load_training_data(data_dir="data"):
    """
    Load training data from files in the specified directory.
    
    Args:
        data_dir: Directory to load data files from
        
    Returns:
        tuple: (X_r, X_data, u_data, X_mc) in the same format as generate_training_data()
    """
    print(f"\nLoading training data from '{data_dir}/' directory...")
    
    # Check if all required files exist
    required_files = ["X_r.pt", "X_terminal.pt", "u_terminal.pt", 
                     "X_boundary.pt", "u_boundary.pt", "X_mc.pt", "metadata.pt"]
    
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required training data file not found: {file_path}")
    
    # Load data files
    X_r = torch.load(os.path.join(data_dir, "X_r.pt"), map_location=device)
    X_terminal = torch.load(os.path.join(data_dir, "X_terminal.pt"), map_location=device)
    u_terminal = torch.load(os.path.join(data_dir, "u_terminal.pt"), map_location=device)
    X_boundary = torch.load(os.path.join(data_dir, "X_boundary.pt"), map_location=device)
    u_boundary = torch.load(os.path.join(data_dir, "u_boundary.pt"), map_location=device)
    X_mc = torch.load(os.path.join(data_dir, "X_mc.pt"), map_location=device)
    metadata = torch.load(os.path.join(data_dir, "metadata.pt"), map_location=device)
    
    # Enable gradient calculation for residual points
    X_r.requires_grad_(True)
    
    # Reconstruct data lists
    X_data = [X_terminal, X_boundary]
    u_data = [u_terminal, u_boundary]
    
    print(f"  - Loaded X_r.pt: {X_r.shape}")
    print(f"  - Loaded X_terminal.pt: {X_terminal.shape}")
    print(f"  - Loaded u_terminal.pt: {u_terminal.shape}")
    print(f"  - Loaded X_boundary.pt: {X_boundary.shape}")
    print(f"  - Loaded u_boundary.pt: {u_boundary.shape}")
    print(f"  - Loaded X_mc.pt: {X_mc.shape}")
    print(f"  - Loaded metadata: N_T={metadata['N_T']}, N_b={metadata['N_b']}, N_r={metadata['N_r']}, N_mc={metadata['N_mc']}")
    
    # Verify metadata matches current config
    if (metadata['N_T'] != N_T or metadata['N_b'] != N_b or 
        metadata['N_r'] != N_r or metadata['N_mc'] != N_mc):
        print("WARNING: Loaded data parameters don't match current config:")
        print(f"  Loaded: N_T={metadata['N_T']}, N_b={metadata['N_b']}, N_r={metadata['N_r']}, N_mc={metadata['N_mc']}")
        print(f"  Current: N_T={N_T}, N_b={N_b}, N_r={N_r}, N_mc={N_mc}")
    
    print("Training data loaded successfully!")
    
    return X_r, X_data, u_data, X_mc


def check_training_data_exists(data_dir="data"):
    """
    Check if training data files exist in the specified directory.
    
    Args:
        data_dir: Directory to check for data files
        
    Returns:
        bool: True if all required files exist, False otherwise
    """
    required_files = ["X_r.pt", "X_terminal.pt", "u_terminal.pt",
                     "X_boundary.pt", "u_boundary.pt", "X_mc.pt", "metadata.pt"]
    
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            return False
    return True


def get_training_data(use_saved_data=False, data_dir="data", save_generated_data=True):
    """
    Get training data by either loading from files or generating new data.
    
    Args:
        use_saved_data: If True, try to load from saved files first
        data_dir: Directory for data files
        save_generated_data: If True, save newly generated data to files
        
    Returns:
        tuple: (X_r, X_data, u_data, X_mc) 
    """
    if use_saved_data and check_training_data_exists(data_dir):
        return load_training_data(data_dir)
    else:
        if use_saved_data:
            print(f"Saved training data not found in '{data_dir}/', generating new data...")
        
        # Generate new training data
        X_r, X_data, u_data, X_mc = generate_training_data()
        
        # Save the generated data if requested
        if save_generated_data:
            save_training_data(X_r, X_data, u_data, X_mc, data_dir)
        
        return X_r, X_data, u_data, X_mc


def generate_training_data():
    """
    Generate training data for the PINN including terminal conditions,
    boundary conditions, and residual points.
    
    Returns:
        tuple: (X_r, X_data, u_data, X_mc) where:
            - X_r: Residual/collocation points
            - X_data: List of [terminal_points, boundary_points] 
            - u_data: List of [terminal_targets, boundary_targets]
            - X_mc: Monte Carlo anchor points
    """
    print(f"Domain: t in [{tmin}, {tmax}], x in [{xmin}, {xmax}]")
    print(f"Points: N_T={N_T}, N_b={N_b}, N_r={N_r}, N_mc={N_mc}")
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

    # --- 4. Monte Carlo anchor points (interior) ---
    t_mc = torch.rand((N_mc, 1), dtype=DTYPE, device=device) * (ub[0] - lb[0]) + lb[0]
    x_mc = torch.rand((N_mc, 1), dtype=DTYPE, device=device) * (ub[1] - lb[1]) + lb[1]
    X_mc = torch.cat([t_mc, x_mc], dim=1)
    print(f"Generated X_mc shape: {X_mc.shape}")
    
    # --- Combine data lists for training ---
    X_data = [X_T, X_b]  # Terminal and Boundary coordinate tensors
    u_data = [u_T, u_b]  # Corresponding target value tensors
    
    print("\nData Generation Complete.")
    print(f"X_data contains {len(X_data)} tensors (Terminal, Boundary)")
    print(f"u_data contains {len(u_data)} tensors (Terminal Targets, Boundary Targets)")
    print(f"X_r tensor shape: {X_r.shape} (Residual Points)")
    print(f"X_mc tensor shape: {X_mc.shape} (Monte Carlo Points)")
    
    return X_r, X_data, u_data, X_mc 