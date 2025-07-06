import torch
from residual import get_r
from integration import simulate_ou_paths
from config import xmin, xmax, K_threshold, tmax, DTYPE, device


def compute_loss(model, X_r, X_data, u_data, k, theta, sigma, lambda_jump, jump_std):
    """
    Compute the total loss for training the PINN.
    
    Args:
        model: Neural network model
        X_r: Residual/collocation points
        X_data: List of boundary/terminal condition points
        u_data: List of target values for boundary/terminal conditions
        k, theta, sigma: OU process parameters
        lambda_jump, jump_std: Jump parameters
        
    Returns:
        torch.Tensor: Total loss
    """
    # Extract interior points for PDE residual
    t_r_interior, x_r_interior = X_r[:, 0:1], X_r[:, 1:2]

    # Compute PDE residual loss
    pde_residual = get_r(model, t_r_interior, x_r_interior, k, theta, sigma, lambda_jump, jump_std)
    loss_interior = torch.mean(pde_residual**2)

    # Terminal condition loss
    X_terminal_cond_points = X_data[0]
    u_terminal_target_values = u_data[0]
    u_terminal_pred = model(X_terminal_cond_points)
    loss_terminal = torch.mean((u_terminal_target_values - u_terminal_pred) ** 2)

    # Original Boundary condition loss (vs. 1 at xmin, 0 at xmax)
    X_boundary_cond_points = X_data[1]
    u_boundary_target_values = u_data[1]
    u_boundary_pred = model(X_boundary_cond_points)
    loss_boundary = torch.mean((u_boundary_target_values - u_boundary_pred) ** 2)

    # --- Monte Carlo Boundary Loss (for xmax points only) ---
    X_b = X_data[1]
    is_xmax_mask = (X_b[:, 1] == xmax)
    X_b_xmax = X_b[is_xmax_mask]
    
    loss_boundary_mc = torch.tensor(0.0, device=device, dtype=DTYPE)
    if X_b_xmax.shape[0] > 0:
        u_b_xmax_pred = model(X_b_xmax)
        
        with torch.no_grad():
            u_b_xmax_target_mc = simulate_ou_paths(
                start_x=X_b_xmax[:, 1:2],
                start_t=X_b_xmax[:, 0:1],
                k=k, theta=theta, sigma=sigma,
                K_threshold=K_threshold,
                t_max=tmax
            )
        
        loss_boundary_mc = torch.mean((u_b_xmax_pred - u_b_xmax_target_mc)**2)


    # Total loss
    total_loss = 10 * loss_interior + loss_terminal + loss_boundary + loss_boundary_mc

    return total_loss, loss_interior, loss_terminal, loss_boundary, loss_boundary_mc 